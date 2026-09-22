import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from typing import Sequence

from app.src.data.building.feature_engineering import (
    load_feature_matrix, 
    load_supervised_feature_matrix
)
from . import CalibrationResult, ScoresStats, TuneTemperatureResult
from .anomaly_utils import get_threshold
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.mtae import MTTabularAE
from app.src.ml.training.evaluate_ae import reconstruction
from app.src.ml.training.evaluate_mt import prediction


def calibrate_threshold(
    country: str,
    model: TabularAE | TabularVAE | MTTabularAE,
    scaler: RobustScaler,
    device: str,
    loss_weights: dict[str, float],
    attack_type_weights: torch.Tensor | None,
    pred_quantiles: Sequence[float] | None,
    method: str = "p99",
    cw: int = 30,
    tune_temperature: bool = True,
) -> CalibrationResult:
    """
    Computes anomaly threshold with optional temperature scaling for specified
    calibration window and method.
    """
    # ------------------------------------
    # Set up tracking
    # ------------------------------------
    tracker = CalibrationResult(
        country=country,
        device=device,
        cont_w=loss_weights["cont_w"],
        cat_w=loss_weights["cat_w"],
        method=method,
        cal_window_days=cw,
    )
    
    # ------------------------------------
    # Build features
    # ------------------------------------
    if isinstance(model, MTTabularAE):
        fmatrix = load_supervised_feature_matrix(country)
    else:
        fmatrix = load_feature_matrix(country)

    Xc = fmatrix.X_cont.to_numpy(dtype=np.float32)
    Xk = fmatrix.X_cat.to_numpy(dtype=np.int64)
    ts = fmatrix.X_cont.index
    if (
        isinstance(model, MTTabularAE)
        and fmatrix.y_l3 is not None
        and fmatrix.y_l7 is not None
        and fmatrix.y_at is not None
    ):
        y3 = fmatrix.y_l3.to_numpy(dtype=np.float32)
        y7 = fmatrix.y_l7.to_numpy(dtype=np.float32)
        ya = fmatrix.y_at.to_numpy(dtype=np.int64)

    # ------------------------------------
    # Select calibration window
    # ------------------------------------ 
    end_time = ts.max()
    start_time = end_time - pd.Timedelta(days=cw)
    cal_window = (ts >= start_time)

    Xc_cal = Xc[cal_window]
    Xk_cal = Xk[cal_window]
    if isinstance(model, MTTabularAE):
        y3_cal = y3[cal_window]
        y7_cal = y7[cal_window]
        ya_cal = ya[cal_window]

    print(f"[INFO] Calibration window: {start_time} to {end_time}")
    print(f"[INFO] Calibration samples: {len(Xc_cal)}")

    # ------------------------------------
    # Apply scaler
    # ------------------------------------ 
    Xc_cal_scld = scaler.transform(Xc_cal).astype(np.float32)

    # ------------------------------------
    # Tune temperature scaling 
    # ------------------------------------
    best_temp = 1.0
    temp_results: dict[str, dict[str, float | int]] = {}
    if tune_temperature and len(Xc_cal) > 100:
        print(f"[INFO] Tuning inference temperature...")
        
        temperature_range = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        
        best_temp_metric = float('inf')
        for temp in temperature_range:
            if (
                isinstance(model, MTTabularAE) 
                and attack_type_weights is not None 
                and pred_quantiles is not None
            ):
                temp_scores = prediction(
                    model,
                    Xc_cal_scld,
                    Xk_cal,
                    y3_cal,
                    y7_cal, 
                    ya_cal,
                    loss_weights,
                    attack_type_weights,
                    pred_quantiles,
                    temp,
                    device,
                    calibration=True
                ).scores
            elif isinstance(model, (TabularAE, TabularVAE)):
                temp_scores = reconstruction(
                    model,
                    Xc_cal_scld,
                    Xk_cal,
                    loss_weights,
                    temp,
                    device
                )

            temp_prelim = get_threshold(method, temp_scores)
            temp_clean_scores = temp_scores[temp_scores < temp_prelim]
            temp_metric = np.median(temp_clean_scores) if len(temp_clean_scores) > 0 else np.median(temp_scores)

            result = TuneTemperatureResult(
                scores_mean=float(temp_scores.mean()),
                scores_std=float(temp_scores.std()),
                scores_median=float(np.median(temp_scores)),
                clean_scores_median=float(temp_metric),
                prelim_threshold=float(temp_prelim),
                clean_samples=len(temp_clean_scores),
            )
            temp_results[f"{temp}"] = result.to_dict()

            stability_score = result.scores_std / (result.scores_mean + 1e-8)
            extreme_penalty = abs(temp - 1.0) * 0.1
            combined_metric = temp_metric * (1 + stability_score * 0.1 + extreme_penalty)
        
            if combined_metric < best_temp_metric:
                best_temp_metric = combined_metric
                best_temp = temp
        
        print(f"[CAL] Computed optimal inference temperature in cw: {best_temp:.1f}")

        # ------------------------------------
        # Track temperature tuning
        # ------------------------------------
        tracker["temp_training"] = model.config.temperature
        tracker["temp_inference"] = best_temp
        tracker["temp_results"] = temp_results
        tracker["temp_range"] = temperature_range
    else:
        print(f"[INFO] Inference temperature set to config default: {model.config.temperature}")
        best_temp = model.config.temperature

    # ------------------------------------
    # Compute recon error in window
    # ------------------------------------ 
    if (
        isinstance(model, MTTabularAE) 
        and attack_type_weights is not None 
        and pred_quantiles is not None
    ):
        scores = prediction(
            model,
            Xc_cal_scld,
            Xk_cal,
            y3_cal,
            y7_cal, 
            ya_cal,
            loss_weights,
            attack_type_weights,
            pred_quantiles,
            best_temp,
            device,
            calibration=True
        ).scores
    elif isinstance(model, (TabularAE, TabularVAE)):
        scores = reconstruction(
            model,
            Xc_cal_scld,
            Xk_cal,
            loss_weights,
            best_temp,
            device
        )
    print(f"[CAL] Computed {len(scores)} scores")

    # ------------------------------------
    # Remove spikes/anomalies in window
    # ------------------------------------ 
    prelim = get_threshold(method, scores)
    clean_scores = scores[scores < prelim]
    removed_count = len(scores) - len(clean_scores)
    print(f"[CAL] Removed {removed_count} preliminary anomalies to clean window")

    if len(clean_scores) < 10:
        print(f"[WARN] Very few clean samples ({len(clean_scores)}). Using all scores")
        clean_scores = scores

    # ------------------------------------
    # Compute threshold on cleaned window
    # ------------------------------------
    threshold = get_threshold(method, clean_scores)
    print(f"[CAL] Computed threshold ({method}) = {threshold:.6f}")

    anomaly_rate = float(np.mean(scores > threshold) * 100)
    print(f"[CAL] Expected anomaly rate: {anomaly_rate:.2f}%")

    # ------------------------------------
    # Track threshold calibration
    # ------------------------------------
    scores_stats = ScoresStats(
        min=float(scores.min()),
        max=float(scores.max()),
        median=float(np.median(scores)),
        mean=float(scores.mean()),
        std=float(scores.std())
    )
    scores_stats[f"{method}"] = get_threshold(method, scores)
    tracker["threshold"] = threshold
    tracker["cal_samples"] = len(Xc_cal)
    tracker["clean_samples"] = len(clean_scores)
    tracker["prelim_anom_removed"] = removed_count
    tracker["scores_stats"] = scores_stats.to_dict()
    tracker["pred_anom_rate"] = anomaly_rate

    return tracker
