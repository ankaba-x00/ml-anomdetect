import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin

from app.src.data.feature_engineering import (
    load_feature_matrix, 
    load_supervised_feature_matrix
)
from .anomaly_utils import get_threshold
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.mtae import MTTabularAE
from app.src.ml.training.evaluate_ae import reconstruction
from app.src.ml.training.evaluate_mt import prediction


def calibrate_threshold(
    country: str,
    model: TabularAE | TabularVAE | MTTabularAE,
    scaler: TransformerMixin,
    device: str,
    loss_weights: dict[str, float],
    attack_type_weights: torch.Tensor | None,
    pred_quantiles: list[float] | None,
    beta: float,
    cw: int = 30,
    method: str = "p99",
    tune_temperature: bool = True
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray |float]]:
    """Computes anomaly threshold with optional temperature scaling for specified calibration window and method."""
    
    # ------------------------------------
    # Build features
    # ------------------------------------
    if isinstance(model, MTTabularAE):
        X_cont, X_cat, y_l3, y_l7, y_at, _, _ = load_supervised_feature_matrix(country)
    else:
        X_cont, X_cat, _, _ = load_feature_matrix(country)

    Xc = X_cont.values.astype(np.float32)
    Xk = X_cat.values.astype(np.int64)
    ts = X_cont.index
    if isinstance(model, MTTabularAE):
        y3 = y_l3.values.astype(np.float32)
        y7 = y_l7.values.astype(np.float32)
        ya = y_at.values.astype(np.int64)

    # ------------------------------------
    # Select calibration window
    # ------------------------------------ 
    end_time = ts.max()
    start_time = end_time - pd.Timedelta(days=cw)
    cal_window = (ts >= start_time)

    Xc_cal = Xc[cal_window]
    Xk_cal = Xk[cal_window]
    ts_cal = ts[cal_window]
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
    temperature_tuned = False
    temp_results = {}
    if tune_temperature and len(Xc_cal) > 100:
        print(f"[INFO] Tuning inference temperature...")
        
        temperature_range = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        
        best_temp_metric = float('inf')
        for temp in temperature_range:
            if isinstance(model, MTTabularAE):
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
                )
            else:
                temp_scores = reconstruction(
                    model,
                    Xc_cal_scld,
                    Xk_cal,
                    loss_weights,
                    temp,
                    beta,
                    device
                )
            
            temp_prelim = get_threshold(method, temp_scores)
            temp_clean_scores = temp_scores[temp_scores < temp_prelim]
            temp_metric = np.median(temp_clean_scores) if len(temp_clean_scores) > 0 else np.median(temp_scores)
            temp_results[temp] = {
                "scores_mean": float(temp_scores.mean()),
                "scores_std": float(temp_scores.std()),
                "scores_median": float(np.median(temp_scores)),
                "clean_scores_median": float(temp_metric),
                "prelim_threshold": float(temp_prelim),
                "clean_samples": len(temp_clean_scores),
            }
            stability_score = temp_results[temp]["scores_std"] / (temp_results[temp]["scores_mean"] + 1e-8)
            extreme_penalty = abs(temp - 1.0) * 0.1
            combined_metric = temp_metric * (1 + stability_score * 0.1 + extreme_penalty)
        
            if combined_metric < best_temp_metric:
                best_temp_metric = combined_metric
                best_temp = temp
        
        print(f"[CAL] Computed optimal inference temperature in cw: {best_temp:.1f}")
        temperature_tuned = True
    else:
        print(f"[INFO] Inference temperature set to config default: {model.config.temperature}")
        best_temp = model.config.temperature

    # ------------------------------------
    # Compute recon error in window
    # ------------------------------------ 
    if isinstance(model, MTTabularAE):
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
        )
    else:
        scores = reconstruction(
            model,
            Xc_cal_scld,
            Xk_cal,
            loss_weights,
            best_temp,
            beta,
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

    anomaly_rate = np.mean(scores > threshold) * 100
    print(f"[CAL] Expected anomaly rate: {anomaly_rate:.2f}%")

    # ------------------------------------
    # Return results
    # ------------------------------------
    threshold_dict = {
        "country": country,
        "device": device, 
        "method": method, 
        "threshold": threshold,
        "calibration_window_days": cw,
        "calibration_samples": len(Xc_cal),
        "clean_samples": len(clean_scores),
        "preliminary_anomalies_removed": removed_count,
        "cont_w": loss_weights["cont_w"],
        "cat_w": loss_weights["cat_w"],
        "scores_stats": {
            "min": float(scores.min()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "max": float(scores.max()),
            "std": float(scores.std()),
            "p99": float(np.percentile(scores, 99)),
        },
        "anomaly_rate_pct": float(anomaly_rate),
        "temperature_training": 1.0,
        "temperature_inference": best_temp,
        "temperature_tuned": temperature_tuned,
        "temperature_results": temp_results if tune_temperature else {},
        "temperature_range": temperature_range if tune_temperature else [],
    }
    
    debug_dict = {
        "window": ts_cal, 
        "scores": scores,
        "clean_scores": clean_scores,
        "preliminary_threshold": float(prelim),
    }

    return threshold_dict, debug_dict
