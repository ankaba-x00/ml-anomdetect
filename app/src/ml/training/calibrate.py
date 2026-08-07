import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

from app.src.data.feature_engineering import load_feature_matrix
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.training.evaluate_ae import reconstruction
from .core.anomaly_utils import get_threshold


# TODO: benchmark threshold calibration window 7 to 30 days 
def calibrate_threshold(
    country: str,
    model: TabularAE | TabularVAE,
    scaler: TransformerMixin,
    device: str = "cpu",
    method: str = "p99",
    cw: int | None = 30,
    cont_w: float = 1.0,
    cat_w: float = 0.0, 
    tune_temperature: bool = True,
    temperature_range: list | None = None,
    use_mc_elbo: bool = False,
    beta: float = 1.0
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray |float]]:
    """
    Computes anomaly threshold and temperature scaling for a given model on 
    a specified calibration window, method and temperature range.
    """
    
    # ------------------------------------
    # Build raw feature matrix
    # ------------------------------------ 
    X_cont, X_cat, _, _ = load_feature_matrix(country)
    Xc_np = X_cont.values.astype(np.float64)
    Xk_np = X_cat.values.astype(np.int64)
    ts = X_cont.index

    # ------------------------------------
    # Select calibration window
    # ------------------------------------ 
    end_time = ts.max()
    start_time = end_time - pd.Timedelta(days=cw)
    cal_window = (ts >= start_time)

    X_cont_cal = Xc_np[cal_window]
    X_cat_cal = Xk_np[cal_window]
    ts_cal = ts[cal_window]

    print(f"\n[INFO] Calibration window: {start_time} to {end_time}")
    print(f"[INFO] Calibration samples: {len(X_cont_cal)}")

    # ------------------------------------
    # Apply scaler from fit_transforming full data on cont data
    # ------------------------------------ 
    X_cont_cal_scld = scaler.transform(X_cont_cal).astype(np.float32)

    # ------------------------------------
    # Tune inference temperature scaling 
    # ------------------------------------
    best_temp = 1.0
    temperature_tuned = False
    temp_results = {}
    if tune_temperature and len(X_cont_cal) > 100:
        print(f"[INFO] Tuning inference temperature...")
        
        if temperature_range is None:
            temperature_range = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        
        best_temp_metric = float('inf')
        for temp in temperature_range:
            temp_scores = reconstruction(
                model,
                X_cont_cal_scld,
                X_cat_cal,
                device,
                cont_w,
                cat_w,
                use_mc_elbo,
                temp,
                beta
            )
            
            # Compute preliminary threshold with this temperature
            temp_prelim = get_threshold(method, temp_scores)
            temp_clean_scores = temp_scores[temp_scores < temp_prelim]
            temp_metric = np.median(temp_clean_scores) if len(temp_clean_scores) > 0 else np.median(temp_scores)
            temp_results[temp] = {
                "scores_mean": float(temp_scores.mean()),
                "scores_std": float(temp_scores.std()),
                "scores_median": float(np.median(temp_scores)),
                "clean_scores_median": float(np.median(temp_clean_scores)) if len(temp_clean_scores) > 0 else float(np.median(temp_scores)),
                "prelim_threshold": float(temp_prelim),
                "clean_samples": len(temp_clean_scores),
            }
            stability_score = temp_results[temp]["scores_std"] / (temp_results[temp]["scores_mean"] + 1e-8)
            extreme_penalty = abs(temp - 1.0) * 0.1
            combined_metric = temp_metric * (1 + stability_score * 0.1 + extreme_penalty)
        
            if combined_metric < best_temp_metric:
                best_temp_metric = combined_metric
                best_temp = temp
        
        print(f"[CAL] Computed optimal inference temperature in cw: {best_temp:.3f}")
        temperature_tuned = True
    else:
        best_temp = 1.0

    # ------------------------------------
    # Compute recon error in window
    # ------------------------------------ 
    scores = reconstruction(
        model,
        X_cont_cal_scld,
        X_cat_cal,
        device,
        cont_w,
        cat_w,
        use_mc_elbo,
        best_temp,
        beta
    )
    print(f"[INFO] Computed {len(scores)} scores with temp={best_temp}")

    # ------------------------------------
    # Remove spikes/anomalies in window
    # ------------------------------------ 
    prelim = get_threshold(method, scores)
    clean_scores = scores[scores < prelim]
    removed_count = len(scores) - len(clean_scores)
    print(f"[INFO] Removed {removed_count} preliminary anomalies to clean window")

    if len(clean_scores) < 10:
        print(f"[WARN] Very few clean samples ({len(clean_scores)}). Using all scores.")
        clean_scores = scores

    # ------------------------------------
    # Compute threshold on cleaned window
    # ------------------------------------
    threshold = get_threshold(method, clean_scores)
    print(f"[CAL] Computed threshold ({method}) = {threshold:.6f}")

    anomaly_rate = np.mean(scores > threshold) * 100
    print(f"[INFO] Expected anomaly rate: {anomaly_rate:.2f}%")

    # ------------------------------------
    # Output
    # ------------------------------------
    threshold_dict = {
        "country": country,
        "device": device, 
        "method": method, 
        "threshold": threshold,
        "calibration_window_days": cw,
        "calibration_samples": len(X_cont_cal),
        "clean_samples": len(clean_scores),
        "preliminary_anomalies_removed": removed_count,
        "cont_w": cont_w,
        "cat_w": cat_w,
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
