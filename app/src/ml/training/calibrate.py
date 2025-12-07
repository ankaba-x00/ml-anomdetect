import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import TransformerMixin
from app.src.data.feature_engineering import build_feature_matrix
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.training.evaluate import reconstruction_error, threshold_percentile, threshold_mad


def _compute_threshold(method: str, err_array: np.ndarray):
    """Computes threshold for a given method and error array"""
    if method == "p99":
        return threshold_percentile(err_array, 99)
    elif method == "p995":
        return threshold_percentile(err_array, 99.5)
    elif method == "mad":
        return threshold_mad(err_array, k=6)
    else:
        raise ValueError(f"[ERROR] Unknown threshold method {method}")

# TODO: benchmark threshold calibration window 7 to 30 days 
def calibrate_threshold(
        country: str,
        model: Union[TabularAE, TabularVAE],
        scaler: TransformerMixin,
        device: str = None,
        method: str = "p99",
        cw: int = 30,
        cont_weight: float = 1.0,
        cat_weight: float = 0.0, 
        tune_temperature: bool = True,
        temperature_range: list = None
) -> dict[str, Union[np.ndarray, float]]:
        """Computes anomaly threshold and temperature scaling for a given model on a specified calibration window, threshold method and temperature range."""
        
        # ------------------------------------
        # Build raw feature matrix
        # ------------------------------------ 
        X_cont, X_cat, _, _ = build_feature_matrix(country)
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
        print(f"[INFO] Calibration window: {start_time} to {end_time}")
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
                temp_errors = reconstruction_error(
                    model,
                    X_cont_cal_scld,
                    X_cat_cal,
                    device,
                    cont_weight,
                    cat_weight,
                    temp,
                )
                
                # Compute preliminary threshold with this temperature
                temp_prelim = _compute_threshold(method, temp_errors)
                temp_clean_errors = temp_errors[temp_errors < temp_prelim]
                temp_metric = np.median(temp_clean_errors) if len(temp_clean_errors) > 0 else np.median(temp_errors)
            
                temp_results[temp] = {
                    "errors_mean": float(temp_errors.mean()),
                    "errors_std": float(temp_errors.std()),
                    "errors_median": float(np.median(temp_errors)),
                    "clean_errors_median": float(np.median(temp_clean_errors)) if len(temp_clean_errors) > 0 else float(np.median(temp_errors)),
                    "prelim_threshold": float(temp_prelim),
                    "clean_samples": len(temp_clean_errors),
                }
                stability_score = temp_results[temp]["errors_std"] / (temp_results[temp]["errors_mean"] + 1e-8)
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
        errors = reconstruction_error(
            model,
            X_cont_cal_scld,
            X_cat_cal,
            device,
            cont_weight,
            cat_weight,
            best_temp
        )
        print(f"[INFO] Computed {len(errors)} errors with temp={best_temp}")

        # ------------------------------------
        # Remove spikes/anomalies in window
        # ------------------------------------ 
        prelim = _compute_threshold(method, errors)
        clean_errors = errors[errors < prelim]
        removed_count = len(errors) - len(clean_errors)
        print(f"[INFO] Removed {removed_count} preliminary anomalies to clean window")

        if len(clean_errors) < 10:
            print(f"[WARN] Very few clean samples ({len(clean_errors)}). Using all errors.")
            clean_errors = errors

        # ------------------------------------
        # Compute threshold on cleaned window
        # ------------------------------------
        threshold = _compute_threshold(method, clean_errors)
        print(f"[CAL] Computed threshold ({method}) = {threshold:.6f}")

        anomaly_rate = np.mean(errors > threshold) * 100
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
            "clean_samples": len(clean_errors),
            "preliminary_anomalies_removed": removed_count,
            "cont_weight": cont_weight,
            "cat_weight": cat_weight,
            "error_stats": {
                "min": float(errors.min()),
                "mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "max": float(errors.max()),
                "std": float(errors.std()),
                "p99": float(np.percentile(errors, 99)),
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
            "errors": errors,
            "clean_errors": clean_errors,
            "preliminary_threshold": float(prelim),
        }

        return threshold_dict, debug_dict