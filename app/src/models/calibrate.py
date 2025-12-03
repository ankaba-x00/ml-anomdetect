
import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import TransformerMixin
from app.src.data.feature_engineering import build_feature_matrix
from app.src.models.autoencoder import TabularAutoencoder
from app.src.models.evaluate import reconstruction_error, threshold_percentile, threshold_mad


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
        model: TabularAutoencoder,
        scaler: TransformerMixin,
        device: str = None,
        method: str = "p99",
        cw: int = 30,
        cont_weight: float = 1.0,
        cat_weight: float = 1.0
) -> dict[str, Union[np.ndarray, float]]:
        """Computes anomaly threshold for a given model on a specified calibration window and threshold method."""
        
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
        # Compute recon error in window
        # ------------------------------------ 
        errors = reconstruction_error(
            model,
            X_cont_cal_scld,
            X_cat_cal,
            device,
            cont_weight,
            cat_weight
        )
        print(f"[INFO] Computed {len(errors)} errors")

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
        print(f"[INFO] Computed threshold ({method}) = {threshold:.6f}")

        anomaly_rate = np.mean(errors > threshold) * 100
        print(f"[INFO] Expected anomaly rate: {anomaly_rate:.2f}%")

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
        }
        debug_dict = {
            "window": ts_cal, 
            "errors": errors,
            "clean_errors": clean_errors,
            "preliminary_threshold": float(prelim),
        }

        return threshold_dict, debug_dict