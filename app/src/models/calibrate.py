
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
) -> dict[str, Union[np.ndarray, float]]:
        """Computes anomaly threshold for a given model on a specified calibration window and threshold method."""
        
        # ------------------------------------
        # Apply trained scaler on raw feature matrix
        # ------------------------------------ 
        X_cont, X_cat, _, _, _ = build_feature_matrix(country, scaler=scaler)
        Xc_np = X_cont.values.astype(np.float32)
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
        # Compute recon error in window
        # ------------------------------------ 
        errors = reconstruction_error(
            model,
            X_cont_cal,
            X_cat_cal,
            device
        )
        print(f"[INFO] Computed {len(errors)} errors")

        # ------------------------------------
        # Remove spikes/anomalies in window
        # ------------------------------------ 
        prelim = _compute_threshold(method, errors)
        clean_errors = errors[errors < prelim]
        print(f"[INFO] Removed {len(errors)-len(clean_errors)} preliminary anomalies to clean window")

        # ------------------------------------
        # Compute threshold on cleaned window
        # ------------------------------------
        threshold = _compute_threshold(method, clean_errors)
        print(f"[INFO] Computed threshold ({method}) = {threshold:.6f}")

        threshold_dict = {"device": device, "method": method, "threshold": threshold}
        debug_dict = {"window": ts_cal, "errors": clean_errors, }

        return threshold_dict, debug_dict