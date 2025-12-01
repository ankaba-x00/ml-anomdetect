from pathlib import Path
from typing import Dict, Any
import json, pickle
import numpy as np
from app.src.models.autoencoder import TabularAutoencoder
from app.src.models.evaluate import reconstruction_error, anomaly_mask, find_anomalies
from app.src.models.train import load_autoencoder


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FILE_DIR / "models"


#########################################
#           LOAD DEPENDENCIES
#########################################

def load_inference_bundle(country: str) -> Dict[str, Any]:
    """Load inference bundle for prediction incl. model, scaler, cat_dims for order."""
    print(f"[INFO] Loading inference bundle for {country}...")
    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    scaler_path = MODELS_DIR / f"{country}_scaler_cont.pkl"
    catdims_path = MODELS_DIR / f"{country}_cat_dims.json"
    threshold_path = MODELS_DIR / f"{country}_cal_threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")
    if not catdims_path.exists():
        raise FileNotFoundError(f"[ERROR] Cat_dims not found: {catdims_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"[ERROR] Threshold not found: {threshold_path}")

    model, cfg = load_autoencoder(model_path)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(catdims_path, "r") as f:
        cat_dims = json.load(f)

    with open(threshold_path, "r") as f:
        calibration_obj = json.load(f)
        threshold = calibration_obj["threshold"]
        method = calibration_obj["method"]

    return dict(
        model=model,
        config=cfg,
        scaler=scaler,
        cat_dims=cat_dims,
        threshold=float(threshold),
        method=method
    )


#########################################
#             RUN INFERENCE 
#########################################

def run_inference(
    model: TabularAutoencoder,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    threshold: float,
    device: str = None,
    min_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, Any]:
    """Applies model on data and compute reconstruction errors, threshold, anomaly mask, anomaly intervals."""
    errors = reconstruction_error(
        model,
        X_cont,
        X_cat,
        device=device,
    )
    mask = anomaly_mask(errors, threshold)
    intervals = find_anomalies(
        mask,
        min_length=min_length,
        merge_gap=merge_gap,
    )    
    starts = np.array([s for s, _ in intervals])
    ends   = np.array([e for _, e in intervals])
    return {
        "errors": errors,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }