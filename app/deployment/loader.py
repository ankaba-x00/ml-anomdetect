import json, pickle, torch
from pathlib import Path
from typing import Any

from app.src.ml.models.helpers import load_autoencoder


FILE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FILE_DIR / "models"


def load_model_bundle(
    ae_type: str, 
    country: str
) -> dict[str, Any]:
    """Loads model bundle for inference."""
    print(f"[INFO] Loading inference bundle for {country}...")

    model_path = MODELS_DIR / f"{ae_type.upper()}" / f"{country}_autoencoder.pt"
    scaler_path = MODELS_DIR / f"{ae_type.upper()}" / f"{country}_scaler.pkl"
    threshold_path = MODELS_DIR / f"{ae_type.upper()}" / f"{country}_cal_threshold.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"[ERROR] Threshold not found: {threshold_path}")

    model, cfg, model_num_cont, model_cat_dims, metadata = load_autoencoder(model_path, metadata=True)
    
    try:
        loss_weights = metadata["loss_weights"]
        if ae_type in ["mtae"]:
            attack_type_weights = torch.Tensor(metadata["attack_type_weights"])
    except KeyError:
        print("[ERROR] No weights saved in model metadata")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(threshold_path, "r") as f:
        calibration_obj = json.load(f)
        threshold = calibration_obj["threshold"]
        method = calibration_obj["method"]
        temperature = calibration_obj.get("temperature_inference", 1.0)

    bundle_dict = {
        "model": model,
        "config": cfg,
        "loss_weights": loss_weights,
        "scaler": scaler,
        "model_num_cont": model_num_cont,
        "model_cat_dims": model_cat_dims,
        "threshold": float(threshold),
        "method": method,
        "temperature": temperature,
    }
    if ae_type in ["mtae"]:
        bundle_dict["attack_type_weights"] = attack_type_weights

    return bundle_dict
