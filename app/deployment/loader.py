import json, pickle
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from torch import Tensor
from typing import Literal

from app.src.ml.models.configs import AEConfig, MTAEConfig, VAEConfig
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.mtae import MTTabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.helpers import load_autoencoder


FILE_DIR = Path(__file__).resolve().parent
MODELS_DIR = FILE_DIR / "models"


@dataclass(slots=True, frozen=True)
class InferenceBundle:
    """
    Immutable container holding inference bundle and its metadata.
    """

    model: TabularAE | TabularVAE | MTTabularAE
    cfg: AEConfig | VAEConfig | MTAEConfig
    loss_weights: dict[str, float]
    attack_type_weights: Tensor | None
    scaler: RobustScaler
    model_num_cont: int
    model_cat_dims: dict[str, int]
    threshold: float
    method: str
    temperature: float


def load_inference_bundle(
    ae_type: Literal["ae", "vae", "mtae"], 
    country: str
) -> InferenceBundle:
    """Loads inference bundle incl. model bundle, scaler and threshold.."""
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

    model_bundle = load_autoencoder(model_path)
    
    if model_bundle.metadata is not None:
        loss_weights = model_bundle.metadata["loss_weights"]
        if ae_type in ["mtae"]:
            attack_type_weights = Tensor(model_bundle.metadata["attack_type_weights"])
        else:
            attack_type_weights = None

    with open(scaler_path, "rb") as f:
        scaler: RobustScaler = pickle.load(f)

    with open(threshold_path, "r") as f:
        cal_res = json.load(f)
        threshold = cal_res["threshold"]
        method = cal_res["method"]
        temperature = cal_res.get("temperature_inference", model_bundle.cfg.temperature)

    return InferenceBundle(
        model=model_bundle.model,
        cfg=model_bundle.cfg,
        loss_weights=loss_weights,
        attack_type_weights=attack_type_weights,
        scaler=scaler,
        model_num_cont=model_bundle.num_cont,
        model_cat_dims=model_bundle.cat_dims,
        threshold=float(threshold),
        method=method,
        temperature=temperature
    )