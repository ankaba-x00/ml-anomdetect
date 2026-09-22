import torch
from pathlib import Path
from typing import Any

from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig
from app.src.ml.models.helpers import LoadedAutoencoder
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.mtae import MTTabularAE


def save_autoencoder(
    model: TabularAE | TabularVAE | MTTabularAE,
    config: AEConfig | VAEConfig | MTAEConfig,
    cat_dims: dict[str, int],
    num_cont: int,
    path: Path,
    metadata: dict[str, Any] | None = None
) -> None:
    """
    Stores model, config, cat_dims, num_cont, model_class and metadata in
    pt file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = config.to_dict()
    del config_payload["hidden_dims"]

    payload = {
        "model_class": model.__class__.__name__,
        "state_dict": model.state_dict(),
        "config": config_payload,
        "cat_dims": cat_dims,
        "num_cont": num_cont,
        "metadata": metadata or {},
    }
    torch.save(payload, path)

    print(f"[OK] Saved autoencoder to {path}")

def load_autoencoder(
    path: Path,
    device: str = "cpu",
) -> LoadedAutoencoder:
    """
    Unpacks pt file and loads model, config, cat_dims, num_cont, model_class 
    and metadata.
    """
    payload = torch.load(path, map_location=device, weights_only=True)

    num_cont = payload["num_cont"]
    cat_dims = payload["cat_dims"]
    metadata = payload["metadata"] if "metadata" in payload.keys() else None

    class_map = {
        "TabularAE": [AEConfig, TabularAE],
        "TabularVAE": [VAEConfig, TabularVAE],
        "MTTabularAE": [MTAEConfig, MTTabularAE],
    }

    ae_class = payload["model_class"]
    cfg = class_map[ae_class][0](**payload["config"])
    model = class_map[ae_class][1](config=cfg)
    
    model.load_state_dict(payload["state_dict"])
    target_device = torch.device(cfg.device)
    model = model.to(target_device)
    
    print(f"[INFO] Loaded autoencoder from {path}")
    print(f"[INFO] Model moved to device: {target_device}")

    return LoadedAutoencoder(
        model=model,
        cfg=cfg,
        num_cont=num_cont,
        cat_dims=cat_dims,
        metadata=metadata
    )