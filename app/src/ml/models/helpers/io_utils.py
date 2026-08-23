import torch
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig
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
    """Stores model, config, cat_dims, num_cont, model_class and metadata in pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "cat_dims": cat_dims,
        "num_cont": num_cont,
        "model_class": model.__class__.__name__,
        "metadata": metadata or {},
    }
    torch.save(payload, path)

    print(f"[OK] Saved autoencoder to {path}")


def load_autoencoder(
    path: Path,
    device: str = "cpu",
    metadata: bool = False
) -> tuple[
        TabularAE | TabularVAE | MTTabularAE, 
        AEConfig | VAEConfig | MTAEConfig, 
        int, 
        dict[str, int],
        dict[str, Any]
    ]:
    """Loads model, config, cat_dims, num_cont, model_class and metadata in pt file."""
    payload = torch.load(path, map_location=device, weights_only=True)

    num_cont = payload["num_cont"]
    cat_dims = payload["cat_dims"]
    metadata_dict = payload["metadata"]

    ae_class = payload["model_class"]
    if ae_class == "TabularAE":
        cfg = AEConfig(**payload["config"])
        model = TabularAE(config=cfg)
    elif ae_class == "TabularVAE":
        cfg = VAEConfig(**payload["config"])
        model = TabularVAE(config=cfg)
    elif ae_class == "MTTabularAE":
        cfg = MTAEConfig(**payload["config"])
        model = MTTabularAE(config=cfg)
    else:
        raise ValueError(f"Unknown model_class: {ae_class}")
    
    model.load_state_dict(payload["state_dict"])
    target_device = torch.device(cfg.device)
    model = model.to(target_device)
    
    print(f"[INFO] Loaded autoencoder from {path}")
    print(f"[INFO] Model moved to device: {target_device}")
    if metadata:
        return model.eval(), cfg, num_cont, cat_dims, metadata_dict
    return model.eval(), cfg, num_cont, cat_dims