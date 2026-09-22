from dataclasses import dataclass
from typing import Any

from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from app.src.ml.models.mtae import MTTabularAE


@dataclass(slots=True, frozen=True)
class LoadedAutoencoder:
    """
    Immutable container holding a model and its metadata.
    """

    model: TabularAE | TabularVAE | MTTabularAE 
    cfg: AEConfig | VAEConfig | MTAEConfig
    num_cont: int
    cat_dims: dict[str, int]
    metadata: dict[str, Any] | None = None
