import torch
from dataclasses import dataclass, field
from typing import Sequence

from .cbase import SharedVEncoderConfig, SharedDecoderConfig


@dataclass
class VAEConfig(SharedVEncoderConfig, SharedDecoderConfig):
    """
    Represents a VAE config object for TabularVAE.
    """

    num_cont: int
    cat_dims: dict[str, int]
    use_embedding: bool = False
    embedding_dim: int | None = None
    depth: int = 2
    base_dim: int = 256
    hidden_dims: Sequence[int] = field(init=False)
    latent_dim: int = 32
    activation_en: str = "relu"
    activation_de: str = "relu"
    dropout: float = 0.1
    optimizer: str = "adam"
    lr: float = 1e-5
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.0
    lr_scheduler: str = "none"
    gradient_clip: float | None = None
    batch_size: int = 256
    allow_noise_injection: bool = True
    noise_gauss_std: float = 0.1
    noise_mask_prob: float = 0.05
    num_epochs: int = 60
    patience: int = 10
    temperature: float = 1.0
    use_beta_annealing: bool = True
    beta_schedule: str = "linear"
    beta: float = 1.0
    use_kl_clipping: bool = True
    kl_clamp: float = 0.001
    debug_kl_stats: bool = True
    logvar_clamp: float = 4.0
    cont_w: float = 1.0
    cat_w: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"