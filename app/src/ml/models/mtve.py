import json
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################
##                CONFIG                      ##
################################################

@dataclass
class MTVAEConfig:
    num_cont: int
    cat_dims: dict[str, int]
    n_attack_types: int
    hidden_dims: Sequence[int] = (128, 64)
    latent_dim: int = 32
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
    dropout: float = 0.1
    activation_en: str = "relu"
    activation_de_reg: str = "relu"
    activation_de_cls: str = "relu"
    head_hidden_dim: int = 32
    lambda_l3: float = 1.0
    lambda_l7: float = 1.0
    lambda_attack: float = 3.0 # maybe 5.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0 # 0 = standard CE, 2 = common default, 3+ = very aggressive and migth be unstable 
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 60
    warmup_epochs: int = 5
    patience: int = 6
    gradient_clip: float = 1.0
    use_lr_scheduler: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "MTEConfig":
        return MTEConfig(**json.loads(s))


################################################
##   MULTI-TASK VARIATIONAL ENCODER/DECODER   ##
################################################

class MTVEncoder():
    pass


class MTVDecoder():
    pass


################################################
##          MULTI-TASK TABULAR VAE            ##
################################################

class MTTabularVAE(BaseTabularPredictor):
    """
    Hybrid tabular variational autoencoder with:
      - cont and cat inputs
      - learned cat embeddings
      - optional denoising for cont features
      - L3/L7 regression heads and attack type classification head 
      - KL + ELBO reconstruction
      - metrics: xx (for L3/L7) and xx (for attack type)
    """
    pass