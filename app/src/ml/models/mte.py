import json
from dataclasses import dataclass, asdict
from typing import Sequence, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################
##                CONFIG                      ##
################################################

@dataclass
class MTEConfig:
    num_cont: int
    cat_dims: Dict[str, int]
    n_attack_types: int
    hidden_dims: Sequence[int] = (128, 64)
    latent_dim: int = 32
    dropout: float = 0.1
    activation: str = "relu"
    head_hidden_dim: int = 32
    lambda_l3: float = 1.0
    lambda_l7: float = 1.0
    lambda_attack: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 60
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
##               TRAFFIC ENCODER              ##
################################################

class TrafficEncoder(nn.Module):
    """Shared encoder for multitask learning. Continuous + categorical embeddings → latent z."""

    def __init__(
        self,
        num_cont: int,
        cat_dims: Dict[str, int],
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
        continuous_noise_std: float = 0.0,
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.latent_dim = latent_dim
        self.continuous_noise_std = continuous_noise_std
        self.activation = self._make_activation(activation)

        # -----------------------------
        # Categorical embeddings
        # -----------------------------
        def emb_dim(card: int) -> int:
            return min(max(4, card // 2), 16)

        self.embeddings = nn.ModuleDict()
        self.emb_sizes = {}

        for name, card in cat_dims.items():
            d = emb_dim(card)
            self.embeddings[name] = nn.Embedding(card, d)
            self.emb_sizes[name] = d

        emb_total = sum(self.emb_sizes.values())
        input_dim = num_cont + emb_total

        # -----------------------------
        # Encoder MLP
        # -----------------------------
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                self.activation,
                nn.Dropout(dropout),
            ]
            prev = h

        layers.append(nn.Linear(prev, latent_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    # -----------------------------
    # Utilities
    # -----------------------------
    def _make_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(inplace=True),
        }
        if name not in activations:
            raise ValueError(f"[ERROR] Unknown activation: {name}.")
        return activations[name]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        parts = []
        for i, name in enumerate(self.cat_dims.keys()):
            parts.append(self.embeddings[name](x_cat[:, i]))
        return torch.cat(parts, dim=1)

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor
    ) -> torch.Tensor:
        if self.training and self.continuous_noise_std > 0:
            x_cont = x_cont + self.continuous_noise_std * torch.randn_like(x_cont)

        x = torch.cat([x_cont, self._embed(x_cat)], dim=1)
        return self.mlp(x)


################################################
##     MULTI-HEAD L3/L7 + ATTACK PREDICTOR    ##
################################################

class TrafficAttackPredictor(nn.Module):
    """
    Encoder: TrafficEncoder
    Heads:
    - L3 regression: R
    - L7 regression: R
    - Attack type classifier: R^K → softmax → cat dist over K attack types
    """

    def __init__(self, config: MTEConfig):
        super().__init__()

        self.config = config

        self.encoder = TrafficEncoder(
            num_cont=config.num_cont,
            cat_dims=config.cat_dims,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            activation=config.activation,
            continuous_noise_std=0.01,
        )

        def make_head(out_dim: int) -> nn.Module:
            if config.head_hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(config.latent_dim, config.head_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.head_hidden_dim, out_dim),
                )
            else:
                return nn.Linear(config.latent_dim, out_dim)

        self.l3_head = make_head(1)
        self.l7_head = make_head(1)
        self.attack_head = make_head(config.n_attack_types)

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [self.l3_head, self.l7_head, self.attack_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(x_cont, x_cat)
        return {
            "z": z,
            "l3": self.l3_head(z).squeeze(-1),
            "l7": self.l7_head(z).squeeze(-1),
            "attack_logits": self.attack_head(z),
        }

    def compute_loss(
        self,
        outputs: dict,
        y_l3: torch.Tensor,
        y_l7: torch.Tensor,
        y_attack: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l3_loss = F.mse_loss(outputs["l3"], y_l3)
        l7_loss = F.mse_loss(outputs["l7"], y_l7)
        attack_loss = F.cross_entropy(outputs["attack_logits"], y_attack)

        total = (
            self.config.lambda_l3 * l3_loss +
            self.config.lambda_l7 * l7_loss +
            self.config.lambda_attack * attack_loss
        )

        return {
            "total": total,
            "l3": l3_loss,
            "l7": l7_loss,
            "attack": attack_loss,
        }