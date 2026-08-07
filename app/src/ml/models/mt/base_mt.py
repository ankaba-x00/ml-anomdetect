import json
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################
##                    UTILS                   ##
################################################

def _make_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(inplace=True),
            "silu": nn.SiLU(inplace=True)
        }
        if name not in activations:
            raise ValueError(f"[ERROR] Unknown activation: {name}.")
        return activations[name]

def _pick_init_function(activation, module) -> None:
    if activation in ["relu", "leaky_relu", "gelu", "elu", "silu"]:
        return nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
    elif activation in ["sigmoid", "tanh"]:
        return nn.init.xavier_normal_(module.weight)

class BaseTrafficEncoder(nn.Module):
    """Base encoder for multitask learning. Continuous + categorical embeddings → latent z."""

    def __init__(
        self,
        num_cont: int,
        cat_dims: dict[str, int],
        embedding_dim: Optional[int],
        activation: str = "relu",
        continuous_noise_std: float = 0.0,
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.embedding_dim = embedding_dim
        self.activation = _make_activation(activation)
        self.continuous_noise_std = continuous_noise_std

        # -----------------------------
        # Categorical embeddings
        # -----------------------------
        def emb_dim(card: int) -> int:
            return min(max(4, card // 2), 16)

        self.embeddings = nn.ModuleDict()
        self.emb_sizes = {}

        for name, card in cat_dims.items():
            dim = embedding_dim if embedding_dim is not None else emb_dim(card)
            self.embeddings[name] = nn.Embedding(card, dim)
            self.emb_sizes[name] = dim

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

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _pick_init_function(self.activation, m)
                nn.init.zeros_(m.bias)
                # nn.init.orthogonal_(m.bias) # TODO: test option
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


class BaseTrafficDecoder(nn.Module):
    """
    Base decoder with separate heads for
    - L3 regression: R
    - L7 regression: R
    - Attack type classifier: R^K → softmax → cat dist over K attack types
    """

    def __init__(
        self, 
        config: MTEConfig,
        attack_class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.config = config
        self.register_buffer(
            "attack_class_weights",
            attack_class_weights if attack_class_weights is not None else None
        )

        def make_head_reg(out_dim: int) -> nn.Module:
            if config.head_hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(config.latent_dim, config.head_hidden_dim),
                    _make_activation(config.activation_de_reg),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.head_hidden_dim, out_dim),
                )
            else:
                return nn.Linear(config.latent_dim, out_dim)
        
        def make_head_cls(out_dim: int) -> nn.Module:
            if config.head_hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(config.latent_dim, config.head_hidden_dim),
                    _make_activation(config.activation_de_cls),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.head_hidden_dim, out_dim),
                    nn.Softmax(dim=1),
                )
            else:
                return nn.Linear(config.latent_dim, out_dim)


        self.quantiles = config.quantiles
        nq = len(self.quantiles)

        self.l3_head = make_head_reg(nq)
        self.l7_head = make_head_reg(nq)
        self.attack_head = make_head_cls(config.n_attack_types)

        self._init_heads()

    def _init_heads(self) -> None:
        for head in [self.l3_head, self.l7_head, self.attack_head]:
            for m in head.modules():
                if head == self.attack_head:
                    activation = self.config.activation_de_cls
                else: 
                    activation = self.config.activation_de_reg
                if isinstance(m, nn.Linear):
                    _pick_init_function(activation, m)


class BaseTrafficAttackPredictor(nn.Module):
    """Base class"""

    def __init__(self):
        pass
    
    super().__init__()
    
    def encode():
        pass
    
    def decode():
        pass

    def forward():
        pass

