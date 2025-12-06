import json, torch
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch.nn as nn
import torch.nn.functional as F
from app.src.ml.models.base import BaseTabularModel


################################################
##                   CONFIG                   ##
################################################

@dataclass
class AEConfig:
    num_cont: int
    cat_dims: dict[str, int]
    latent_dim: int = 8 # bottleneck size
    hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.1
    embedding_dim: Optional[int] = None
    continuous_noise_std: float = 0.0
    residual_strength: float = 0.0
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    patience: int = 5
    gradient_clip: float = 1.0
    optimizer: str = "adam"
    lr_scheduler: str = "none"
    use_lr_scheduler: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_threshold: Optional[float] = None
    temperature: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "AEConfig":
        return AEConfig(**json.loads(s))


################################################
##     HYBRID TABULAR AE WITH EMBEDDINGS      ##
################################################

class TabularAE(BaseTabularModel):
    """
    Hybrid tabular autoencoder with:
      - continuous + categorical inputs
      - learned categorical embeddings
      - optional denoising on continuous inputs
      - anomaly scoring
    """

    def __init__(
        self,
        num_cont: int,
        cat_dims: dict[str, int],
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        embedding_dim: Optional[int] = None, 
        continuous_noise_std: float = 0.0,
        residual_strength: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            num_cont=num_cont,
            cat_dims=cat_dims,
            embedding_dim=embedding_dim,
            continuous_noise_std=continuous_noise_std,
            activation=activation,
        )

        self.latent_dim = latent_dim
        self.residual_strength = residual_strength

        if residual_strength > 0:
            self.residual_proj = nn.Linear(self.input_dim, latent_dim)

        # -------------------------------
        # Encoder
        # -------------------------------
        self.encoder_layers = nn.ModuleList()
        prev = self.input_dim
        
        for h in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout)
                )
            )
            prev = h
        
        # Final encoder layer (no activation for bottleneck)
        self.encoder_out = nn.Linear(prev, latent_dim)

        # -------------------------------
        # Decoder (mirror)
        # -------------------------------
        self.decoder_layers = nn.ModuleList()
        prev = latent_dim
        
        for h in reversed(hidden_dims):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout)
                )
            )
            prev = h
        
        # Continuous reconstruction head
        self.cont_recon = nn.Linear(prev, num_cont)
        
        # Categorical reconstruction heads
        self.cat_recon_heads = nn.ModuleDict()
        for name, card in cat_dims.items():
            self.cat_recon_heads[name] = nn.Linear(prev, card)

    # -------------------------------
    # Encode/Decode
    # -------------------------------
    def encode(
            self, 
            x_cont: torch.Tensor, 
            x_cat: torch.Tensor
        ) -> torch.Tensor:
        """Encode cont and cat to latent space z."""
        x_emb = self._embed(x_cat)
        
        # Apply noise to continuous part (denoising AE)
        x_cont_n = self._maybe_noisy_cont(x_cont)

        # Concatenate into unified representation
        x = torch.cat([x_cont_n, x_emb], dim=1)

        # Forward through encoder
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        z = self.encoder_out(h)
        
        # Optional residual connection
        if self.residual_strength > 0:
            z = z + self.residual_strength * self.residual_proj(x)
        
        return z
    
    def decode(
            self, 
            z: torch.Tensor, 
            temperature: float = 1.0
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode from latent space to cont reconstruction + cat logits with temperature scaling"""
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        
        # Continuous reconstruction
        cont_recon = self.cont_recon(h)
        
        # Categorical reconstructions
        cat_logits = {}
        for name in self.cat_dims.keys():
            logits = self.cat_recon_heads[name](h)
            if temperature != 1.0:
                logits = logits / temperature
            cat_logits[name] = logits
        
        return cont_recon, cat_logits
    
    # -------------------------------
    # Anomaly score
    # -------------------------------
    def anomaly_score(
            self, 
            x_cont: torch.Tensor, 
            x_cat: torch.Tensor, 
            cont_weight: float = 1.0, 
            cat_weight: float = 0.0, 
            temperature: float = 1.0
        ) -> torch.Tensor:
        """Combined reconstruction error with optional categorical weighting which returns per-sample anomaly score [batch,]."""
        with torch.no_grad():
            cont_recon, cat_logits = self.forward(
                x_cont, 
                x_cat, 
                True, 
                temperature
            )
            
            # Continuous MSE per sample
            cont_error = ((cont_recon - x_cont) ** 2).mean(dim=1)
            
            if cat_weight <= 0.0 or len(self.cat_dims) == 0:
                return cont_error

            # Categorical average CE accross cat features 
            cat_error = torch.zeros_like(cont_error)
            n_cats = len(self.cat_dims)
            for i, name in enumerate(self.cat_dims.keys()):
                targets = x_cat[:, i].long()
                ce = F.cross_entropy(
                    cat_logits[name],
                    targets,
                    reduction="none"
                )
                cat_error += ce
            cat_error = cat_error / float(n_cats)

            # Weighted combination
            total_weight = cont_weight + cat_weight
            return (cont_weight * cont_error + cat_weight * cat_error) / total_weight