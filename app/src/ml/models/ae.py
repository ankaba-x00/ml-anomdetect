import json, torch
from dataclasses import dataclass, asdict
from typing import Sequence, Optional, Union
import torch.nn as nn
import torch.nn.functional as F


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
    activation: str = "relu"
    temperature: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "AEConfig":
        return AEConfig(**json.loads(s))


################################################
## HYBRID TABULAR AUTOENCODER WITH EMBEDDINGS ##
################################################

class TabularAutoencoder(nn.Module):

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
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.latent_dim = latent_dim
        self.continuous_noise_std = continuous_noise_std
        self.residual_strength = residual_strength

        # -----------------------
        # Activation function
        # -----------------------
        self.activation = self._make_activation(activation)

        # -------------------------------
        # Embeddings
        # -------------------------------
        # Example cat_dims:
        # {"weekday":7, "daytype":2, "daytime":5, "month":12, "week":53}
        # For embedding dim: d = min( max(4, card//2), 16 )
        def emb_dim(card: int) -> int:
            return min(max(4, card // 2), 16)

        self.embeddings = nn.ModuleDict()
        self.emb_sizes: dict[str, int] = {}

        for name, card in cat_dims.items():
            d = embedding_dim if embedding_dim is not None else emb_dim(card)
            self.embeddings[name] = nn.Embedding(card, d)
            self.emb_sizes[name] = d

        total_emb_dim = sum(self.emb_sizes.values())
        self.input_dim = num_cont + total_emb_dim

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
    # Utilities
    # -------------------------------
    def _make_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(inplace=True),
        }
        if name not in activations:
            raise ValueError(f"[ERROR] Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]
    
    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical features."""
        parts = []
        for i, name in enumerate(self.cat_dims.keys()):
            parts.append(self.embeddings[name](x_cat[:, i].long()))
        return torch.cat(parts, dim=1)

    # -------------------------------
    # Encode/Decode
    # -------------------------------
    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Encode cont and cat to latent space z."""
        x_emb = self._embed(x_cat)
        
        # apply noise only to cont part of training
        if self.training and self.continuous_noise_std > 0.:
            noise = torch.randn_like(x_cont) * self.continuous_noise_std
            # Only add noise to continuous part
            x_cont_noisy = x_cont + noise
            x = torch.cat([x_cont_noisy, x_emb], dim=1)
        else:
            x = torch.cat([x_cont, x_emb], dim=1)
        
        # Forward through encoder
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        z = self.encoder_out(h)
        
        # Optional residual connection
        if self.residual_strength > 0:
            z = z + self.residual_strength * self.residual_proj(x)
        
        return z
    
    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Decode from latent space to 
            - cont reconstruction
            - cat logits with temperature scaling
        """
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        
        # Continuous reconstruction
        cont_recon = self.cont_recon(h)
        
        # Categorical reconstructions
        cat_logits = {}
        for name in self.cat_dims.keys():
            logits = self.cat_recon_heads[name](h)
            cat_logits[name] = logits if temperature == 1.0 else logits / temperature
        
        return cont_recon, cat_logits

    # -------------------------------
    # Forward
    # -------------------------------
    # TODO: train to get both and inference
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor, return_cat: bool = True, temperature: float = 1.0) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """Full forward pass: returns cont recon and optionally cat logits"""
        z = self.encode(x_cont, x_cat)
        cont_recon, cat_logits = self.decode(z, temperature)
        if return_cat:
            return cont_recon, cat_logits
        else:
            return cont_recon
    
    # -------------------------------
    # Anomaly score
    # -------------------------------
    def anomaly_score(self, x_cont: torch.Tensor, x_cat: torch.Tensor, cont_weight: float = 1.0, cat_weight: float = 0.0, temperature: float = 1.0) -> torch.Tensor:
        """Combined reconstruction error with optional categorical weighting which returns per-sampel anomaly score [batch,]."""
        with torch.no_grad():
            cont_recon, cat_logits = self.forward(x_cont, x_cat, True, temperature)
            
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
        
################################################
## HYBRID TABULAR AUTOENCODER WITH EMBEDDINGS ##
################################################