import json, torch
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
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
        residual_strength: float = 0.0
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.latent_dim = latent_dim
        self.continuous_noise_std = continuous_noise_std
        self.residual_strength = residual_strength

        # Example cat_dims:
        # {"weekday":7, "daytype":2, "daytime":5, "month":12, "week":53}
        # For embedding dim: d = min( max(4, card//2), 16 )
        def emb_dim(card):
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
            self.residual_proj = nn.Linear(num_cont + total_emb_dim, latent_dim)

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
                    nn.ReLU(inplace=True),
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
                    nn.ReLU(inplace=True),
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
    # Embedding fuser
    # -------------------------------
    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical features."""
        parts = []

        # ensure consistent order of categories:
        for i, name in enumerate(self.cat_dims.keys()):
            parts.append(self.embeddings[name](x_cat[:, i]))

        return torch.cat(parts, dim=1)

    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        x_emb = self._embed(x_cat)
        x = torch.cat([x_cont, x_emb], dim=1)
        
        # Add noise during training (denoising autoencoder)
        if self.training and self.continuous_noise_std > 0.:
            noise = torch.randn_like(x_cont) * self.continuous_noise_std
            # Only add noise to continuous part
            x_cont_noisy = x_cont + noise
            x = torch.cat([x_cont_noisy, x_emb], dim=1)
        
        # Forward through encoder
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        z = self.encoder_out(h)
        
        # Optional residual connection
        if self.residual_strength > 0:
            z = z + self.residual_strength * self.residual_proj(x)
        
        return z
    
    def decode(self, z: torch.Tensor):
        """Decode from latent space."""
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        
        # Continuous reconstruction
        cont_recon = self.cont_recon(h)
        
        # Categorical reconstructions
        cat_recons = {}
        for name in self.cat_dims.keys():
            cat_recons[name] = self.cat_recon_heads[name](h)
        
        return cont_recon, cat_recons

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        """Full forward pass."""
        z = self.encode(x_cont, x_cat)
        return self.decode(z)
    
    # -------------------------------
    # Anomaly score
    # -------------------------------
    def anomaly_score(self, x_cont: torch.Tensor, x_cat: torch.Tensor, cont_weight: float = 1.0, cat_weight: float = 1.0) -> torch.Tensor:
        """Combined reconstruction error."""
        with torch.no_grad():
            cont_recon, cat_recons = self.forward(x_cont, x_cat)
            
            # Continuous MSE
            cont_error = ((cont_recon - x_cont) ** 2).mean(dim=1)
            
            # Categorical cross-entropy
            cat_error = torch.zeros_like(cont_error)
            for i, name in enumerate(self.cat_dims.keys()):
                cat_error += F.cross_entropy(
                    cat_recons[name],
                    x_cat[:, i].long(),
                    reduction='none'
                )

            # no weights per feature category
            # total_features = self.num_cont + len(self.cat_dims)
            # return (cont_error * self.num_cont + cat_error) / total_features
            
            # weights per feature category
            total_weight = cont_weight + cat_weight
            return (cont_weight * cont_error + cat_weight * cat_error) / total_weight