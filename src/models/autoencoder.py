import json
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch
import torch.nn as nn


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

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    patience: int = 5 # early stopping patience (epochs) TODO: best?
    val_split: float = 0.2
    gradient_clip: float = 1.0
    use_lr_scheduler: bool = True
    time_series_split: bool = True  # use time-aware split not random
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # anomaly detection threshold (filled after training)
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
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.latent_dim = latent_dim

        # -------------------------------
        # Categorical embedding dynamically
        # -------------------------------

        # Example cat_dims:
        # {"weekday":7, "daytype":2, "daytime":5, "month":12, "week":53}
        # For embedding dim: d = min( max(4, card//2), 16 )
        def emb_dim(card):
            return min(max(4, card // 2), 16)

        self.embeddings = nn.ModuleDict()
        self.emb_sizes: dict[str, int] = {}

        for name, card in cat_dims.items():
            d = emb_dim(card)
            self.embeddings[name] = nn.Embedding(card, d)
            self.emb_sizes[name] = d

        total_emb_dim = sum(self.emb_sizes.values())

        # -------------------------------
        # Encoder
        # -------------------------------
        enc_in_dim = num_cont + total_emb_dim
        
        enc_layers = []
        prev = enc_in_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h

        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # -------------------------------
        # Decoder (mirror)
        # -------------------------------
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h

        # OUTPUT = reconstruct continuous features only
        dec_layers.append(nn.Linear(prev, num_cont))
        self.decoder = nn.Sequential(*dec_layers)

    # -------------------------------
    # Embedding fuser
    # -------------------------------
    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat shape: (batch, num_categorical)
        Mapped in fixed order based on cat_dims.keys()
        """
        parts: list[torch.Tensor] = []

        # ensure consistent order of categories:
        for i, name in enumerate(self.cat_dims.keys()):
            parts.append(self.embeddings[name](x_cat[:, i]))

        return torch.cat(parts, dim=1)

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cont : (batch, num_cont)
        x_cat  : (batch, num_categories)
        """
        # embeddings
        x_emb = self._embed(x_cat)

        # concatenate continuous + embedding features
        x = torch.cat([x_cont, x_emb], dim=1)

        # optional denoising in training only
        if self.training:
            x = x + 0.01 * torch.randn_like(x)

        z = self.encoder(x)
        recon = self.decoder(z)

        return recon

    # -------------------------------
    # Anomaly score
    # -------------------------------
    def anomaly_score(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x_cont, x_cat)
            return ((recon - x_cont) ** 2).mean(dim=1)
