import json
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch
import torch.nn as nn


#########################################
##                CONFIG               ##
#########################################

@dataclass
class AEConfig:
    input_dim: int # number of features
    latent_dim: int = 8 # bottleneck size
    hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    patience: int = 5 # early stopping patience (epochs) TODO: best?
    val_split: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # add new training options
    gradient_clip: float = 1.0
    use_lr_scheduler: bool = True
    time_series_split: bool = True  # use time-aware split not random
    
    # anomaly detection threshold (can be set after training)
    anomaly_threshold: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "AEConfig":
        d = json.loads(s)
        return AEConfig(**d)


#########################################
##      HYBRID TABULAR AUTOENCODER     ##
#########################################

class TabularAutoencoder(nn.Module):
    """
    MLP autoencoder for tabular time-series features.
    Hybrid aspect:
      - deep non-linear encoder/decoder
      - plus a residual linear skip from input → output
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # -------------------------------
        # Encoder
        # -------------------------------
        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # -------------------------------
        # Decoder (mirror)
        # -------------------------------
        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # residual path: project input → input_dim
        self.input_proj = nn.Linear(input_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: test denoising and hybrid reconstruction options
        # denoising (only during training)
        if self.training:
            x_noisy = x + 0.01 * torch.randn_like(x)
        else:
            x_noisy = x
        z = self.encoder(x_noisy)
        recon = self.decoder(z)

        # hybrid residual path
        residual = self.input_proj(x).detach()
        out = recon + 0.1 * residual  # hybrid: reconstruction + small residual

        return out
        # OPTION 1 : output-level residual
        #z = self.encode(x)
        #recon = self.decode(z)
        # Hybrid: controlled, non-bypass residual: scaled by 0.1; stop gradients through the residual (so it cannot learn to simply copy the input)
        #residual = self.input_proj(x).detach() # no gradient
        #out = recon + 0.1 * residual # small contribution only 
        #return out
    
        # OPTION 2: latent-level skip
        # Add direct input influence to latent space
        #z = self.encode(x)
        #skip_latent = self.skip_connections[0](x)
        #z_enhanced = z + 0.05 * skip_latent  # Weighted combination: learnable transformation
        #recon = self.decode(z_enhanced)
        #return recon
    
        # OPTION 3: denoising autoencoder
        # add noise during training only
        # if self.training:
            # x_noisy = x + torch.randn_like(x) * 0.1
        # else:
            # x_noisy = x
        # z = self.encode(x_noisy)
        # recon = self.decode(z)
        # return recon

        # OPTION 4: or proper hybrid approach
        # z = self.encode(x)
        # recon_main = self.decode(z)
        # Small, fixed residual that CANNOT reconstruct alone
        # residual = self.input_proj(x) * 0.05  # Very small weight
        # The magic: residual helps but main network must do most work
        # out = recon_main + residual
        # return out
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error as anomaly score"""
        with torch.no_grad():
            recon = self.forward(x)
            # MSE per sample
            return torch.mean((recon - x) ** 2, dim=1)
