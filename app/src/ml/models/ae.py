import json, torch
from dataclasses import dataclass, asdict
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from app.src.ml.models.base import BaseTabularEncoder, BaseTabularDecoder, BaseTabularPredictor, _init_head


################################################
##                   CONFIG                   ##
################################################

@dataclass
class AEConfig:
    num_cont: int
    cat_dims: dict[str, int]
    use_embedding: bool = False
    embedding_dim: int | None = None
    hidden_dims: Sequence[int] = (128, 64)
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
    warmup_epochs: int = 10
    patience: int = 10
    anomaly_threshold: float | None = None 
    temperature: float = 1.0  
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "AEConfig":
        return AEConfig(**json.loads(s))


################################################
##               ENCODER/DECODER              ##
################################################

class Encoder(BaseTabularEncoder):
    """
    Encoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig
    ):
        super().__init__(
            num_cont=config.num_cont,
            cat_dims=config.cat_dims,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            use_embedding=config.use_embedding,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            activation=config.activation_en,
        )
        
        self.hidden_dims = config.hidden_dims
        self.latent_dim = config.latent_dim
        self.act_name = config.activation_en
    
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""

        self.comp_head = nn.Linear(min(self.hidden_dims), self.latent_dim)
    
    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""

        _init_head(self.comp_head, self.act_name)


class Decoder(BaseTabularDecoder):
    """
    Decoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig
    ):
        super().__init__(
            num_cont=config.num_cont,
            cat_dims=config.cat_dims,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            activation=config.activation_de
        )

        self.num_cont = config.num_cont
        self.cat_dims = config.cat_dims
        self.hidden_dims = config.hidden_dims
        self.act_name = config.activation_de
    
    def make_recon_heads(self) -> None:
        """Adds final reconstruction heads of decoder."""

        dim = max(self.hidden_dims)
        
        self.cont_recon_head = nn.Linear(dim, self.num_cont)
    
        self.cat_recon_heads = nn.ModuleDict()
        for name, card in self.cat_dims.items():
            self.cat_recon_heads[name] = nn.Linear(dim, card)
    
    def init_recon_heads(self) -> None:
        """Initializes final reconstruction heads of decoder."""

        _init_head(self.cont_recon_head, self.act_name)

        for module in self.cat_recon_heads.children():
            _init_head(module, self.act_name)


################################################
##             HYBRID TABULAR AE              ##
################################################

class TabularAE(BaseTabularPredictor):
    """
    Hybrid tabular AE with:
      - cont and cat inputs
      - learned cat embeddings
      - optional noise injection
      - separate decoding heads for cont and cat features
      - anomaly scoring
    """
    def __init__(
        self,
        config: AEConfig,
    ):
        super().__init__()
        self.config = config
        self.E = Encoder(config)
        self.D = Decoder(config)
    
    def encode(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor
    ) -> torch.Tensor:
        """Pass input through encoder and return latent variables."""
        
        # either embeds or encodes cat feature vector
        if self.config.use_embedding:
            x_cat_e = self._embed(x_cat)
        else:
            x_cat_e = self._encod(x_cat)

        x = torch.cat([x_cont, x_cat_e], dim=1)
        h = x
        for layer in self.E.encoder_layers:
            h = layer(h)
        
        z = self.E.comp_head(h)
        
        return z

    def decode(
        self, 
        z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode from latent space and apply temperature scaling."""
        
        h = z
        for layer in self.D.decoder_layers:
            h = layer(h)
        
        cont_recon = self.D.cont_recon_head(h)

        cat_logits = {}
        for name in self.config.cat_dims.keys():
            logits = self.D.cat_recon_heads[name](h)
            if self.config.temperature != 0.:
                logits = logits / self.config.temperature
            cat_logits[name] = logits
        
        return cont_recon, cat_logits

    def forward(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass through autoencoder."""

        if self.config.allow_noise_injection:
            Xc, Xk = self._noise_injection(x_cont, x_cat)
        else:
            Xc, Xk = x_cont, x_cat

        z = self.encode(x_cont, x_cat)
        cont_recon, cat_logits = self.decode(z)

        return cont_recon, cat_logits
    
    # -----------------------------
    # Utils
    # -----------------------------
    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed cat features using Embedding."""

        parts = []
        for i, name in enumerate(self.config.cat_dims.keys()):
            parts.append(self.E.embeddings[name](x_cat[:, i].long()))
        
        return torch.cat(parts, dim=1)
    
    def _encod(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Encode cat features via OneHotEncoding."""
        
        parts = []
        for i, card in enumerate(self.config.cat_dims.values()):
            parts.append(F.one_hot(x_cat[:, i].long(), num_classes=card).float())
        
        return torch.cat(parts, dim=1)
    
    def _noise_injection(self, x_cont, x_cat) -> tuple[torch.Tensor]:
        """Inject noise to enable denoising autoencoder"""

        # Cont: adds Gaussian noise
        if self.config.noise_gauss_std > 0:
            noise = torch.randn_like(x_cont) * self.config.noise_gauss_std
            x_cont_noisy = x_cont + noise
        else:
            x_cont_noisy = x_cont

        # Cat: random masking noise (replaces values with low probability)
        if self.config.noise_mask_prob > 0:
            x_cat_noisy = x_cat.clone()
            mask = torch.randn_like(x_cat.float()) < self.config.noise_mask_prob
            x_cat_noisy[mask] = 0
        else:
            x_cat_noisy = x_cat

        return x_cont_noisy, x_cat_noisy

    # -----------------------------
    # Model scoring
    # -----------------------------
    def scoring(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        cont_recon: torch.Tensor,
        cat_logits: dict[str, torch.Tensor],
        loss_weights: dict[str, float],
        in_warmup: bool = False,
        reduction: str = "mean"
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """Calculates normalized weighted per sample score."""

        # Cont Huber loss per sample
        cont_loss = F.huber_loss(cont_recon, x_cont, reduction="none").mean(dim=1)

        # Cat CE per sample
        cat_loss = torch.zeros_like(cont_loss)
        cat_names = self.config.cat_dims.keys()
        for i, name in enumerate(cat_names):
            ce = F.cross_entropy(
                cat_logits[name],
                x_cat[:, i].long(),
                reduction="none"
            )
            cat_loss += ce
        cat_loss = cat_loss / float(len(cat_names))
    
        w_cont, w_cat = loss_weights["cont_w"], loss_weights["cat_w"]

        if in_warmup:
            per_sample_score = cat_loss
        else:
            per_sample_score = (w_cont * cont_loss + w_cat * cat_loss) / (w_cont + w_cat)

        if reduction == "mean":
            return cont_loss, cat_loss, per_sample_score.mean()
        else:
            return per_sample_score

    
    

