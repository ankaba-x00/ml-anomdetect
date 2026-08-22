import json, torch
from dataclasses import dataclass, asdict
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from .base import BaseTabularEncoder, BaseTabularDecoder, BaseTabularPredictor
from .mixins import TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularLayerInitMixin, TabularReconScoringMixin


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

class Encoder(BaseTabularEncoder, TabularLayerInitMixin):
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

        self.init_head(self.comp_head, self.act_name)


class Decoder(BaseTabularDecoder, TabularLayerInitMixin):
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

        self.init_head(self.cont_recon_head, self.act_name)

        for module in self.cat_recon_heads.children():
            self.init_head(module, self.act_name)


################################################
##             HYBRID TABULAR AE              ##
################################################

class TabularAE(BaseTabularPredictor, TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin):
    """
    Hybrid tabular AE with:
      - cont and cat inputs
      - Embedding or OneHotEncoding for cat features
      - (optional) noise injection for cont and cat features
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
            x_cat_e = self._embed(
                x_cat, 
                self.config.cat_dims, 
                self.E.embeddings
            )
        else:
            x_cat_e = self._encod(
                x_cat, 
                self.config.cat_dims
            )

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
            Xc, Xk = self._noise_injection(
                x_cont, 
                x_cat,
                self.config.noise_gauss_std,
                self.config.noise_mask_prob
            )
        else:
            Xc, Xk = x_cont, x_cat

        z = self.encode(Xc, Xk)
        cont_recon, cat_logits = self.decode(z)

        return cont_recon, cat_logits

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
        """Computes normalized weighted per-sample score and optionally reduces to average or sum."""

        cont_score, cat_score, per_sample_score = self.recon_scoring(
            x_cont,
            x_cat,
            cont_recon,
            cat_logits,
            self.config.cat_dims,
            loss_weights,
            in_warmup
        )

        if reduction == "mean":
            return cont_score, cat_score, per_sample_score.mean()
        elif reduction == "sum":
            return cont_score, cat_score, per_sample_score.sum()
        else:
            return per_sample_score
