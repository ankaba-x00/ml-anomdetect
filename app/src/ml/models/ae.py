import torch
import torch.nn as nn

from .base import BaseTabularEncoder, BaseTabularDecoder, BaseTabularPredictor
from .configs import AEConfig, MTAEConfig
from .mixins import TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularLayerInitMixin, TabularReconScoringMixin


class Encoder(BaseTabularEncoder, TabularLayerInitMixin):
    """
    Encoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig | MTAEConfig
    ):
        super().__init__(config=config)
        
        self.config = config
    
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""

        self.comp_head = nn.Linear(min(self.config.hidden_dims), self.config.latent_dim)
    
    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""

        self.init_head(self.comp_head, self.config.activation_en)


class Decoder(BaseTabularDecoder, TabularLayerInitMixin):
    """
    Decoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig
    ):
        super().__init__(config=config)
        
        self.config = config
    
    def make_recon_heads(self) -> None:
        """Adds final reconstruction heads of decoder."""

        dim = max(self.config.hidden_dims)
        
        self.cont_recon_head = nn.Linear(dim, self.config.num_cont)
    
        self.cat_recon_heads = nn.ModuleDict()
        for name, card in self.config.cat_dims.items():
            self.cat_recon_heads[name] = nn.Linear(dim, card)
    
    def init_recon_heads(self) -> None:
        """Initializes final reconstruction heads of decoder."""

        self.init_head(self.cont_recon_head, self.config.activation_de)

        for module in self.cat_recon_heads.children():
            self.init_head(module, self.config.activation_de)


class TabularAE(BaseTabularPredictor, TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin):
    """
    Hybrid tabular AE with:
      - cont and cat inputs
      - Embedding or OneHotEncoding for cat features
      - (optional) noise injection for cont and cat features
      - separate decoding heads for cont and cat features
      - unsupervised learning for anomaly scoring
      - metrics: Huber loss for cont and CE for cat features 
    """

    def __init__(
        self,
        config: AEConfig | MTAEConfig,
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
        """Passes input through encoder and returns latent variables."""
        
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
        """Decodes from latent space and applies temperature scaling."""
        
        h = z
        for layer in self.D.decoder_layers:
            h = layer(h)
        
        cont_recon = self.D.cont_recon_head(h)

        cat_logits = {}
        for name in self.config.cat_dims.keys():
            logits = self.D.cat_recon_heads[name](h)
            if self.config.temperature != 1.0:
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
        else:
            return per_sample_score
