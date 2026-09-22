import torch
import torch.nn as nn

from .base import TabularBase, SharedEncoder, SharedDecoder
from .configs import AEConfig, VAEConfig, MTAEConfig
from .mixins import TabularDecodePassingMixin, TabularEncodePassingMixin, TabularFeatureForwardMixin, TabularLayerInitMixin, TabularReconScoringMixin


class Encoder(SharedEncoder[AEConfig | MTAEConfig], TabularLayerInitMixin):
    """
    Encoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig | MTAEConfig
    ) -> None:
        super().__init__(config=config)
        
        self.config = config
    
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""

        self.comp_head = nn.Linear(min(self.config.hidden_dims), self.config.latent_dim)
    
    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""

        self.init_head(self.comp_head, self.config.activation_en)


class Decoder(SharedDecoder[AEConfig | VAEConfig], TabularLayerInitMixin):
    """
    Decoder class for AE model.
    """

    def __init__(
        self,
        config: AEConfig | VAEConfig
    ) -> None:
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
            if isinstance(module, nn.Linear):
                self.init_head(module, self.config.activation_de)


class TabularAE(TabularEncodePassingMixin[AEConfig], TabularDecodePassingMixin[AEConfig], TabularBase[AEConfig], TabularFeatureForwardMixin, TabularReconScoringMixin):
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
        config: AEConfig
    ) -> None:
        super().__init__(config)
        self.config = config
        self.E = Encoder(config)
        self.D = Decoder(config)

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
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Computes normalized weighted per-sample score and optionally reduces to average or sum."""

        cont_score, cat_score, per_sample_score = self.recon_scoring(
            x_cont,
            x_cat,
            cont_recon,
            cat_logits,
            loss_weights,
            self.config.cat_dims,
            in_warmup
        )

        if reduction == "mean":
            return cont_score, cat_score, torch.mean(per_sample_score)
        elif reduction == "sum":
            return cont_score, cat_score, torch.sum(per_sample_score)
        else:
            return per_sample_score
