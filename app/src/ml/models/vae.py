import torch
import numpy as np
import torch.nn as nn

from .ae import Decoder
from .base import SharedVEncoder, TabularBase
from .configs import VAEConfig
from .mixins import TabularDecodePassingMixin, TabularVEncodePassMixin, TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin, TabularKLScoringMixin


class VEncoder(SharedVEncoder[VAEConfig]):
    """Encoder class for VAE model."""

    def __init__(
        self,
        config: VAEConfig
    ) -> None:
        super().__init__(config=config)
    
        self.config = config
    
    def make_comp_heads(self) -> None:
        """Adds final compression heads of encoder."""  
        self.mu_head = nn.Linear(min(self.config.hidden_dims), self.config.latent_dim)
        self.logvar_head = nn.Linear(min(self.config.hidden_dims), self.config.latent_dim)

    def init_comp_heads(self) -> None:
        """Initializes final compression heads of encoder."""
        self.init_head(self.mu_head, None)
        self.init_head(self.logvar_head, None)


class TabularVAE(TabularVEncodePassMixin[VAEConfig], TabularDecodePassingMixin[VAEConfig], TabularBase[VAEConfig], TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin, TabularKLScoringMixin):
    """
    Hybrid tabular VAE with:
      - cont and cat inputs
      - Embedding or OneHotEncoding for cat features
      - (optional) denoising for cont features
      - separate decoding heads for cont and cat features
      - unsupervised learning for anomaly scoring
      - metrics: mixed ELBO loss with KL divergence and recon error combining 
        Huber loss for cont and CE for cat features
    """

    def __init__(
        self,
        config: VAEConfig
    ) -> None:
        super().__init__(config)
        self.config = config
        self.E = VEncoder(config)
        self.D = Decoder(config)
        self.current_beta = 0.
    
    def forward(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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

        z, mu, logvar = self.encode(Xc, Xk, all_vars=True)
        cont_recon, cat_logits = self.decode(z)

        # for benchmarking kl_divergence clamp values
        # μ ~ N(0, 1) → mean near 0, std near 1
        # logvar near 0, between ~[-4, +4]
        if self.config.debug_kl_stats:
            print(
                f"[KL DIAGNOSTICS] μ mean={mu.mean().item():.4f} std={mu.std().item():.4f} | "
                f"logvar mean={logvar.mean().item():.4f} std={logvar.std().item():.4f}"
            )

        return cont_recon, cat_logits, mu, logvar

    def scoring(
        self,
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        cont_recon: torch.Tensor,
        cat_logits: dict[str, torch.Tensor],
        mu: torch.Tensor | None,
        logvar: torch.Tensor | None,
        loss_weights: dict[str, float],
        reduction: str = "mean"
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """
        Computes normalized weighted per-sample score and optionally reduces to
        average or sum.
        """
        cont_loss, cat_loss, recon_score = self.recon_scoring(
            x_cont,
            x_cat,
            cont_recon,
            cat_logits,
            loss_weights,
            self.config.cat_dims
        )

        kl_score = torch.zeros_like(recon_score)
        if mu is not None and logvar is not None:
            kl_score = self.kl_scoring(
                mu,
                logvar,
                logvar_clamp=self.config.logvar_clamp,
                use_free_bits=self.config.use_kl_clipping,
                free_bits=self.config.kl_clamp
            )
            
        per_sample_score = recon_score + self.current_beta * kl_score

        if reduction == "mean":
            return cont_loss, cat_loss, recon_score, kl_score, torch.mean(per_sample_score)
        elif reduction == "sum":
            return cont_loss, cat_loss, recon_score, kl_score, torch.sum(per_sample_score)
        else:
            return per_sample_score
    
    # -----------------------------
    # Utils
    # -----------------------------
    def set_beta_annealing(
        self, 
        epoch: int, 
    ) -> None:
        """
        Beta-annealing gives model time to learn reconstructions before forcing
        latent regularization. Dynamically adjusts beta for each epoch.
        """
        # linear schedule within 40% of epochs
        if self.config.beta_schedule == "linear":
            warmup = int(0.4 * self.config.num_epochs)
            factor = min(1.0, epoch / max(1, warmup))
            self.current_beta = self.config.beta * factor

        # cosine-cycle schedule
        elif self.config.beta_schedule == "cyclic":
            period = self.config.num_epochs // 4
            phase = (epoch % period) / period
            self.current_beta = self.config.beta * 0.5 * (1 - np.cos(np.pi * phase))

        # no annealing
        else:  
            self.current_beta = self.config.beta
