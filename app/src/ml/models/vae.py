import torch
import numpy as np
import torch.nn as nn

from .base import BaseTabularEncoder, BaseTabularDecoder, BaseTabularPredictor
from .configs import VAEConfig, MTVAEConfig
from .mixins import TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin, TabularKLScoringMixin


class VEncoder(BaseTabularEncoder):
    """
    Encoder class for VAE model.
    """

    def __init__(
        self,
        config: VAEConfig | MTVAEConfig
    ):
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


class VDecoder(BaseTabularDecoder):
    """
    Decoder class for VAE model.
    """

    def __init__(
        self,
        config: VAEConfig
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


class TabularVAE(BaseTabularPredictor, TabularFeatureEncodeMixin, TabularFeatureForwardMixin, TabularReconScoringMixin, TabularKLScoringMixin):
    """
    Hybrid tabular VAE with:
      - cont and cat inputs
      - Embedding or OneHotEncoding for cat features
      - (optional) denoising for cont features
      - separate decoding heads for cont and cat features
      - unsupervised learning for anomaly scoring
      - metrics: mixed ELBO loss with KL divergence and recon error combining Huber loss for cont and CE for cat features
    """

    def __init__(
        self,
        config: VAEConfig
    ):
        super().__init__()
        self.config = config
        self.E = VEncoder(config)
        self.D = VDecoder(config)
        self.current_beta = 0.
    
    def encode(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        param_out: bool = False
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

        mu, logvar = self._parametrize(x)
        z = self._reparametrize(mu, logvar)

        if not param_out:
            return z
        return z, mu, logvar
    
    def _parametrize(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parametrizes latent distribution determistically. Input data is mapped to the variational distribution and mean (mu) and log-variance (logvar) returned.
        """
        
        h = x
        for layer in self.E.encoder_layers:
            h = layer(h)
        
        mu = self.E.mu_head(h)
        logvar = self.E.logvar_head(h)

        # for benchmarking kl_divergence clamp values
        # μ ~ N(0, 1) → mean near 0, std near 1
        # logvar near 0, between ~[-4, +4]
        if self.config.debug_kl_stats:
            print(
                f"[KL DIAGNOSTICS] μ mean={mu.mean().item():.4f} std={mu.std().item():.4f} | "
                f"logvar mean={logvar.mean().item():.4f} std={logvar.std().item():.4f}"
            )
        
        return mu, logvar

    def _reparametrize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparametrizes to stochastically sample z from the distribution via z = mu + eps ⊙ std with eps ~ N(0, I).
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
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
            if self.config.temperature != 0.:
                logits = logits / self.config.temperature
            cat_logits[name] = logits
        
        return cont_recon, cat_logits
    
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

        z, mu, logvar = self.encode(Xc, Xk, param_out=True)
        cont_recon, cat_logits = self.decode(z)

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
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """Computes normalized weighted per-sample score and optionally reduces to average or sum."""

        cont_loss, cat_loss, recon_score = self.recon_scoring(
            x_cont,
            x_cat,
            cont_recon,
            cat_logits,
            self.config.cat_dims,
            loss_weights,
        )

        kl_score = self.kl_scoring(
            mu,
            logvar,
            logvar_clamp=self.config.logvar_clamp,
            use_free_bits=self.config.use_kl_clipping,
            free_bits=self.config.kl_clamp
        )
            
        per_sample_score = recon_score + self.current_beta * kl_score

        if reduction == "mean":
            return cont_loss, cat_loss, recon_score, kl_score, per_sample_score.mean()
        else:
            return per_sample_score
    
    # -----------------------------
    # Utils
    # -----------------------------
    def set_beta_annealing(
        self, 
        epoch: int, 
    ) -> None:
        """Beta-annealing gives model time to learn reconstructions before forcing latent regularization. Dynamically adjusts beta for each epoch."""

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
