import torch
import torch.nn as nn
from typing import TYPE_CHECKING, TypeVar, Generic
from .features import TabularFeatureEncodeMixin

if TYPE_CHECKING:
    from app.src.ml.models.base import SharedDecoder, SharedEncoder, SharedMTDecoder, SharedVEncoder
    from app.src.ml.models.configs.cbase import SharedDecoderConfig, SharedEncoderConfig, SharedMTDecoderConfig, SharedVEncoderConfig


SharedEncoderConfigT = TypeVar("SharedEncoderConfigT", bound="SharedEncoderConfig")

class TabularEncodePassingMixin(TabularFeatureEncodeMixin, Generic[SharedEncoderConfigT]):
    """
    Stateless Helper Mixin providing functionality for passing data through encoder.
    """

    E: "SharedEncoder"
    config: "SharedEncoderConfigT"

    def encode(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor
    ) -> torch.Tensor:
        """Preprocesses and passes input through encoder and returns latent variables."""

        embeddings = self.E.embeddings if self.config.use_embedding else None
        x = self.prepare_input(
            x_cont, 
            x_cat, 
            self.config.use_embedding, 
            self.config.cat_dims, 
            embeddings
        )
        h = x
        for layer in self.E.encoder_layers:
            h = layer(h)
        
        z: torch.Tensor = self.E.comp_head(h)

        return z


SharedVEncoderConfigT = TypeVar("SharedVEncoderConfigT", bound="SharedVEncoderConfig")

class TabularVEncodePassMixin(TabularFeatureEncodeMixin, Generic[SharedVEncoderConfigT]):
    """
    Stateless Helper Mixin providing functionality for passing data through decoder.
    """
    
    E: "SharedVEncoder"
    config: "SharedVEncoderConfigT"

    def encode(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        all_vars: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Passes input through encoder and returns latent variables."""
        
        embeddings = self.E.embeddings if self.config.use_embedding else None
        x = self.prepare_input(
            x_cont, 
            x_cat, 
            self.config.use_embedding, 
            self.config.cat_dims, 
            embeddings
        )

        mu, logvar = self._parametrize(x)
        z = self._reparametrize(mu, logvar)

        if all_vars:
            return z, mu, logvar
        return z
    
    def _parametrize(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Parametrizes latent distribution determistically. Input data is mapped to the variational distribution and mean (mu) and log-variance (logvar) returned.
        """
        
        h = x
        for layer in self.E.encoder_layers:
            h = layer(h)
        
        mu = self.E.mu_head(h)
        logvar = self.E.logvar_head(h)
        
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


SharedDecoderConfigT = TypeVar("SharedDecoderConfigT", bound="SharedDecoderConfig")

class TabularDecodePassingMixin(Generic[SharedDecoderConfigT]):
    """
    Stateless Helper Mixin providing functionality for passing data through decoder.
    """

    D: "SharedDecoder"
    config: "SharedDecoderConfigT"

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


SharedMTDecoderConfigT = TypeVar("SharedMTDecoderConfigT", bound="SharedMTDecoderConfig")

class TabularMTDecodePassMixin(Generic[SharedMTDecoderConfigT]):
    """
    Stateless Helper Mixin providing functionality for passing data through multi-task decoder.
    """

    D: "SharedMTDecoder"
    config: "SharedMTDecoderConfigT"

    def decode(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        l3_recon = self._enforce_monotonicity(self.D.l3_head, h)
        l7_recon = self._enforce_monotonicity(self.D.l7_head, h)
        at_logits = self.D.at_head(h)
        
        return cont_recon, cat_logits, l3_recon, l7_recon, at_logits
    
    def _enforce_monotonicity(
        self, 
        module: nn.Module,
        feat: torch.Tensor
    ) -> torch.Tensor:
        """Enforces monotonicity along quantile axis to avoid quantile crossover."""
        
        head = module(feat)
        softplus = nn.Softplus()

        base = head[:, :1]
        step_offsets = softplus(head[:, 1:])

        quantile_matrix = torch.cat([base, step_offsets], dim=1)

        return torch.cumsum(quantile_matrix, dim=1)