import numpy as np
import torch
import torch.nn as nn

from .ae import Encoder, TabularAE
from .base import BaseTabularDecoder
from .configs import MTAEConfig
from .mixins import TabularMTScoringMixin


class MTDecoder(BaseTabularDecoder):
    """
    Decoder class for MTAE model.
    """

    def __init__(
        self,
        config: MTAEConfig
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
        
        self.l3_head = nn.Linear(dim, len(self.config.quantiles))

        self.l7_head = nn.Linear(dim, len(self.config.quantiles))

        self.at_head = nn.Sequential(
            nn.Linear(dim, dim//2),
            self.pick_act_func(self.config.activation_de_cls),
            nn.Linear(dim//2, self.config.n_attack_types)
        )
    
    def init_recon_heads(self) -> None:
        """Initializes final reconstruction heads of decoder."""

        for module in self.cat_recon_heads.children():
            self.init_head(module, self.config.activation_de)
        
        for head in [self.cont_recon_head, self.l3_head, self.l7_head]:
            self.init_head(head, self.config.activation_de)
        
        for module in self.at_head.children():
            if isinstance(module, nn.Linear):
                self.init_head(module, self.config.activation_de_cls)


class MTTabularAE(TabularAE, TabularMTScoringMixin):
    """
    Hybrid tabular autoencoder with:
      - cont and cat inputs
      - Embedding or OneHotEncoding for cat features
      - (optional) noise injection for cont and cat features
      - L3/L7 regression heads and attack type classification head 
      - supervised learning for L3/L7 and attack type prediction 
      - metrics: pinball loss for L3/L7 and focal or CE for attack type
    """

    def __init__(
        self,
        config: MTAEConfig
    ):
        super().__init__(
            config=config
        )
        self.config = config
        self.E = Encoder(config)
        self.D = MTDecoder(config)
        self.current_alpha = 0.0

    def decode(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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

    def forward(
        self,
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor    
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        cont_recon, cat_logits, l3_pred, l7_pred, at_logits = self.decode(z)

        return cont_recon, cat_logits, l3_pred, l7_pred, at_logits
    
    def scoring(
        self,
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        y3: torch.Tensor,
        y7:torch.Tensor,
        ya: torch.Tensor,
        cont_recon: torch.Tensor,
        cat_logits: dict[str, torch.Tensor],
        l3_pred: torch.Tensor,
        l7_pred: torch.Tensor,
        at_logits: torch.Tensor,
        loss_weights: dict[str, float],
        attack_type_weights: torch.Tensor,
        in_warmup: bool = False,
        reduction: str = "mean"
    ) -> None:
        """Computes per-sample reconstruction error, quantile and focal loss and optionally reduces to average or sum."""
        
        cont_loss, cat_loss, recon_score = self.recon_scoring(
            x_cont,
            x_cat,
            cont_recon,
            cat_logits,
            self.config.cat_dims,
            loss_weights,
        )

        l3_loss, l7_loss, at_loss, total_score = self.hybrid_scoring(
            y3,
            y7,
            ya,
            l3_pred,
            l7_pred,
            at_logits,
            self.config.quantiles,
            self.config.focal_gamma,
            loss_weights,
            attack_type_weights
        )
        
        if in_warmup:
            per_sample_score = recon_score
        else:
            per_sample_score = recon_score + self.current_alpha * total_score

        if reduction == "mean":
            return cont_loss, cat_loss, recon_score, l3_loss, l7_loss, at_loss, per_sample_score.mean()
        elif reduction == "sum":
            return cont_loss, cat_loss, recon_score, l3_loss, l7_loss, at_loss, per_sample_score.sum()
        else:
            return per_sample_score
    
    # -----------------------------
    # Utils
    # -----------------------------
    def compute_attack_type_weights(
        self,
        ya: np.ndarray,
        n_attack_types: int,
        type_max_ratio: float = 20.0,
        eps: float = 1.0
    ) -> np.ndarray | torch.Tensor:
        """Computes inverse-frequency type weights for scoring attack types."""

        counts = np.bincount(ya, minlength=n_attack_types).astype(np.float32)
        counts = np.maximum(counts, eps)

        weights = counts.sum() / (n_attack_types * counts)
        weights = np.clip(weights, 1.0 / type_max_ratio, type_max_ratio)
        weights = weights / weights.mean()

        return torch.tensor(weights, dtype=torch.float32)
    
    def set_alpha(
        self,
        epoch: int,
        in_warmup: bool = False,
        in_stepup: bool = False,
    ) -> None:
        """Alpha allows model to learn reconstructions before forcing latent regularization. Dynamically adjusts alpha for each epoch."""

        if self.config.stepup_epochs > 0:
            if in_warmup:
                self.current_alpha = 0.0
            elif in_stepup:
                increment = self.config.alpha / self.config.stepup_epochs
                self.current_alpha = float(increment * (epoch - self.config.warmup_epochs + 1))
            else:
                self.current_alpha = self.config.alpha
        else:
            self.current_alpha = self.config.alpha
