import json, torch
from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import torch.nn as nn
import torch.nn.functional as F
from app.src.ml.models.base import BaseTabularModel

################################################
##                  CONFIG                    ##
################################################

@dataclass
class VAEConfig:
    num_cont: int
    cat_dims: dict[str, int]
    latent_dim: int = 8   # bottleneck size
    hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.1
    embedding_dim: Optional[int] = None
    continuous_noise_std: float = 0.0
    residual_strength: float = 0.0
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    patience: int = 5
    gradient_clip: float = 1.0
    optimizer: str = "adam"
    lr_scheduler: str = "none"
    use_lr_scheduler: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # anomaly / ELBO related
    anomaly_threshold: Optional[float] = None
    beta: float = 1.0 # weight for KL term in ELBO
    temperature: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "VAEConfig":
        return VAEConfig(**json.loads(s))


################################################
##     TABULAR VAE WITH EMBEDDINGS + NOISE    ##
################################################

class TabularVAE(BaseTabularModel):
    """
    Tabular variational autoencoder with:
      - continuous + categorical inputs
      - learned categorical embeddings
      - optional denoising on continuous inputs
      - KL + ELBO + anomaly scoring
    """

    def __init__(
        self,
        num_cont: int,
        cat_dims: dict[str, int],
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        embedding_dim: Optional[int] = None,
        continuous_noise_std: float = 0.0,
        residual_strength: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            num_cont=num_cont,
            cat_dims=cat_dims,
            embedding_dim=embedding_dim,
            continuous_noise_std=continuous_noise_std,
            activation=activation,
        )

        self.latent_dim = latent_dim
        self.residual_strength = residual_strength
    
        # optional residual projection to latent space
        if residual_strength > 0:
            self.residual_proj = nn.Linear(self.input_dim, latent_dim)
        else:
            self.residual_proj = None

        # -------------------------------
        # Encoder
        # -------------------------------
        self.encoder_layers = nn.ModuleList()
        prev = self.input_dim

        for h in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout),
                )
            )
            prev = h

        # latent mean and log-variance heads
        self.mu_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)

        # -------------------------------
        # Decoder
        # -------------------------------
        self.decoder_layers = nn.ModuleList()
        prev = latent_dim

        for h in reversed(hidden_dims):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    self.activation,
                    nn.Dropout(dropout),
                )
            )
            prev = h

        # Continuous reconstruction head
        self.cont_recon = nn.Linear(prev, num_cont)

        # Categorical reconstruction heads (logits)
        self.cat_recon_heads = nn.ModuleDict()
        for name, card in cat_dims.items():
            self.cat_recon_heads[name] = nn.Linear(prev, card)

        # weight initialization
        self._init_weights()
        self._init_vae_heads()
        # TODO: Benchmark AE/VAE learning curves with vs. without initialization
        
        # -------------------------------
        # Benchmark toggles
        # -------------------------------
        self.debug_kl_stats = False

    # -------------------------------
    # Utilities
    # -------------------------------
    def _init_vae_heads(self):
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    # -------------------------------
    # Encode/Decode
    # -------------------------------
    def encode(
            self, 
            x_cont: torch.Tensor, 
            x_cat: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode (x_cont, x_cat) -> (mu, logvar)
        """
        x_emb = self._embed(x_cat)

        # Apply noise to continuous part (denoising AE)
        x_cont_n = self._maybe_noisy_cont(x_cont)
        
        # Concatenate into unified representation
        x = torch.cat([x_cont_n, x_emb], dim=1)

        # Forward through encoder
        h = x
        for layer in self.encoder_layers:
            h = layer(h)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # for benchmarking kl_divergence clamp values
        # μ ~ N(0, 1) → mean near 0, std near 1
        # logvar near 0, between ~[-4, +4]
        if getattr(self, "debug_kl_stats", False):
            print(
                f"[KL DIAGNOSTICS] μ mean={mu.mean().item():.4f} std={mu.std().item():.4f} | "
                f"logvar mean={logvar.mean().item():.4f} std={logvar.std().item():.4f}"
            )

        # (residual_strength applies later to sampled z)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std, eps ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
            self, 
            z: torch.Tensor, 
            temperature: float = 1.0
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode latent z -> continuous reconstruction + categorical logits."""
        h = z
        for layer in self.decoder_layers:
            h = layer(h)

        # Continuous reconstruction
        cont_recon = self.cont_recon(h)

        # Categorical logits (optionally temperature-scaled)
        cat_logits = {}
        for name in self.cat_dims.keys():
            logits = self.cat_recon_heads[name](h)
            if temperature != 1.0:
                logits = logits / temperature
            cat_logits[name] = logits

        return cont_recon, cat_logits

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        return_cat: bool = True,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """
        Full forward pass: returns cont recon and optional cat_logits, as well as mu, logvar.
        """
        mu, logvar = self.encode(x_cont, x_cat)
        z = self.reparameterize(mu, logvar)

        # Optional residual connection in latent space
        if self.residual_strength > 0.0 and hasattr(self, "residual_proj") and self.residual_proj is not None:
            x_emb = self._embed(x_cat)
            x = torch.cat([x_cont, x_emb], dim=1)
            z = z + self.residual_strength * self.residual_proj(x)

        cont_recon, cat_logits = self.decode(z, temperature)

        if return_cat:
            return cont_recon, cat_logits, mu, logvar
        else:
            return cont_recon, None, mu, logvar

    # -------------------------------
    # KL / ELBO LOSS      
    # -------------------------------
    @staticmethod
    def kl_divergence(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        logvar_clip: Optional[tuple[float, float]] = (-20, 20),
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        KL(q(z|x) || p(z)) with p(z) ~ N(0, I)

        per-sample KL:
          KL_i = 0.5 * sum_j (exp(logvar) + mu^2 - 1 - logvar)
        """
        # TODO: check gradients, if extremely large/small gradients clamp!
        # logvar = torch.clamp(logvar, min=logvar_clip[0], max=logvar_clip[1])
        var = torch.exp(logvar)
        # TODO: add eps to avoid exp(0) = 1 issues OR clamp; ergo numerical stability
        # var = logvar.exp() + eps 
        # var = torch.clamp(var, min=1e-12, max=1e6)
        kl = -0.5 * (1 + logvar - mu.pow(2) - var)
        kl = kl.sum(dim=1)  # sum over latent dims -> shape [batch]

        if reduction == "none":
            return kl
        elif reduction == "mean":
            return kl.mean()
        elif reduction == "sum":
            return kl.sum()
        else:
            raise ValueError(f"[ERROR] Unknown KL reduction: {reduction}")

    def reconstruction_error_per_sample(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        cont_weight: float = 1.0,
        cat_weight: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute pure per-sample reconstruction error combining:
          - continuous MSE
          - categorical cross-entropy averaged over categories
        Use when recon-only optimization for tuning to debug recon quality or compate AE/VAE recon capabilities!
        """
        cont_recon, cat_logits, _, _ = self.forward(
            x_cont, x_cat, return_cat=True, temperature=temperature
        )

        # Continuous MSE per sample
        cont_err = ((cont_recon - x_cont) ** 2).mean(dim=1)

        if cat_weight <= 0.0 or len(self.cat_dims) == 0:
            return cont_err

        # Categorical CE averaged over categorical features
        cat_err = torch.zeros_like(cont_err)
        n_cats = len(self.cat_dims)

        for i, name in enumerate(self.cat_dims.keys()):
            logits = cat_logits[name]
            targets = x_cat[:, i].long()
            ce = F.cross_entropy(logits, targets, reduction="none")  # [batch]
            cat_err = cat_err + ce

        cat_err = cat_err / float(n_cats)

        total_weight = cont_weight + cat_weight
        # to prevent exploding loss
        if total_weight <= 0:
            total_weight = 1.0
            cont_weight = 1.0
            cat_weight = 0.0
        return (cont_weight * cont_err + cat_weight * cat_err) / total_weight

    def elbo_loss(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        beta: float = 1.0,
        cont_weight: float = 1.0,
        cat_weight: float = 0.0,
        temperature: float = 1.0,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO components:
          recon_loss  = E_q[ -log p(x|z) ]  (here: weighted recon error)
          kl_loss     = KL(q(z|x) || p(z))
          total_loss  = recon_loss + beta * kl_loss

        Returns (total_loss, recon_loss, kl_loss)
        """
        cont_recon, cat_logits, mu, logvar = self.forward(
            x_cont, x_cat, return_cat=True, temperature=temperature
        )

        # -------------------------------
        # Reconstruction error (per-sample)
        # -------------------------------
        cont_err = ((cont_recon - x_cont) ** 2).mean(dim=1)  # [batch]

        if cat_weight > 0.0 and len(self.cat_dims) > 0:
            cat_err = torch.zeros_like(cont_err)
            n_cats = len(self.cat_dims)
            for i, name in enumerate(self.cat_dims.keys()):
                logits = cat_logits[name]
                targets = x_cat[:, i].long()
                ce = F.cross_entropy(logits, targets, reduction="none")
                cat_err += ce
            cat_err = cat_err / float(n_cats)
        else:
            cat_err = torch.zeros_like(cont_err)

        total_weight = cont_weight + cat_weight
        # to prevent exploding loss
        if total_weight <= 0:
            total_weight = 1.0
            cont_weight = 1.0
            cat_weight = 0.0
        recon_per_sample = (cont_weight * cont_err + cat_weight * cat_err) / max(
            total_weight, 1e-8
        )

        # aggregate reconstruction loss
        if reduction == "mean":
            recon_loss = recon_per_sample.mean()
        elif reduction == "sum":
            recon_loss = recon_per_sample.sum()
        elif reduction == "none":
            recon_loss = recon_per_sample
        else:
            raise ValueError(f"[ERROR] Unknown reduction: {reduction}")

        # KL loss
        kl_per_sample = self.kl_divergence(mu, logvar, reduction="none")

        if reduction == "mean":
            kl_loss = kl_per_sample.mean()
        elif reduction == "sum":
            kl_loss = kl_per_sample.sum()
        elif reduction == "none":
            kl_loss = kl_per_sample
        else:
            raise ValueError(f"[ERROR] Unknown reduction: {reduction}")

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # -------------------------------
    # Anomaly score
    # -------------------------------
    def anomaly_score(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        cont_weight: float = 1.0,
        cat_weight: float = 0.0,
        beta: float = 1.0,
        temperature: float = 1.0,
        score_type: str = "elbo",
    ) -> torch.Tensor:
        """
        Compute per-sample anomaly scores; score_type options:
          - recon : reconstruction error only
          - kl    : KL divergence only
          - elbo  : recon + beta * KL
        """
        self.eval()
        with torch.no_grad():
            cont_recon, cat_logits, mu, logvar = self.forward(
                x_cont, x_cat, return_cat=True, temperature=temperature
            )

            # recon per sample
            cont_err = ((cont_recon - x_cont) ** 2).mean(dim=1)

            if cat_weight > 0.0 and len(self.cat_dims) > 0:
                cat_err = torch.zeros_like(cont_err)
                n_cats = len(self.cat_dims)
                for i, name in enumerate(self.cat_dims.keys()):
                    logits = cat_logits[name]
                    targets = x_cat[:, i].long()
                    ce = F.cross_entropy(logits, targets, reduction="none")
                    cat_err += ce
                cat_err = cat_err / float(n_cats)
            else:
                cat_err = torch.zeros_like(cont_err)

            total_weight = cont_weight + cat_weight
            # to prevent exploding loss
            if total_weight <= 0:
                total_weight = 1.0
                cont_weight = 1.0
                cat_weight = 0.0
            recon = (cont_weight * cont_err + cat_weight * cat_err) / max(
                total_weight, 1e-8
            )

            kl = self.kl_divergence(mu, logvar, reduction="none")

            if score_type == "recon":
                return recon
            elif score_type == "kl":
                return kl
            elif score_type == "elbo":
                return recon + beta * kl
            else:
                raise ValueError(
                    f"[ERROR] Unknown score_type: {score_type}. "
                    f"Use 'recon', 'kl', or 'elbo'."
                )
