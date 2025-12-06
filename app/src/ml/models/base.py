import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union


class BaseTabularModel(nn.Module):
    """
    Base class for tabular autoencoder models with:
      - categorical embeddings
      - continuous + categorical fusion
      - activation factory
      - optional denoising on continuous inputs

    Subclasses must implement:
        encode(x_cont, x_cat) -> latent representation
        decode(z, temperature=1.0) -> (cont_recon, cat_logits)

    Subclasses may override:
        anomaly_score(...)
    """

    def __init__(
        self,
        num_cont: int,
        cat_dims: Dict[str, int],
        embedding_dim: Optional[int],
        continuous_noise_std: float,
        activation: str = "relu",
    ):
        super().__init__()

        self.num_cont = num_cont
        self.cat_dims = cat_dims
        self.continuous_noise_std = continuous_noise_std

        # -----------------------
        # Activation function
        # -----------------------
        self.activation = self._make_activation(activation)

        # -------------------------------
        # Embeddings
        # -------------------------------
        # Example cat_dims:
        # {"weekday":7, "daytype":2, "daytime":5, "month":12, "week":53}
        # For embedding dim: d = min( max(4, card//2), 16 )
        def emb_dim(card: int) -> int:
            return min(max(4, card // 2), 16)

        self.embeddings = nn.ModuleDict()
        self.emb_sizes = {}

        for name, card in cat_dims.items():
            dim = embedding_dim if embedding_dim is not None else emb_dim(card)
            self.embeddings[name] = nn.Embedding(card, dim)
            self.emb_sizes[name] = dim

        self.total_emb_dim = sum(self.emb_sizes.values())
        self.input_dim = num_cont + self.total_emb_dim

    # -------------------------------
    # Utilities
    # -------------------------------
    def _make_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(inplace=True),
        }
        if name not in activations:
            raise ValueError(f"[ERROR] Unknown activation: {name}.")
        return activations[name]

    def _embed(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical features."""
        parts = []
        for i, name in enumerate(self.cat_dims.keys()):
            # cast to long if required
            parts.append(self.embeddings[name](x_cat[:, i].long()))
        return torch.cat(parts, dim=1)

    def _maybe_noisy_cont(self, x_cont: torch.Tensor) -> torch.Tensor:
        """Noise injection for continuous features"""
        if self.training and self.continuous_noise_std > 0:
            noise = torch.randn_like(x_cont) * self.continuous_noise_std
            # Only add noise to continuous part
            return x_cont + noise
        return x_cont

    # -------------------------------
    # Abstract methods
    # -------------------------------
    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return latent variables for AE or (mu, logvar) for VAE."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor, temperature: float = 1.0):
        """
        Return continuous reconstruction (batch, num_cont) and categorical logits as {name: logits}.
        """
        raise NotImplementedError

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        return_cat: bool = True,
        temperature: float = 1.0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Full forward pass: returns cont recon and optionally cat logits.
        VAE models may override encode to return (mu, logvar, z)
        """
        z = self.encode(x_cont, x_cat)
        cont_recon, cat_logits = self.decode(z, temperature)
        if return_cat:
            return cont_recon, cat_logits
        return cont_recon

    # -------------------------------
    # Default anomaly score
    # -------------------------------
    def anomaly_score(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        cont_weight: float = 1.0,
        cat_weight: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generic reconstruction-based anomaly score.
        VAE subclasses must override to include KL or ELBO.
        """
        with torch.no_grad():
            cont_recon, cat_logits = self.forward(
                x_cont, 
                x_cat, 
                True, 
                temperature
            )

            # Continuous MSE per sample
            cont_error = ((cont_recon - x_cont) ** 2).mean(dim=1)
            
            if cat_weight <= 0.0 or len(self.cat_dims) == 0:
                return cont_error

            # Categorical average CE accross cat features 
            cat_error = torch.zeros_like(cont_error)
            n_cats = len(self.cat_dims)
            for i, name in enumerate(self.cat_dims.keys()):
                targets = x_cat[:, i].long()
                ce = F.cross_entropy(
                    cat_logits[name],
                    targets,
                    reduction="none"
                )
                cat_error += ce
            cat_error = cat_error / float(n_cats)

            # Weighted combination
            total_weight = cont_weight + cat_weight
            return (cont_weight * cont_error + cat_weight * cat_error) / total_weight