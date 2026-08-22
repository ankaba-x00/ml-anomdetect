import torch
import torch.nn as nn
import torch.nn.functional as F
        

class TabularFeatureEncodeMixin:
    """
    Stateless Helper Mixin providing functionality for categorical feature encoding for tabular autoencoders during encoding phase.
    """

    @staticmethod
    def _embed(
        x_cat: torch.Tensor, 
        cat_dims: dict[str, int], 
        embeddings: nn.ModuleDict
    ) -> torch.Tensor:
        """Embed cat features using Embedding."""

        parts = []
        for i, name in enumerate(cat_dims.keys()):
            parts.append(embeddings[name](x_cat[:, i].long()))
        
        return torch.cat(parts, dim=1)
    
    @staticmethod
    def _encod(
        x_cat: torch.Tensor, 
        cat_dims: dict[str, int]
    ) -> torch.Tensor:
        """Encode cat features via OneHotEncoding."""
        
        parts = []
        for i, card in enumerate(cat_dims.values()):
            parts.append(F.one_hot(x_cat[:, i].long(), num_classes=card).float())
        
        return torch.cat(parts, dim=1)


class TabularFeatureForwardMixin:
    """
    Stateless Helper Mixin providing functionality for noise injection in hybrid tabular autoencoders during forward pass.
    """

    @staticmethod
    def _noise_injection(
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        noise_gauss_std: float,
        noise_mask_prob: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject noise to continuous and categorical features enabling denoising autoencoder"""

        # Cont: adds Gaussian noise
        if noise_gauss_std > 0:
            noise = torch.randn_like(x_cont) * noise_gauss_std
            x_cont_noisy = x_cont + noise
        else:
            x_cont_noisy = x_cont

        # Cat: random masking noise (replaces values with low probability)
        if noise_mask_prob > 0:
            x_cat_noisy = x_cat.clone()
            mask = torch.randn_like(x_cat.float()) < noise_mask_prob
            x_cat_noisy[mask] = 0
        else:
            x_cat_noisy = x_cat

        return x_cont_noisy, x_cat_noisy
