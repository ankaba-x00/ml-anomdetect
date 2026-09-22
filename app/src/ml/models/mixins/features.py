import torch
import torch.nn as nn
import torch.nn.functional as F
        

class TabularFeatureEncodeMixin:
    """
    Stateless Feature Mixin providing functionality preprosessing categorical 
    features via encoding or embedding.
    """

    def prepare_input(
        self, 
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        use_embedding: bool,
        cat_dims: dict[str, int],
        embeddings: nn.ModuleDict | None = None
    ) -> torch.Tensor:
        """Prepares categorical input tensors via embedding or encoding."""
        if use_embedding and embeddings is not None:
            x_cat_e = TabularFeatureEncodeMixin._embed(
                x_cat, 
                cat_dims, 
                embeddings
            )
        else:
            x_cat_e = TabularFeatureEncodeMixin._encod(
                x_cat, 
                cat_dims
            )
        return torch.cat([x_cont, x_cat_e], dim=1)

    @staticmethod
    def _embed(
        x_cat: torch.Tensor, 
        cat_dims: dict[str, int], 
        embeddings: nn.ModuleDict
    ) -> torch.Tensor:
        """Embeds cat features using Embedding."""
        parts = []
        for i, name in enumerate(cat_dims.keys()):
            parts.append(embeddings[name](x_cat[:, i].long()))
        
        return torch.cat(parts, dim=1)
    
    @staticmethod
    def _encod(
        x_cat: torch.Tensor, 
        cat_dims: dict[str, int]
    ) -> torch.Tensor:
        """Encodes cat features via OneHotEncoding."""
        parts = []
        for i, card in enumerate(cat_dims.values()):
            parts.append(F.one_hot(x_cat[:, i].long(), num_classes=card).float())
        
        return torch.cat(parts, dim=1)


class TabularFeatureForwardMixin:
    """
    Stateless Helper Mixin providing functionality for noise injection in 
    hybrid tabular autoencoders during forward pass.
    """

    @staticmethod
    def _noise_injection(
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        noise_gauss_std: float,
        noise_mask_prob: float
    ) -> tuple[torch.Tensor, ...]:
        """Injects noise to continuous and categorical features enabling 
        denoising autoencoder."""
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
