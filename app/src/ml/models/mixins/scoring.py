import torch

from . import huber_loss, cross_entropy, quantile_loss, focal_loss


class TabularReconScoringMixin:
    """
    Stateless Helper Mixin providing functionality for computing reconstruction errors for hybrid features.
    """
    
    @staticmethod
    def recon_scoring(
        x_cont: torch.Tensor, 
        x_cat: torch.Tensor,
        cont_recon: torch.Tensor,
        cat_logits: dict[str, torch.Tensor],
        cat_dims: dict[str, int],
        loss_weights: dict[str, float],
        in_warmup: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Huber loss for cont features, CE for cat features and normalized weighted per-sample loss.
        """

        # Cont Huber loss per-sample and feature-averaged
        cont_loss = huber_loss(cont_recon, x_cont).mean(dim=1)

        # Cat CE per-sample and feature-averaged
        cat_loss = torch.zeros_like(cont_loss)
        cat_names = cat_dims.keys()
        for i, name in enumerate(cat_names):
            ce = cross_entropy(
                cat_logits[name],
                x_cat[:, i].long(),
            )
            cat_loss += ce
        cat_loss = cat_loss / float(len(cat_names))
        
        cont_w = loss_weights["cont_w"]
        cat_w= loss_weights["cat_w"]

        if in_warmup:
            total_loss = cat_loss
        else:
            total_loss = cont_w * cont_loss + cat_w * cat_loss

        return cont_loss, cat_loss, total_loss


class TabularKLScoringMixin:
    """
    Stateless Helper Mixin providing functionality for computing elbo loss.
    """
    
    @staticmethod
    def kl_scoring(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        logvar_clamp: float = 5.0,
        use_free_bits: bool = False,
        free_bits: float = 0.001
    ) -> torch.Tensor:
        """
        Computes per-sample KL divergence as a loss term to measure how much the learned latent distribution differs from a standard normal distribution. Effectively, this term acts as a regularizer keeping the latent space smooth and continuous.
        """
        
        logvar = torch.clamp(logvar, min=-logvar_clamp, max=logvar_clamp)
        var = torch.exp(logvar)
        kl = -0.5 * (1 + logvar - mu.pow(2) - var)
        if use_free_bits:
            kl_clipped = torch.clamp(kl, min=free_bits)
            return kl_clipped.mean(dim=1)
        else:
            return kl.mean(dim=1)
        

class TabularMTScoringMixin:
    """
    Stateless Helper Mixin providing functionality for computing focal and quantile loss for hybrid task scoring.
    """

    @staticmethod
    def hybrid_scoring(
        y3: torch.Tensor,
        y7:torch.Tensor,
        ya: torch.Tensor,
        l3_pred: torch.Tensor,
        l7_pred: torch.Tensor,
        at_logits: torch.Tensor,
        quantiles: tuple[float],
        gamma: float,
        loss_weights: dict[str, float],
        attack_type_weights: torch.Tensor
    ):
        loss_l3 = quantile_loss(
            l3_pred,
            y3,
            quantiles,
            reduction="mean"
        )

        loss_l7 = quantile_loss(
            l7_pred,
            y7,
            quantiles,
            reduction="mean"
        )

        loss_att = focal_loss(
            at_logits,
            ya,
            gamma,
            attack_type_weights,
        )

        l3_w = loss_weights["l3_w"]
        l7_w = loss_weights["l7_w"]
        at_w = loss_weights["at_w"]

        total_loss = l3_w * loss_l3 + l7_w * loss_l7 + at_w * loss_att

        return loss_l3, loss_l7, loss_att, total_loss
