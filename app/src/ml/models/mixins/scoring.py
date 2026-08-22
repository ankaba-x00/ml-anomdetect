import torch
import torch.nn.functional as F

class TabularReconScoringMixin:
    """
    Stateless Helper Mixin providing functionality for computing reconstruction errors.
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

        # Cont Huber loss per sample
        cont_loss = F.huber_loss(cont_recon, x_cont, reduction="none").mean(dim=1)

        # Cat CE per sample
        cat_loss = torch.zeros_like(cont_loss)
        cat_names = cat_dims.keys()
        for i, name in enumerate(cat_names):
            ce = F.cross_entropy(
                cat_logits[name],
                x_cat[:, i].long(),
                reduction="none"
            )
            cat_loss += ce
        cat_loss = cat_loss / float(len(cat_names))
    
        w_cont, w_cat = loss_weights["cont_w"], loss_weights["cat_w"]

        if in_warmup:
            total_loss = cat_loss
        else:
            total_loss = (w_cont * cont_loss + w_cat * cat_loss) / (w_cont + w_cat)

        return cont_loss, cat_loss, total_loss


class TabularKLScoringMixin:
    """
    Stateless Helper Mixin providing functionality for computing elbo loss.
    """
    
    @staticmethod
    def kl_scoring(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        logvar_clip: tuple[float, float] = (-20., 20.),
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Computes KL divergence as a loss term to measure how much the learned latent distribution differs from a standard normal distribution. Effectifly, this term acts as a regularizer keeping the latent space smooth and continuous.
        """
        # TODO: check gradients, if extremely large/small gradients clamp!
        logvar = torch.clamp(logvar, min=logvar_clip[0], max=logvar_clip[1])
        var = torch.exp(logvar)
        # TODO: add eps to avoid exp(0) = 1 issues OR clamp; ergo numerical stability
        # var = logvar.exp() + eps 
        # var = torch.clamp(var, min=1e-12, max=1e6)
        kl = -0.5 * (1 + logvar - mu.pow(2) - var)
        # sum over latent dims -> shape [batch]
        return kl.sum(dim=1)
        
