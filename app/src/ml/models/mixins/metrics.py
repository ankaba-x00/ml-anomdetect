import torch
import torch.nn.functional as F


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none"
):
    """
    Computes mean-squared loss.
    """

    return F.mse_loss(pred, target, reduction=reduction)

def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none"
):
    """
    Computes Huber loss.
    """

    return F.huber_loss(pred, target, reduction=reduction)

def pinball_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """
    Computes average pinball loss for a given quantile.
    """
    e = target - pred
    return torch.maximum(q * e, (q - 1) * e)

def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
    reduction: str = "none"
):
    """
    Computes pinball loss over Q quantiles.
    """

    losses = []
    for i, q in enumerate(quantiles):
        losses.append(pinball_loss(pred[:,i], target, q))

    if reduction == "mean":
        return torch.stack(losses, dim=0).mean(dim=0)
    elif reduction == "sum":
        return torch.stack(losses, dim=0).sum(dim=0)
    return torch.stack(losses)

def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none"
):
    """
    Computes cross-entropy loss.
    """

    return F.cross_entropy(logits, target, reduction=reduction)

def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weights: torch.Tensor | None = None,
    reduction: str = "none"
):
    """
    Computes focal loss at given quantile.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    target_logp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    target_p = probs.gather(1, target.unsqueeze(1)).squeeze(1)

    focal_factor = (1.0 - target_p)**gamma
    loss = -focal_factor * target_logp

    if weights is not None:
        type_weights = weights.gather(0, target)
        loss = loss * type_weights

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss