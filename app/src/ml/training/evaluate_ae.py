import numpy as np
import torch

from .anomaly_utils import get_threshold, get_anomaly_mask, find_anomalies
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE


def reconstruction(
    model: TabularAE | TabularVAE,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    loss_weights: dict[str, float],
    temperature: float = 1.0,
    beta: float = 1.0,
    device: str = "cpu"
) -> np.ndarray:
    """Computes per-sample reconstruction error normalized by features."""

    model.eval()

    Xc = torch.from_numpy(X_cont).to(device)
    Xk = torch.from_numpy(X_cat).to(device)

    model.config.temperature = temperature

    with torch.no_grad():
        if isinstance(model, TabularVAE):
            cont_recon, cat_logits, mu, logvar = model(Xc, Xk)
            scores = model.scoring(
                Xc, 
                Xk,
                cont_recon,
                cat_logits,
                mu,
                logvar,
                loss_weights,
                reduction="none"
            )
        elif isinstance(model, TabularAE):
            cont_recon, cat_logits = model(Xc, Xk)
            scores = model.scoring(
                Xc,
                Xk,
                cont_recon,
                cat_logits,
                loss_weights, 
                reduction="none"
            )

    return scores.cpu().numpy()

def apply_model(
    model: TabularAE | TabularVAE,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    loss_weights: dict[str, float],
    device: str = "cpu",
    method: str = "p99",
    temperature: float = 1.0,
    beta: float = 1.0,
    min_length: int = 1,
    merge_gap: int = 0,
    threshold: float | None = None
) -> dict[str, np.ndarray]:
    """Applies autoencoder and returns reconstruction errors, threshold, anomaly mask and anomaly intervals."""

    scores = reconstruction(
        model,
        X_cont,
        X_cat,
        loss_weights,
        temperature,
        beta,
        device
    )

    if threshold is None:
        threshold = get_threshold(method, scores)
    mask = get_anomaly_mask(scores, threshold)
    intervals = find_anomalies(
        mask,
        min_length,
        merge_gap,
    )
    starts = np.array([s for s, _ in intervals], dtype=int)
    ends   = np.array([e for _, e in intervals], dtype=int)
    
    return {
        "scores": scores,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends
    }
