import numpy as np
import numpy.typing as npt
import torch
from typing import cast

from . import EvaluationResult
from .anomaly_utils import get_threshold, get_anomaly_mask, find_anomalies
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE


def reconstruction(
    model: TabularAE | TabularVAE,
    X_cont: npt.NDArray[np.float32],
    X_cat: npt.NDArray[np.int64],
    loss_weights: dict[str, float],
    temperature: float = 1.0,
    device: str = "cpu"
) -> npt.NDArray[np.float32]:
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

    return cast(torch.Tensor, scores).detach().cpu().numpy()

def apply_model(
    model: TabularAE | TabularVAE,
    X_cont: npt.NDArray[np.float32],
    X_cat: npt.NDArray[np.int64],
    loss_weights: dict[str, float],
    device: str = "cpu",
    method: str = "p99",
    temperature: float = 1.0,
    min_length: int = 1,
    merge_gap: int = 0,
    threshold: float | None = None
) -> EvaluationResult:
    """Applies autoencoder and returns reconstruction errors, threshold, anomaly mask and anomaly intervals."""

    scores = reconstruction(
        model,
        X_cont,
        X_cat,
        loss_weights,
        temperature,
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
    starts = np.array([s for s, _ in intervals], dtype=np.int64)
    ends   = np.array([e for _, e in intervals], dtype=np.int64)

    eval_results = EvaluationResult(
        scores=scores,
        threshold=threshold,
        mask=mask,
        anom_starts=starts,
        anom_ends=ends
    )

    return eval_results
