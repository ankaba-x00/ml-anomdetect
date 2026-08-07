import numpy as np
import torch

from app.src.ml.models.ae import TabularAE
from app.src.ml.models.vae import TabularVAE
from .core.anomaly_utils import get_threshold, get_anomaly_mask, find_anomalies

##############################
##       SCORING UTILS      ##
##############################

def reconstruction(
    model: TabularAE | TabularVAE,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    device: str | None = None,
    cont_w: float = 1.0,
    cat_w: float = 0.0,
    use_mc_elbo: bool = False,
    temperature: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Per-sample reconstruction error normalized by features."""

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    Xc = torch.from_numpy(X_cont.astype(np.float32)).to(device)
    Xk = torch.from_numpy(X_cat.astype(np.int64)).to(device)

    with torch.no_grad():
        if use_mc_elbo and isinstance(model, TabularVAE):
            scores = model.mc_elbo_score(
                Xc, 
                Xk, 
                cont_w, 
                cat_w, 
                temperature, 
                n_samples=20, 
                beta=beta
            )
        else:
            cont_recon, cat_logits = model(Xc, Xk)
            scores = model.scoring(
                Xc, 
                Xk,
                cont_recon,
                cat_logits,
                {"cont_w": cont_w, "cat_w": cat_w}, 
                reduction="none"
            )

    return scores.cpu().numpy()

#########################################
##      FULL EVALUATION PIPELINE       ##
#########################################

def apply_model(
    model: TabularAE | TabularVAE,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    method: str = "p99",
    device: str | None = None,
    cont_w: float = 1.0,
    cat_w: float = 0.0,
    temperature: float = 1.0,
    use_mc_elbo: bool = False,
    beta: float = 1.0,
    min_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, np.ndarray]:
    """Compute reconstruction errors, threshold, anomaly mask, anomaly intervals."""

    scores = reconstruction(
        model,
        X_cont,
        X_cat,
        device,
        cont_w,
        cat_w,
        use_mc_elbo,
        temperature,
        beta
    )
    
    threshold = get_threshold(method, scores)
    mask = get_anomaly_mask(scores, threshold)
    intervals = find_anomalies(
        mask,
        min_length=min_length,
        merge_gap=merge_gap,
    )
    starts = np.array([s for s, _ in intervals])
    ends   = np.array([e for _, e in intervals])
    
    return {
        "scores": scores,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }
