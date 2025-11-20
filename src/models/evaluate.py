import numpy as np
import torch
from typing import Any
from src.models.autoencoder import TabularAutoencoder


#########################################
##  ANOMALY SCORE (RECON ERROR) UTILS  ##
#########################################

def reconstruction_error(
    model: TabularAutoencoder,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    device: str = None,
) -> np.ndarray:
    """Per-row reconstruction MSE."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    Xc = torch.from_numpy(X_cont.astype(np.float32)).to(device)
    Xk = torch.from_numpy(X_cat.astype(np.int64)).to(device)

    with torch.no_grad():
        recon_cont = model(Xc, Xk)
        mse = ((recon_cont - Xc) ** 2).mean(dim=1)

    return mse.cpu().numpy()



#########################################
##          THRESHOLD METHODS          ##
#########################################

def threshold_percentile(errors: np.ndarray, p: float = 99.0) -> float:
    """P-th percentile threshold."""
    return float(np.percentile(errors, p))


def threshold_mad(errors: np.ndarray, k: float = 6.0) -> float:
    """Median Absolute Deviation threshold. (typically k≈6–8 is good)"""
    med = np.median(errors)
    mad = np.median(np.abs(errors - med))
    return float(med + k * mad)


#########################################
##          ANOMALY DETECTION          ##
#########################################

def anomaly_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask."""
    return errors > threshold


def group_anomalies(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a boolean mask [F,F,T,T,T,F,F,T] into (start_indices, end_indices)."""
    mask = mask.astype(bool)
    if len(mask) == 0:
        return np.array([]), np.array([])

    # rising edges
    starts = np.where((~mask[:-1] & mask[1:]))[0] + 1
    # falling edges
    ends = np.where((mask[:-1] & ~mask[1:]))[0] + 1

    # handle anomaly at index 0
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    # handle anomaly ending at final sample
    if mask[-1]:
        ends = np.append(ends, len(mask))

    return starts, ends


#########################################
##      FULL EVALUATION PIPELINE       ##
#########################################

def evaluate_autoencoder(
    model: TabularAutoencoder,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    method: str = "p99",
    device: str = None,
) -> dict[str, Any]:
    """
    Full evaluation pipeline:
    - compute reconstruction errors
    - compute threshold
    - compute anomaly mask
    - compute anomaly intervals
    """

    # 1) errors
    errors = reconstruction_error(
        model,
        X_cont,
        X_cat,
        device=device,
    )
    # 2) threshold
    if method == "p99":
        threshold = threshold_percentile(errors, p=99)
    elif method == "p995":
        threshold = threshold_percentile(errors, p=99.5)
    elif method == "mad":
        threshold = threshold_mad(errors)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    # 3) mask
    mask = anomaly_mask(errors, threshold)
    # 4) group anomalies
    starts, ends = group_anomalies(mask)

    return {
        "errors": errors,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }