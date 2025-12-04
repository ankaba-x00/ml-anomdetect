import numpy as np
import torch
from typing import Any, Optional
from app.src.models.autoencoder import TabularAutoencoder


#########################################
##  ANOMALY SCORE (RECON ERROR) UTILS  ##
#########################################

def reconstruction_error(
    model: TabularAutoencoder,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    device: Optional[str] = None,
    cont_weight: float = 1.0,
    cat_weight: float = 0.0,
    temperature: float = 1.0,
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
        errors = model.anomaly_score(Xc, Xk, cont_weight, cat_weight, temperature)

    return errors.cpu().numpy()


#########################################
##          THRESHOLD METHODS          ##
#########################################

def threshold_percentile(errors: np.ndarray, p: float = 99.0) -> float:
    """Computes p-th percentile threshold."""
    return float(np.percentile(errors, p))


# TODO: test how k = 3.5, 5, 6 differ!
# TODO: clip threshold to 99.9 (or 99.5) percentile to avoid ridiculous MAD inflation if necessary
def threshold_mad(errors: np.ndarray, k: float = 6.0) -> float:
    """
    Computes median Absolute Deviation threshold. Scaling factor 
        k = 3-3.5 : used for mododerately heavy-tailed dist
        k = 6-8 : used for rare anomaly detection
        k = >10 : conservative (almost nothing flagged)
    """
    med = np.median(errors)
    mad = np.median(np.abs(errors - med)) + 1e-12
    thr = med + k * mad
    thr = min(thr, np.percentile(errors, 99.9)) # to avoid extreme outliers
    return float(thr)


#########################################
##          ANOMALY DETECTION          ##
#########################################

def anomaly_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Creates boolean mask."""
    return errors > threshold


def find_anomalies(mask: np.ndarray, 
    min_length: int = 1,
    merge_gap: int = 0,
) -> list[tuple[int, int]]:
    """
    Converts bool mask [F,F,T,T,T,F,F,T] into list of anomalous sample intervals [(start, end), ...]; end is exclusive. 
    min_length : min anomaly sample length to be considered
    merge_gap : sample interval gap to merge 2 adjacent anomalies into one
    """
    mask = mask.astype(bool)
    N = len(mask)
    if N == 0:
        return []

    intervals = []
    in_anom = False
    start = None

    # identify raw intervals
    for i, is_anom in enumerate(mask):
        if is_anom and not in_anom:
            in_anom = True
            start = i
        elif not is_anom and in_anom:
            intervals.append((start, i))
            in_anom = False

    if in_anom:
        intervals.append((start, N))

    # filter by minimum length
    if min_length > 1:
        intervals = [
            (s, e) for (s, e) in intervals if (e - s) >= min_length
        ]

    # merge intervals close to each other (gap < merge_gap)
    if merge_gap > 0 and len(intervals) > 1:
        merged = []
        cur_s, cur_e = intervals[0]

        for s, e in intervals[1:]:
            if s - cur_e <= merge_gap:
                cur_e = e  # extend
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e

        merged.append((cur_s, cur_e))
        intervals = merged

    return intervals


#########################################
##      FULL EVALUATION PIPELINE       ##
#########################################

def apply_model(
    model: TabularAutoencoder,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    method: str = "p99",
    device: Optional[str] = None,
    min_length: int = 1,
    merge_gap: int = 0,
    cont_weight: float = 1.0,
    cat_weight: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Applies model on data and compute reconstruction errors, threshold, anomaly mask, anomaly intervals."""
    errors = reconstruction_error(
        model,
        X_cont,
        X_cat,
        device,
        cont_weight,
        cat_weight,
        temperature
    )
    if method == "p99":
        threshold = threshold_percentile(errors, p=99)
    elif method == "p995":
        threshold = threshold_percentile(errors, p=99.5)
    elif method == "mad":
        threshold = threshold_mad(errors)
    else:
        raise ValueError(f"[Error] Unknown threshold method: {method}")
    mask = anomaly_mask(errors, threshold)
    intervals = find_anomalies(
        mask,
        min_length=min_length,
        merge_gap=merge_gap,
    )    
    starts = np.array([s for s, _ in intervals])
    ends   = np.array([e for _, e in intervals])
    return {
        "errors": errors,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }