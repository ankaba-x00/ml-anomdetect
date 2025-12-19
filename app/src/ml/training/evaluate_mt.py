import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict
from app.src.ml.models.mte import TrafficAttackPredictor


#########################################
##      MULTI-TASK SCORING UTILS       ##
#########################################

def prediction_errors(
    model: TrafficAttackPredictor,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    y_l3: Optional[np.ndarray] = None,
    y_l7: Optional[np.ndarray] = None,
    y_attack: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    l3_weight: float = 1.0,
    l7_weight: float = 1.0,
    attack_weight: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute per-sample prediction errors."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    Xc = torch.from_numpy(X_cont.astype(np.float32)).to(device)
    Xk = torch.from_numpy(X_cat.astype(np.int64)).to(device)

    # labels optional
    y3 = torch.from_numpy(y_l3.astype(np.float32)).to(device) if y_l3 is not None else None
    y7 = torch.from_numpy(y_l7.astype(np.float32)).to(device) if y_l7 is not None else None
    ya = torch.from_numpy(y_attack.astype(np.int64)).to(device) if y_attack is not None else None

    with torch.no_grad():
        out = model(Xc, Xk)
        l3_hat = out["l3"]
        l7_hat = out["l7"]
        logits = out["attack_logits"]

        # per-sample regression losses (MSE)
        loss_l3 = ((l3_hat - y3) ** 2) if y3 is not None else None
        loss_l7 = ((l7_hat - y7) ** 2) if y7 is not None else None

        # per-sample classification loss (CE)
        loss_attack = None
        attack_pred = None
        attack_prob_max = None
        if ya is not None:
            loss_attack = F.cross_entropy(logits, ya, reduction="none")
            probs = F.softmax(logits, dim=-1)
            attack_pred = probs.argmax(dim=-1)
            attack_prob_max = probs.max(dim=-1).values

        # total per-sample combined loss of regression
        total = torch.zeros(Xc.size(0), device=device)
        if loss_l3 is not None:
            total = total + l3_weight * loss_l3
        if loss_l7 is not None:
            total = total + l7_weight * loss_l7

    return {
        "loss_total": total.detach().cpu().numpy(),
        "loss_l3": loss_l3.detach().cpu().numpy() if loss_l3 is not None else None,
        "loss_l7": loss_l7.detach().cpu().numpy() if loss_l7 is not None else None,
        "loss_attack": loss_attack.detach().cpu().numpy() if loss_attack is not None else None,
        "l3_pred": l3_hat.detach().cpu().numpy(),
        "l7_pred": l7_hat.detach().cpu().numpy(),
        "attack_pred": attack_pred.detach().cpu().numpy() if attack_pred is not None else None,
        "attack_prob_max": attack_prob_max.detach().cpu().numpy() if attack_prob_max is not None else None,
    }


#########################################
##          THRESHOLD METHODS          ##
#########################################

def threshold_percentile(errors: np.ndarray, p: float = 99.0) -> float:
    return float(np.percentile(errors, p))


def threshold_mad(errors: np.ndarray, k: float = 6.0, min_p: float = 99.5, max_p: float = 99.9) -> float:
    med = np.median(errors)
    mad = np.median(np.abs(errors - med)) + 1e-12
    nmad = 1.4826 * mad
    thr = med + k * nmad
    low = np.percentile(errors, min_p)
    high = np.percentile(errors, max_p)
    thr = np.clip(thr, low, high)
    return float(thr)


#########################################
##          ANOMALY DETECTION          ##
#########################################

def anomaly_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    return errors > threshold


def find_anomalies(mask: np.ndarray, min_length: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    mask = mask.astype(bool)
    N = len(mask)
    if N == 0:
        return []

    intervals = []
    in_anom = False
    start = None

    for i, is_anom in enumerate(mask):
        if is_anom and not in_anom:
            in_anom = True
            start = i
        elif not is_anom and in_anom:
            intervals.append((start, i))
            in_anom = False

    if in_anom:
        intervals.append((start, N))

    if min_length > 1:
        intervals = [(s, e) for (s, e) in intervals if (e - s) >= min_length]

    if merge_gap > 0 and len(intervals) > 1:
        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s - cur_e <= merge_gap:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        intervals = merged

    return intervals


#########################################
##      FULL EVALUATION PIPELINE       ##
#########################################

def apply_multitask_model(
    model: TrafficAttackPredictor,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    y_l3: np.ndarray,
    y_l7: np.ndarray,
    y_attack: np.ndarray,
    method: str = "p99",
    device: Optional[str] = None,
    l3_weight: float = 1.0,
    l7_weight: float = 1.0,
    attack_weight: float = 1.0,
    min_length: int = 1,
    merge_gap: int = 0,
) -> Dict[str, Any]:
    """Computes per-sample combined loss and (optional) anomaly intervals based on a threshold."""
    out = prediction_errors(
        model=model,
        X_cont=X_cont,
        X_cat=X_cat,
        y_l3=y_l3,
        y_l7=y_l7,
        y_attack=y_attack,
        device=device,
        l3_weight=l3_weight,
        l7_weight=l7_weight,
        attack_weight=attack_weight,
    )

    errors = out["loss_total"]

    if method == "p99":
        threshold = threshold_percentile(errors, p=99)
    elif method == "p995":
        threshold = threshold_percentile(errors, p=99.5)
    elif method == "mad":
        threshold = threshold_mad(errors)
    else:
        raise ValueError(f"[Error] Unknown threshold method: {method}")

    mask = anomaly_mask(errors, threshold)
    intervals = find_anomalies(mask, min_length=min_length, merge_gap=merge_gap)

    starts = np.array([s for s, _ in intervals], dtype=int)
    ends = np.array([e for _, e in intervals], dtype=int)

    return {
        **out,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }
