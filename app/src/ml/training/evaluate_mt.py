import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union

from app.src.ml.models.mte import TrafficAttackPredictor
from .core.anomaly_utils import get_threshold, get_anomaly_mask, find_anomalies

#########################################
##      MULTI-TASK SCORING UTILS       ##
#########################################

def pinball_loss_vec(preds, y, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        e = y - preds[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.stack(losses, dim=1).mean(dim=1)


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
    score_quantile: float = 0.99
) -> dict[str, Union[np.ndarray, None]]:
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
        
        quantiles = model.config.quantiles
        if score_quantile not in quantiles:
            raise ValueError(f"[ERROR] Model quantiles does not include score_quantile={score_quantile}.")
        q_idx = quantiles.index(score_quantile)
        l3_q = out["l3_q"]
        l7_q = out["l7_q"]
        l3_p = l3_q[:, q_idx]
        l7_p = l7_q[:, q_idx]

        # anomaly score as weighted tail risk
        score = l3_weight * l3_p + l7_weight * l7_p

        # optional diagnostic losses (not for alerting)
        loss_l3 = pinball_loss_vec(l3_q, y3, quantiles) if y3 is not None else None
        loss_l7 = pinball_loss_vec(l7_q, y7, quantiles) if y7 is not None else None

        # per-sample classification loss (CE)
        loss_attack = None
        attack_pred = None
        attack_prob_max = None

        if ya is not None:
            logits = out["attack_logits"]
            loss_attack = F.cross_entropy(
                logits, 
                ya, 
                reduction="none", 
                weight=model.attack_class_weights if getattr(model, "attack_class_weights", None) is not None
                else None,
            )
            probs = F.softmax(logits, dim=-1)
            attack_pred = probs.argmax(dim=-1)
            attack_prob_max = probs.max(dim=-1).values
        
        total = (
            model.config.lambda_l3 * loss_l3 +
            model.config.lambda_l7 * loss_l7 +
            model.config.lambda_attack * loss_attack
        )

    return {
        "score_quantile": int(score_quantile*100),
        "score": score.detach().cpu().numpy().astype(np.float32),
        f"l3_p{int(score_quantile*100)}": l3_p.cpu().numpy(),
        f"l7_p{int(score_quantile*100)}": l7_p.cpu().numpy(),
        "loss_l3": loss_l3.cpu().numpy() if loss_l3 is not None else None,
        "loss_l7": loss_l7.cpu().numpy() if loss_l7 is not None else None,
        "loss_attack": loss_attack.detach().cpu().numpy() if loss_attack is not None else None,
        "loss_total": total.detach().cpu().numpy().astype(np.float32),
        "attack_pred": attack_pred.detach().cpu().numpy() if attack_pred is not None else None,
        "attack_prob_max": attack_prob_max.detach().cpu().numpy() if attack_prob_max is not None else None,
    }


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
    min_length: int = 1,
    merge_gap: int = 0,
    score_quantile: float = 0.99
) -> dict[str, Union[np.ndarray, None]]:
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
        score_quantile=score_quantile
    )

    scores = out["score"]

    threshold = get_threshold(method, scores)
    mask = get_anomaly_mask(scores, threshold)
    intervals = find_anomalies(
        mask,
        min_length=min_length,
        merge_gap=merge_gap,
    )

    starts = np.array([s for s, _ in intervals], dtype=int)
    ends = np.array([e for _, e in intervals], dtype=int)

    return {
        **out,
        "threshold": threshold,
        "mask": mask,
        "anomaly_starts": starts,
        "anomaly_ends": ends,
    }
