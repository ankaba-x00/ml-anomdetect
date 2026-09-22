import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from typing import cast, Sequence

from . import EvaluationResult, MTPredictionResult
from .anomaly_utils import get_threshold, get_anomaly_mask, find_anomalies
from app.src.ml.models.mtae import MTTabularAE


def prediction(
    model: MTTabularAE,
    X_cont: npt.NDArray[np.float32],
    X_cat: npt.NDArray[np.int64],
    y_l3: npt.NDArray[np.float32],
    y_l7: npt.NDArray[np.float32],
    y_attack: npt.NDArray[np.int64],
    loss_weights: dict[str, float],
    attack_type_weights: torch.Tensor,
    pred_quantiles: Sequence[float] = [0.5],
    temperature: float = 1.0,
    device: str = "cpu",
    calibration: bool = False
) -> MTPredictionResult:
    """
    Computes per-sample reconstruction error normalized by features, 
    quantile-bound predictions of L3/l7 attack intensities and most probable 
    attack type.
    """
    model.eval()
    
    Xc = torch.from_numpy(X_cont.astype(np.float32)).to(device)
    Xk = torch.from_numpy(X_cat.astype(np.int64)).to(device)
    y3 = torch.from_numpy(y_l3.astype(np.float32)).to(device)
    y7 = torch.from_numpy(y_l7.astype(np.float32)).to(device)
    ya = torch.from_numpy(y_attack.astype(np.int64)).to(device)

    model.config.temperature = temperature

    with torch.no_grad():
        cont_recon, cat_logits, l3_pred, l7_pred, at_logits = model(Xc, Xk)
        
        _, _, recon_score, l3_loss, l7_loss, at_loss, loss_total = model.scoring(
            Xc, 
            Xk, 
            y3,
            y7,
            ya,
            cont_recon,
            cat_logits,
            l3_pred,
            l7_pred,
            at_logits,
            loss_weights,
            attack_type_weights
        )

        if calibration:
            return MTPredictionResult(scores=recon_score.detach().cpu().numpy().astype(np.float32))

        l3_score, l7_score = {}, {}
        for q in pred_quantiles:
            idx = model.config.quantiles.index(q)
            l3_score[f"{q}"] = l3_pred[:,idx].detach().cpu().numpy()
            l7_score[f"{q}"] = l7_pred[:,idx].detach().cpu().numpy()
        
        at_probs = F.softmax(at_logits, dim=-1)
        at_maxconf, at_maxprob = torch.max(at_probs, dim=-1)
        
        return MTPredictionResult(
            scores=recon_score.detach().cpu().numpy(),
            l3_pred=l3_score,
            l7_pred=l7_score,
            at_pred=at_maxprob.detach().cpu().numpy(),
            at_conf=torch.Tensor(at_maxconf * 100).to(torch.int).detach().cpu().numpy(),
            loss_total=loss_total.detach().cpu().numpy(),
            loss_l3=l3_loss.detach().cpu().numpy(),
            loss_l7=l7_loss.detach().cpu().numpy(),
            loss_at=at_loss.detach().cpu().numpy()
        )

def apply_mt_model(
    model: MTTabularAE,
    X_cont: npt.NDArray[np.float32],
    X_cat: npt.NDArray[np.int64],
    y_l3: npt.NDArray[np.float32],
    y_l7: npt.NDArray[np.float32],
    y_attack: npt.NDArray[np.int64],
    loss_weights: dict[str, float],
    attack_type_weights: torch.Tensor,
    pred_quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    method: str = "p99",
    min_length: int = 1,
    merge_gap: int = 0,
    device: str = "cpu"
) -> EvaluationResult:
    """
    Applies multi-task autoencoder and returns anomaly scoring incl. 
    reconstruction errors, threshold and anomaly intervals, as well as 
    prediction, losses and confidence scoring for L3/L7 intensities and attack
    types.
    """
    result = prediction(
        model=model,
        X_cont=X_cont,
        X_cat=X_cat,
        y_l3=y_l3,
        y_l7=y_l7,
        y_attack=y_attack,
        loss_weights=loss_weights,
        attack_type_weights=attack_type_weights,
        pred_quantiles=pred_quantiles,
        device=device
    )

    scores = result.scores

    threshold = get_threshold(method, scores)
    mask = get_anomaly_mask(scores, threshold)
    intervals = find_anomalies(
        mask,
        min_length,
        merge_gap,
    )

    starts = np.array([s for s, _ in intervals], dtype=np.int64)
    ends = np.array([e for _, e in intervals], dtype=np.int64)

    eval_results = EvaluationResult(
        scores=scores,
        threshold=threshold,
        mask=mask,
        anom_starts=starts,
        anom_ends=ends
    )
    for k, v in result.to_dict().items():
        if k == "scores":
            continue
        elif k in ["l3_pred", "l7_pred"]:
            for q in v.keys():
                values = cast(npt.NDArray[np.float32], v[q])
                eval_results[f"{k}_{q}"] = values
        else:
            eval_results[k] = v
    
    return eval_results
