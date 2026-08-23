import numpy as np


def _threshold_percentile(
    scores: np.ndarray, 
    p: float = 99.
) -> float:
    """Computes p-th percentile threshold."""
    return float(np.percentile(scores, p))


def _threshold_mad(
    scores: np.ndarray, 
    k: float = 6.0, 
    min_p: float = 99.5, 
    max_p: float = 99.9
) -> float:
    """
    Computes normalized median absolute deviation threshold with enforced minimum threshold based on percentile. 
    Scaling factor 
        k = 3-3.5 : used for mododerately heavy-tailed dist
        k = 6-8 : used for rare anomaly detection
        k = >10 : conservative (almost nothing flagged)
    """
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-12
    nmad = 1.4826 * mad
    thr = med + k * nmad
    low = np.percentile(scores, min_p)
    high = np.percentile(scores, max_p)
    thr = np.clip(thr, low, high)

    return float(thr)

def get_threshold(method: str, scores: np.ndarray):
    if method == "p95":
        return _threshold_percentile(scores, p=95.)
    elif method == "p99":
        return _threshold_percentile(scores, p=99.)
    elif method == "p995":
        return _threshold_percentile(scores, p=99.5)
    elif method == "mad":
        return _threshold_mad(scores)
    else:
        raise ValueError(f"[Error] Unknown threshold method: {method}")

def get_anomaly_mask(
    scores: np.ndarray, 
    threshold: float
) -> np.ndarray:
    """Creates boolean mask for anomaly flagging."""
    return scores > threshold

def find_anomalies(
    mask: np.ndarray, 
    min_length: int = 1,
    merge_gap: int = 0,
) -> list[tuple[int, int]]:
    """
    Converts bool mask into list of anomalous sample intervals of format
    [(start, end), ...]; end is exclusive. 

    Args:
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
                cur_e = e # extend
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e

        merged.append((cur_s, cur_e))
        intervals = merged

    return intervals