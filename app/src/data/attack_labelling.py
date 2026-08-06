from dataclasses import dataclass
from typing import Union
import pandas as pd
import numpy as np

# TODO: test on the Canadian netflow dataset for 4,6 especially

@dataclass
class AttackThresholds:
    L3_HIGH: float
    L7_HIGH: float
    BITRATE_HIGH: float
    DURATION_MED: float
    AUTO_HUMAN_HIGH: float
    BOTS_HIGH: float
    UDP_DOMINANT: float
    TCP_DOMINANT: float
    ICMP_DOMINANT: float
    GRE_DOMINANT: float


ATTACK_LABELS = [
    "normal",
    "udp_amplification",
    "tcp_syn_flood",
    "icmp_flood",
    "gre_flood",
    "http_flood",
    "multi_vector",
    "stealth_scan",
]

ATTACK_TO_ID = {name: idx for idx, name in enumerate(ATTACK_LABELS)}
ID_TO_ATTACK = {idx: name for idx, name in enumerate(ATTACK_LABELS)}


def _safe_q(series: pd.Series, q: float, default: float = 0.0) -> float:
    """Ensures safe quantile calculation."""
    s = series.dropna()
    return float(s.quantile(q)) if len(s) > 0 else default


def compute_attack_thresholds(df: pd.DataFrame) -> AttackThresholds:
    """Computes thresholds dynamically for each data matrix."""
    return AttackThresholds(
        L3_HIGH=_safe_q(df["l3_origin"], 0.97),
        L7_HIGH=_safe_q(df["l7_traffic"], 0.97),
        BITRATE_HIGH=_safe_q(df["l3_bitrate_avg"], 0.95),
        DURATION_MED=_safe_q(df["l3_duration_avg"], 0.50),
        AUTO_HUMAN_HIGH=_safe_q(df["ratio_auto_human"], 0.90),
        BOTS_HIGH=_safe_q(df["bots_total"], 0.90),
        UDP_DOMINANT=_safe_q(df.get("udp_frac", pd.Series([0.0])), 0.95),
        TCP_DOMINANT=_safe_q(df.get("tcp_frac", pd.Series([0.0])), 0.95),
        ICMP_DOMINANT=_safe_q(df.get("icmp_frac", pd.Series([0.0])), 0.95),
        GRE_DOMINANT=min(_safe_q(df.get("gre_frac", pd.Series([0.0])), 0.95), 0.1)
    )


def score_attack_types(
    row: pd.Series,
    thresholds: AttackThresholds,
    return_scores: bool = False
) -> Union[int, dict[str, float]]:
    """
    Stage A : Attack label derivation with scoring system.
    Returns
    -------
    if return_scores = False : single label index (0-7)
    if returns_scores = True : scores dict
    """
    # extract features
    udp = float(row.get("udp_frac", 0.0))
    tcp = float(row.get("tcp_frac", 0.0))
    icmp = float(row.get("icmp_frac", 0.0))
    gre = float(row.get("gre_frac", 0.0))
    protocol_entropy = float(row.get("protocol_entropy", 1.0))
    l3 = float(row.get("l3_origin", 0.0))
    l7 = float(row.get("l7_traffic", 0.0))
    bitrate = float(row.get("l3_bitrate_avg", 0.0))
    duration = float(row.get("l3_duration_avg", 0.0))
    auto_ratio = float(row.get("ratio_auto_human", 0.0))
    bots = float(row.get("bots_total", 0.0))
    
    # calculate normalized scores (0-1 range)
    def norm(x, ref):
        return min(x / ref, 2.0) / 2.0 if ref > 0 else 0.0

    l3_norm = norm(l3, thresholds.L3_HIGH)
    l7_norm = norm(l7, thresholds.L7_HIGH)
    bitrate_norm = norm(bitrate, thresholds.BITRATE_HIGH)
    duration_norm = norm(duration, thresholds.DURATION_MED)
    auto_norm = norm(auto_ratio, thresholds.AUTO_HUMAN_HIGH)
    bots_norm = norm(bots, thresholds.BOTS_HIGH)
    
    # use dynamic protocol thresholds if available
    def eval_thres(thres, val):
        return thres if thres > 0 else val
    
    udp_threshold = eval_thres(thresholds.UDP_DOMINANT, 0.6)
    tcp_threshold = eval_thres(thresholds.TCP_DOMINANT, 0.6)
    icmp_threshold = eval_thres(thresholds.ICMP_DOMINANT, 0.3)
    gre_threshold = eval_thres(thresholds.GRE_DOMINANT, 0.05)
    
    # scores for each attack type
    scores = {label: 0.0 for label in ATTACK_LABELS}
    
    # ------------------------
    # 1 = UDP amplification
    # ------------------------
    entropy_factor = max(0.0, 1.0 - protocol_entropy)
    udp_score = (
        udp * 0.35 + 
        l3_norm * 0.35 + 
        bitrate_norm * 0.2 + 
        entropy_factor * 0.2
    ) if udp > udp_threshold * 0.8 and l3_norm > 0.5 else 0.0
    scores["udp_amplification"] = udp_score
    
    # ------------------------
    # 2 = TCP SYN flood
    # ------------------------
    tcp_score = (
        tcp * 0.4 + 
        l3_norm * 0.3 + 
        duration_norm * 0.3
    ) if tcp > tcp_threshold * 0.8 else 0.0
    scores["tcp_syn_flood"] = tcp_score
    
    # ------------------------
    # 3 = ICMP flood
    # ------------------------
    icmp_score = (
        0.5 * min(icmp / max(icmp_threshold, 0.1), 1.0) + 
        0.5 * l3_norm
    ) if icmp > icmp_threshold * 0.7 else 0.0
    scores["icmp_flood"] = icmp_score
    
    # ------------------------
    # 4 = GRE flood
    # ------------------------
    # TODO: test downgraded gre_threshold
    gre_ratio_norm = min(gre / max(gre_threshold, 0.02), 1.0)
    gre_score = (
        gre_ratio_norm * 0.6 + 
        l3_norm * 0.4
    ) if gre > gre_threshold * 0.7 else 0.0
    scores["gre_flood"] = gre_score
    
    # ------------------------
    # 5 = HTTP flood
    # ------------------------
    # penalizes high L3 for HTTP floods (should be L7 focused) TODO: too much?
    l3_penalty = max(0, 1 - l3_norm * 2)
    http_score = (
        l7_norm * 0.4 + 
        auto_norm * 0.3 + 
        l3_penalty * 0.3
    ) if l7_norm > 0.5 else 0.0
    scores["http_flood"] = http_score
    
    # ------------------------
    # 6 = multi-vector
    # ------------------------
    #multi_score = (l3_norm * 0.5 + l7_norm * 0.5) if l3_norm > 0.5 and l7_norm > 0.5 else 0
    #multi_score = min(l3_norm * l7_norm * 1.5, 1.0)
    #multi_score *= 1.2  # slight bias to help emergence?!?!
    #multi_score = min(multi_score, 1.0)
    multi_score = (
        0.6 * max(l3_norm, l7_norm) + 
        0.4 * min(l3_norm, l7_norm)
    ) if l3_norm >= 0.4 or l7_norm >= 0.4 else 0.0
    # suppresses single-vector masquerading as multi
    #if min(l3_norm, l7_norm) < 0.3:
    #    multi_score *= 0.5
    scores["multi_vector"] = multi_score

    # ------------------------
    # 7 = stealth scan
    # ------------------------
    # TODO: too prominent
    #stealth_score = (l3_norm * 0.4 * 0.5 + bots_norm * 0.6) if 0.2 < l3_norm < 0.7 and bots_norm > 0.3 else 0
    l3_band = 1.0 - abs(l3_norm - 0.4) * 2  # peak around 0.4
    l3_band = max(l3_band, 0.0)
    #stealth_score = (0.4 * l3_band + 0.4 * bots_norm + 0.2 * protocol_entropy)
    stealth_score = (
        l3_band * 0.3 + 
        bots_norm * 0.5 + 
        protocol_entropy *0.2
    )
    if max(
        scores["udp_amplification"],
        scores["tcp_syn_flood"],
        scores["http_flood"],
    ) > 0.3:
        stealth_score *= 0.2
    if l7_norm > 0.35 and bitrate_norm > 0.5:
        stealth_score = 0.0
    if l3_norm < 0.25 and l7_norm < 0.25:
        stealth_score = 0.0
    scores["stealth_scan"] = stealth_score

    # ------------------------
    # 0 = normal = no attack
    # ------------------------
    max_attack = max(scores.values())
    scores["normal"] = max(0.0, 1 - 1.2 * max_attack)
    
    # find best label and apply minimum confidence threshold (except for normal)
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    
    if best_label != "normal" and best_score < 0.3:
        best_label = "normal"
    
    if return_scores:
        return scores
    return ATTACK_TO_ID[best_label]


def temporal_attack_labeling(
    semantic_labels: pd.Series,
    semantic_scores: pd.DataFrame,
    min_event_score: float = 0.45,
) -> pd.Series:
    """
    Stage B : temporal validation of sematic attack labels; never re-classifies attack type.
    Returns
    -------
    semantic labels as pd.Series 
    """

    final = semantic_labels.copy()

    # minimum duration per attack type (in hours)
    min_duration = {
        "udp_amplification": 1,
        "tcp_syn_flood": 1,
        "icmp_flood": 1,
        "gre_flood": 1,
        "http_flood": 2,
        "multi_vector": 1,
        "stealth_scan": 4,
    }

    # 1. score-based confidence veto 
    for attack, idx in ATTACK_TO_ID.items():
        if attack == "normal":
            continue

        low_conf = (
            (semantic_labels == idx) &
            (semantic_scores[attack] < min_event_score)
        )
        final.loc[low_conf] = ATTACK_TO_ID["normal"]

    # 2. duration enforcement (label-based only)
    for attack, min_len in min_duration.items():
        idx = ATTACK_TO_ID[attack]
        mask = final == idx

        groups = (mask != mask.shift()).cumsum()
        sizes = mask.groupby(groups).transform("size")

        final.loc[(mask) & (sizes < min_len)] = ATTACK_TO_ID["normal"]

    

    # 3. multi-vector promotion if window is classified TCP or HTTP and window shows strong activity in other layer
    for i in range(1, len(final) - 1):
        if final.iloc[i] in (
            ATTACK_TO_ID["tcp_syn_flood"],
            ATTACK_TO_ID["http_flood"],
        ):
            prev_scores = semantic_scores.iloc[i - 1]
            curr_scores = semantic_scores.iloc[i]
            next_scores = semantic_scores.iloc[i + 1]

            if (
                curr_scores["multi_vector"] > min_event_score * 0.8
                and (
                    prev_scores["http_flood"] > min_event_score
                    or next_scores["tcp_syn_flood"] > min_event_score
                )
            ):
                final.iloc[i] = ATTACK_TO_ID["multi_vector"]
    
    # 4. isolated multi-vector cap to enforce category as transitional/overlap while HTTP/TCP are sustained phases 
    # TODO: test whether this makes sense or not
    # TODO: question also to what it decays, TCP or HTTP or write semantic rule?
    # mv = final == ATTACK_TO_ID["multi_vector"]
    # groups = (mv != mv.shift()).cumsum()
    # sizes = mv.groupby(groups).transform("size")

    # final.loc[(mv) & (sizes > 2)] = ATTACK_TO_ID["http_flood"]
    # final.loc[(mv) & (sizes > 2)] = ATTACK_TO_ID["tcp_syn_flood"]

    return final
