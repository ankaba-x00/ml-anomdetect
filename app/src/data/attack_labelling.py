from dataclasses import dataclass
import pandas as pd


@dataclass
class AttackThresholds:
    L3_HIGH: float
    L7_HIGH: float
    BITRATE_HIGH: float
    DURATION_MED: float
    AUTO_HUMAN_HIGH: float
    BOTS_HIGH: float


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


def _safe_q(
    series: pd.Series, 
    q: float, 
    default: float = 0.0
) -> float:
    s = series.dropna()
    return float(s.quantile(q)) if len(s) > 0 else default


def compute_attack_thresholds(
    df: pd.DataFrame
) -> AttackThresholds:
    """Computes thresholds dynamically for each data matrix."""
    return AttackThresholds(
        L3_HIGH=_safe_q(df["l3_origin"], 0.97),
        L7_HIGH=_safe_q(df["l7_traffic"], 0.97),
        BITRATE_HIGH=_safe_q(df["l3_bitrate_avg"], 0.95),
        DURATION_MED=_safe_q(df["l3_duration_avg"], 0.50),
        AUTO_HUMAN_HIGH=_safe_q(df["ratio_auto_human"], 0.90),
        BOTS_HIGH=_safe_q(df["bots_total"], 0.90),
    )


def derive_attack_label(
    row: pd.Series, 
    threshold: AttackThresholds
) -> int:
    """
    Generates DDoS classification taxonomy dynamically for each feature matrix.
    Returns
    -------
    0 : normal = no attack
    1 : udp_amplification = UDP reflection/amplification 
    2 : tcp_syn_flood = TCP SYN/ACK state-exhaustion
    3 : icmp_flood = ICMP ping flood
    4 : gre_flood = GRE tunnel flood
    5 : http_flood = L7 HTTP request flood
    6 : multi_vector = both L3 and L7 spike
    7 : stealth_scan = low-rate scan or probe
    """
    udp = float(row.get("udp_frac", 0.0))
    tcp = float(row.get("tcp_frac", 0.0))
    icmp = float(row.get("icmp_frac", 0.0))
    gre  = float(row.get("gre_frac", 0.0))

    l3 = float(row.get("l3_origin", 0.0))
    l7 = float(row.get("l7_traffic", 0.0))
    bitrate = float(row.get("l3_bitrate_avg", 0.0))
    duration = float(row.get("l3_duration_avg", 0.0))
    auto_ratio = float(row.get("ratio_auto_human", 0.0))
    bots = float(row.get("bots_total", 0.0))

    # 1. UDP Amplification
    if udp > 0.6 and l3 > threshold.L3_HIGH and bitrate > threshold.BITRATE_HIGH:
        return 1

    # 2. TCP SYN Flood
    if tcp > 0.6 and l3 > threshold.L3_HIGH and duration > threshold.DURATION_MED:
        return 2

    # 3. ICMP Flood
    if icmp > 0.3 and l3 > threshold.L3_HIGH:
        return 3

    # 4. GRE Flood
    if gre > 0.05 and l3 > threshold.L3_HIGH:
        return 4

    # 5. L7 HTTP Flood
    if (
        l7 > threshold.L7_HIGH
        and auto_ratio > threshold.AUTO_HUMAN_HIGH
        and l3 < 0.5 * threshold.L3_HIGH
    ):
        return 5

    # 6. Multi-vector
    if l7 > threshold.L7_HIGH and l3 > threshold.L3_HIGH:
        return 6

    # 7. Stealth scan
    if l3 > 0.3 * threshold.L3_HIGH and bots > threshold.BOTS_HIGH:
        return 7

    return 0  # normal
