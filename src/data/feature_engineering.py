#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from src.data.io_utils import conv_pkltodf


########################################################
##                      PARAMS                        ##
########################################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
COUNTRIES = [
    "US","DE","GB","FR","JP","SG","NL","CA","AU","AT",
    "BR","CH","TW","IN","ZA","KR","SE","IT","ES","PL"
]


########################################################
##            LOADING + NORMALIZATION HELPERS         ##
########################################################

def _load_df(key: str) -> pd.DataFrame:
    """Load processed pkl → dataframe using conv_pkltodf()."""
    return conv_pkltodf(key, PROCESSED_DIR)


def _load_time_series(key: str, country: str, rename: str) -> pd.Series:
    """
    Load any *_time dataset (level 3 structure).
    Returns hourly series indexed by timestamp.
    """
    df = _load_df(key)

    if "regions" in df.columns:
        df = df[df["regions"] == country]

    df["ts"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("ts")

    s = df.set_index("ts")["values"].astype(float)
    s.name = rename
    return s

# TODO: only timeseries data for now use or delete later
def _load_nonperiod_csplit(key: str, country: str, rename: str) -> pd.Series:
    """
    For files like l3_origin / l3_target / l7_origin / l7_target:
    These are NOT timestamps but per-date aggregated values.
    Value for each date and assign that to the date.

    --> !! rrently dead code residue from playing with feature matrix, might use or delete later !!
    """
    df = _load_df(key)

    if "regions" not in df.columns:
        raise ValueError(f"{key}: unexpected structure (no region column).")

    df = df[(df["regions"] == country)]

    # "dates" as timestamp index for X indexing
    df["ts"] = pd.to_datetime(df["dates"])
    s = df.set_index("ts")["values"].astype(float)
    s.name = rename
    return s


########################################################
## BITRATE + DURATION AVG HELPERS (weighted averages) ##
########################################################

def weighted_avg(values: dict, mids: dict) -> float:
    """
    values: dict like {"UNDER_500_MBPS": 94.6, "_500_MBPS_TO_1_GBPS": 3.2, ...}
    mids:   dict with the bucket midpoints for the same keys.
    Returns a single weighted average.
    """
    cleaned = {
        k: float(v)
        for k, v in values.items()
        if k in mids and pd.notna(v)
    }

    if not cleaned:
        return 0.0

    arr_v = np.array(list(cleaned.values()), dtype=float)
    arr_m = np.array([mids[k] for k in cleaned.keys()], dtype=float)

    denom = arr_v.sum()
    if denom <= 0:
        return 0.0

    return float((arr_v * arr_m).sum() / denom)


def _load_weighted_dist(key: str, country: str, rename: str, mids: dict) -> pd.Series:
    """
    Convert multi-bucket distributions (bitrate, duration) into a series
    of weighted averages.
    """
    df = _load_df(key)

    df = df[df["regions"] == country]
    df["ts"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("ts")

    # pivot such that columns = bucket categories
    df_p = df.pivot_table(index="ts", columns="metric", values="values", aggfunc="first")

    out = []
    for ts, row in df_p.iterrows():
        vals = {col: row[col] for col in df_p.columns if not pd.isna(row[col])}
        out.append((ts, weighted_avg(vals, mids)))

    s = pd.Series({ts: v for ts, v in out})
    s.name = rename
    return s


########################################################
##           PROTOCOL FRACTIONS + ENTROPY             ##
########################################################

def shannon_entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def load_protocol_features(key: str, country: str):
    df = _load_df(key)
    df = df[df["regions"] == country]

    df["ts"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("ts")

    df_p = df.pivot_table(index="ts", columns="metric", values="values", aggfunc="first")

    df_p = df_p.rename(columns={
        "UDP": "udp",
        "TCP": "tcp",
        "ICMP": "icmp",
        "GRE": "gre"
    })

    pcols = ["udp", "tcp", "icmp", "gre"]
    df_p["total"] = df_p[pcols].sum(axis=1).replace(0, 1e-6)

    for c in pcols:
        df_p[f"{c}_frac"] = df_p[c] / df_p["total"]

    df_p["protocol_entropy"] = df_p.apply(
        lambda r: shannon_entropy([r["udp_frac"], r["tcp_frac"], r["icmp_frac"], r["gre_frac"]]),
        axis=1
    )

    keep = ["udp_frac", "tcp_frac", "icmp_frac", "gre_frac", "protocol_entropy"]
    return df_p[keep]


########################################################
##       MAIN COUNTRY MERGED DATAFRAME BUILDER        ##
########################################################

def build_country_dataframe(country: str) -> pd.DataFrame:
    print(f"Building base merged DF for country={country}")

    # ============================
    # 1. load all base series
    # ============================

    s_l3o = _load_time_series("l3_origin_time", country, "l3_origin")
    s_l3t = _load_time_series("l3_target_time", country, "l3_target")
    s_l7 = _load_time_series("l7_time", country, "l7_traffic")

    s_http = _load_time_series("httpreq_time", country, "http")
    s_http_auto = _load_time_series("httpreq_automated_time", country, "http_auto")
    s_http_human = _load_time_series("httpreq_human_time", country, "http_human")

    s_netflow = _load_time_series("traffic_time", country, "netflow")

    s_bots = _load_time_series("bots_time", country, "bots_total")
    s_ai = _load_time_series("aibots_crawlers_time", country, "ai_bots")

    # weighted bitrate avg
    bitrate_mids = {
        "UNDER_500_MBPS": 250,
        "_500_MBPS_TO_1_GBPS": 750,
        "_1_GBPS_TO_10_GBPS": 5500,
        "_10_GBPS_TO_100_GBPS": 55000,
        "OVER_100_GBPS": 100000
    }
    s_l3_bitrate = _load_weighted_dist("l3_origin_bitrate_time", country, "l3_bitrate_avg", bitrate_mids)

    # weighted duration avg
    dur_mids = {
        "UNDER_10_MINS": 5,
        "_10_MINS_TO_20_MINS": 15,
        "_20_MINS_TO_40_MINS": 30,
        "_40_MINS_TO_1_HOUR": 50,
        "_1_HOUR_TO_3_HOURS": 120,
        "OVER_3_HOURS": 300
    }
    s_l3_duration = _load_weighted_dist("l3_origin_duration_time", country, "l3_duration_avg", dur_mids)

    # protocol + entropy
    df_protocol = load_protocol_features("l3_origin_protocol_time", country)

    # ============================
    # 2. merge everything
    # ============================

    df = pd.concat(
        [
            s_l3o, s_l3t, s_l7,
            s_http, s_http_auto, s_http_human,
            s_netflow,
            s_bots, s_ai,
            s_l3_bitrate, s_l3_duration,
        ],
        axis=1
    )

    df = df.join(df_protocol, how="outer")
    df = df.sort_index()
    df = df.interpolate().ffill().bfill()

    # ============================
    # 3. derived Ratios
    # ============================

    df["ratio_l3_l7"] = df["l3_origin"] / df["l7_traffic"].replace(0, 1e-6)
    df["ratio_auto_human"] = df["http_auto"] / df["http_human"].replace(0, 1e-6)
    df["ratio_bots_http"] = df["bots_total"] / df["http"].replace(0, 1e-6)
    df["ratio_ai_bots_bots"] = df["ai_bots"] / df["bots_total"].replace(0, 1e-6)
    df["ratio_netflow_http"] = df["netflow"] / df["http"].replace(0, 1e-6)

    # ============================
    # 4. time cyclic encoding
    # ============================

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df = df.drop(columns=["hour", "dow"])

    # ============================
    # 5. rolling aggregates
    # ============================

    roll_cols = ["l3_origin", "l3_target", "l7_traffic",
                 "http", "http_auto", "http_human",
                 "netflow", "bots_total", "ai_bots"]

    for col in roll_cols:
        df[f"{col}_roll3h"] = df[col].rolling("3h").mean()
        df[f"{col}_roll24h"] = df[col].rolling("24h").mean()

    df = df.bfill()

    return df


########################################################
##             SCALING + FINAL X MATRIX               ##
########################################################

def build_feature_matrix(country: str):
    df = build_country_dataframe(country).copy()

    scaler = RobustScaler()
    X = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    return X, scaler
