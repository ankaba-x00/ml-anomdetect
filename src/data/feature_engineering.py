#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from sklearn.preprocessing import RobustScaler
from src.data.io_utils import conv_pkltodf
from src.exploration.core.time_utils import conv_iso_to_local, conv_iso_to_local_with_daytype, conv_iso_to_local_with_daytimes
from src.exploration.core.params import timezones


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

@lru_cache(None)
def _load_df(key: str) -> pd.DataFrame:
    """Load processed pkl as df, convert timestamps once and cache copy."""
    df = conv_pkltodf(key, PROCESSED_DIR).copy()
    if "timestamps" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    if "dates" in df.columns:
        df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
    return df


def _load_time_series(key: str, country: str, rename: str) -> pd.Series:
    """Load any timeseries and returns hourly series indexed by timestamp."""
    df = _load_df(key)
    if "regions" in df.columns:
        df = df[df["regions"] == country]
    df = df.sort_values("timestamps")
    s = df.set_index("timestamps")["values"].astype("float64")
    s.name = rename
    return s


########################################################
## BITRATE + DURATION AVG HELPERS (weighted averages) ##
########################################################

def _load_weighted_dist(key: str, country: str, rename: str, mids: dict) -> pd.Series:
    """
    Convert multi-bucket distributions (bitrate, duration) into a series
    of weighted averages.
    """
    df = _load_df(key)
    if "regions" in df.columns:
        df = df[df["regions"] == country]
    df = df.sort_values("timestamps")

    # pivot such that columns = bucket categories
    df_p = df.pivot_table(index="timestamps", columns="metric", values="values", aggfunc="first").astype(float)

    cols = [c for c in df_p.columns if c in mids]
    if not cols:
        return pd.Series(name=rename, dtype="float64")

    W = np.array([mids[c] for c in cols], dtype=float)
    X = df_p[cols].to_numpy()

    den = X.sum(axis=1)
    den = np.where(den <= 0, np.nan, den)
    weighted = (X * W).sum(axis=1) / den
    weighted = np.nan_to_num(weighted, nan=0.0)

    return pd.Series(weighted, index=df_p.index, name=rename).astype("float64")


########################################################
##           PROTOCOL FRACTIONS + ENTROPY             ##
########################################################

def load_protocol_features(key: str, country: str):
    df = _load_df(key)
    if "regions" in df.columns:
        df = df[df["regions"] == country]
    df = df.sort_values("timestamps")

    df_p = df.pivot_table(index="timestamps", columns="metric", values="values", aggfunc="first").rename(columns={
        "UDP": "udp",
        "TCP": "tcp",
        "ICMP": "icmp",
        "GRE": "gre"
    }).astype(float)

    pcols = ["udp", "tcp", "icmp", "gre"]
    df_p["total"] = df_p[pcols].sum(axis=1).replace(0, 1e-6)

    for c in pcols:
        df_p[f"{c}_frac"] = df_p[c] / df_p["total"]

    # shannon entropy : H(p) = -sum_i(p_i * log_2(p_i)) with 0 * log(0) = 0
    P = df_p[[f"{c}_frac" for c in pcols]].to_numpy()
    P = np.clip(P, 1e-12, 1) # removes zeros to avoid log(0) as not computable
    P = P / P.sum(axis=1, keepdims=True) # normalizes each row to 1
    entropy = -(P * np.log2(P)).sum(axis=1)

    return pd.DataFrame({
        "udp_frac": df_p["udp_frac"],
        "tcp_frac": df_p["tcp_frac"],
        "icmp_frac": df_p["icmp_frac"],
        "gre_frac": df_p["gre_frac"],
        "protocol_entropy": entropy
    }, index=df_p.index).astype("float64")


########################################################
##       MAIN COUNTRY MERGED DATAFRAME BUILDER        ##
########################################################

def build_country_dataframe(country: str) -> pd.DataFrame:
    print(f"[INFO] Building base merged DF for country={country}")

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

    df = pd.concat([
            s_l3o, s_l3t, s_l7,
            s_http, s_http_auto, s_http_human,
            s_netflow,
            s_bots, s_ai,
            s_l3_bitrate, s_l3_duration,
        ], axis=1)
    df = df.join(df_protocol, how="outer")
    df = df.sort_index().interpolate().ffill().bfill()

    # ============================
    # 3. derived ratios
    # ============================
    eps = 1e-6
    df["ratio_l3_l7"] = df["l3_origin"] / (df["l7_traffic"] + eps)
    df["ratio_auto_human"] = df["http_auto"] / (df["http_human"] + eps)
    df["ratio_bots_http"] = df["bots_total"] / (df["http"] + eps)
    df["ratio_ai_bots_bots"] = df["ai_bots"] / (df["bots_total"] + eps)
    df["ratio_netflow_http"] = df["netflow"] / (df["http"] + eps)

    # ============================
    # 4a. local-time daytype + daytime
    # ============================

    iso_series = df.index.to_series().dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Daytype / weekday
    dt_df = conv_iso_to_local_with_daytype(iso_series, country, timezones)
    dt_df["daytype_bin"] = dt_df["daytype"].map({"Weekday": 0, "Weekend": 1}).astype("int64")

    # Daytime buckets
    daytimes = conv_iso_to_local_with_daytimes(iso_series, country, timezones)
    daytime_map = {
        "Deep night": 0,
        "Morning": 1,
        "Business hours": 2,
        "Evening": 3,
        "Early night": 4,
        "Unknown": 5,
    }
    daytimes["daytime_bin"] = daytimes["daytime"].map(daytime_map).astype("int64")

    df["weekday_idx"] = dt_df["weekday"].astype("int64")
    df["daytype_idx"] = dt_df["daytype_bin"]
    df["daytime_idx"] = daytimes["daytime_bin"]

    # ============================
    # 4b. month + week periodic encodings
    # ============================

    idx_local = conv_iso_to_local(iso_series, country, timezones)

    df["month_idx"] = idx_local.dt.month - 1
    df["week_idx"] = idx_local.dt.isocalendar().week.astype(int) - 1

    # ============================
    # 4c. time cyclic encoding
    # ============================

    # local hour-of-day and weekday cyclic
    df["hour_sin"] = np.sin(2 * np.pi * idx_local.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx_local.dt.hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["weekday_idx"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday_idx"] / 7)

    # month cyclic
    df["month_sin"] = np.sin(2 * np.pi * df["month_idx"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_idx"] / 12)

    # week cyclic
    df["week_sin"] = np.sin(2 * np.pi * df["week_idx"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_idx"] / 52)

    # ============================
    # 5. rolling aggregates
    # ============================
    
    df = df.sort_index()
    df = df.asfreq("h")

    roll_cols = ["l3_origin", "l3_target", "l7_traffic",
                 "http", "http_auto", "http_human",
                 "netflow", "bots_total", "ai_bots"]

    for col in roll_cols:
        df[f"{col}_roll3h"] = df[col].rolling(3).mean()
        df[f"{col}_roll24h"] = df[col].rolling(24).mean()

    df = df.bfill()
    return df


########################################################
##             SCALING + FINAL X MATRIX               ##
########################################################

def build_feature_matrix(country: str):
    """
    Build feature matrix for a given country.
    Returns
    -------
    X_cont : pd.DataFrame
        Scaled continuous features (float64)
    X_cat : pd.DataFrame
        Categorical index features (int64 as required for embeddings in pytorch), columns in a FIXED order.
    num_cont : int
        Number of continuous features.
    cat_dims : dict[str, int]
        Mapping from categorical column name -> cardinality.
        Keys match X_cat.columns exactly and order defines embedding order.
    scaler : RobustScaler
        Fitted scaler for continuous features.
    """
    df = build_country_dataframe(country).copy()

    # ==========================================
    # Separate categorical vs continuous
    # ==========================================

    categorical_cols = [
        "weekday_idx",
        "daytype_idx",
        "daytime_idx",
        "month_idx",
        "week_idx",
    ]

    continuous_cols = [c for c in df.columns if c not in categorical_cols]

    df_cont = df[continuous_cols].astype("float64")
    df_cat = df[categorical_cols].astype("int64")

    # ==========================================
    # Fit scaler ONLY on continuous features
    # ==========================================

    scaler = RobustScaler()
    X_cont_scaled = pd.DataFrame(
        scaler.fit_transform(df_cont),
        index=df_cont.index,
        columns=df_cont.columns
    ).astype("float64")

    # ==========================================
    # Generate embedding metadata
    # ==========================================

    num_cont = X_cont_scaled.shape[1]

    # category cardinalities as a dict[col_name: cardinality]
    cat_dims = {
        col: int(df_cat[col].max()) + 1 for col in categorical_cols
    }

    # ==========================================
    # Return 5 components:
    # - scaled continuous features
    # - integer categorical features
    # - number of continuous dims (for autoencoder)
    # - embedding cardinalities (for embedding layers)
    # - scaler
    # ==========================================

    return X_cont_scaled, df_cat, num_cont, cat_dims, scaler
 