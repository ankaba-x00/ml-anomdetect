import yaml, numpy.core, pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast, Optional

from app.deployment import RegionTimeseriesFetchResult
from app.src.data.building import FeatureMatrix
from app.src.data.building.build_utils import load_base
from app.src.data.analysis.time_utils import (
    conv_iso_to_local, 
    conv_iso_to_local_with_daytype, 
    conv_iso_to_local_with_daytimes
)
from app.src.data.analysis.params import timezones
from app.src.data.building.attack_labelling import (
    compute_attack_thresholds, 
    temporal_attack_labeling, 
    score_attack_types, 
    ATTACK_TO_ID
)

sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
FEATURE_DIR = PROJECT_ROOT / "datasets" / "featured"

def load_countries_from_config(models_path: Path) -> list[str]:
    with open(models_path, "r") as f:
        countries: list[str] = yaml.safe_load(f)
    return countries

COUNTRIES = load_countries_from_config(PROJECT_ROOT / "config" / "models.yml")


def build_country_dataframe(
    country: str, 
    data: RegionTimeseriesFetchResult | None = None,
    attack_label: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """Builds raw feature matrix as df for a given country."""
    print(f"[INFO] Building base DF for country={country}")

    # ------------------------------------
    # 1. load all base series
    # ------------------------------------
    bundle = load_base(country, data)

    # ------------------------------------
    # 2. merge everything
    # ------------------------------------
    df = pd.concat([
            bundle.l3o, bundle.l3t, bundle.l7,
            bundle.http, bundle.http_auto, bundle.http_human,
            bundle.netflow,
            bundle.bots, bundle.ai,
            bundle.l3_bitrate, bundle.l3_duration,
        ], axis=1)
    df = df.join(bundle.protocol, how="outer")
    df = df.sort_index().interpolate().ffill().bfill()

    # ------------------------------------
    # 3. derived ratios
    # ------------------------------------
    eps = 1e-6
    df["ratio_l3_l7"] = df["l3_origin"] / (df["l7_traffic"] + eps)
    df["ratio_auto_human"] = df["http_auto"] / (df["http_human"] + eps)
    df["ratio_bots_http"] = df["bots_total"] / (df["http"] + eps)
    df["ratio_ai_bots_bots"] = df["ai_bots"] / (df["bots_total"] + eps)
    df["ratio_netflow_http"] = df["netflow"] / (df["http"] + eps)

    # ------------------------------------
    # 4a. local-time daytype + daytime
    # ------------------------------------
    iso_series = df.index.to_series().dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Daytype / weekday
    dtyp_df = pd.DataFrame(conv_iso_to_local_with_daytype(iso_series, country, timezones), index=df.index)

    dtyp_df["daytype_bin"] = dtyp_df["daytype"].map({"Weekday": 0, "Weekend": 1}).astype("int64")

    # Daytime buckets
    dtim_df = pd.DataFrame(conv_iso_to_local_with_daytimes(iso_series, country, timezones), index=df.index)
    daytime_map = {
        "Deep night": 0,
        "Morning": 1,
        "Business hours": 2,
        "Evening": 3,
        "Early night": 4,
        "Unknown": 5,
    }
    dtim_df["daytime_bin"] = dtim_df["daytime"].map(daytime_map).astype("int64")

    df["weekday_idx"] = dtyp_df["weekday"].astype("int64")
    df["daytype_idx"] = dtyp_df["daytype_bin"]
    df["daytime_idx"] = dtim_df["daytime_bin"]

    # ------------------------------------
    # 4b. month + week periodic encodings
    # ------------------------------------
    idx_local = conv_iso_to_local(iso_series, country, timezones)

    if not isinstance(idx_local, pd.Series):
        raise ValueError("[ERROR] Conversion local time indices failed")
    
    df["month_idx"] = idx_local.dt.month - 1
    df["week_idx"] = idx_local.dt.isocalendar().week.astype(int) - 1

    # ------------------------------------
    # 4c. time cyclic encoding
    # ------------------------------------
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

    # ------------------------------------
    # 5. rolling aggregates
    # ------------------------------------
    df = df.sort_index()
    df = df.asfreq("h")

    roll_cols = ["l3_origin", "l3_target", "l7_traffic",
                 "http", "http_auto", "http_human",
                 "netflow", "bots_total", "ai_bots"]

    for col in roll_cols:
        df[f"{col}_roll3h"] = df[col].rolling(3).mean()
        df[f"{col}_roll24h"] = df[col].rolling(24).mean()

    df = df.bfill()

    # ------------------------------------
    # 6. attack labels
    # ------------------------------------
    if attack_label:
        print(f"[INFO] Computing attack labels for country={country}")

        thresholds = compute_attack_thresholds(df)

        scores_df = df.apply(
            lambda r: score_attack_types(r, thresholds),
            axis=1).apply(pd.Series)
        scores_df = cast(pd.DataFrame, scores_df)

        semantic_labels = scores_df.idxmax(axis=1).map(ATTACK_TO_ID)
        if debug:
            df["uncorr_attack_label"] = semantic_labels

        df["attack_label"] = temporal_attack_labeling(semantic_labels, scores_df)

    return df

def build_feature_matrix(
    country: str, 
    df: Optional[pd.DataFrame] = None
) -> FeatureMatrix:
    """
    Build feature matrix for a given country.
    Returns
    -------
    X_cont : pd.DataFrame
        Continuous features (float64)
    X_cat : pd.DataFrame
        Categorical index features (int64 as required for embeddings in pytorch), columns in a FIXED order.
    num_cont : int
        Number of continuous features.
    cat_dims : dict[str, int]
        Mapping from categorical column name -> cardinality.
        Keys match X_cat.columns exactly and order defines embedding order.
    """
    if df is None:
        df = build_country_dataframe(country, attack_label=False).copy()

    # ------------------------------------
    # Separate categorical vs continuous
    # ------------------------------------
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

    # ------------------------------------
    # Clamp OR logscale for cont features
    # ------------------------------------
    # clamps extremely large raw values
    #df_cont = df_cont.clip(lower=df_cont.quantile(0.005), upper=df_cont.quantile(0.999), axis=1)
    
    # transform to log scale
    LOG_FEATURES = [
        "l3_bitrate_avg",
        "ratio_l3_l7",
        "ratio_bots_http",
        "ratio_netflow_http",
        #"l3_duration_avg"
        #"ratio_ai_bots_bots",
        #"ai_bots",
        #"bots_total",
    ]
    eps = 1e-6
    for feat in LOG_FEATURES:
        if feat in df_cont.columns:
            df_cont[feat] = np.log1p(df_cont[feat].clip(lower=0) + eps)

    # ------------------------------------
    # Generate embedding metadata
    # ------------------------------------
    num_cont = df_cont.shape[1]
    # category cardinalities as a dict[col_name: cardinality]
    cat_dims = {
        col: int(df_cat[col].max()) + 1 for col in categorical_cols
    }

    print(f"[OK] Feature matrix for {country} build!")

    return FeatureMatrix(
        X_cont=df_cont,
        X_cat=df_cat,
        num_cont=num_cont, 
        cat_dims=cat_dims
    )

def build_supervised_feature_matrix(
    country: str,
    df: Optional[pd.DataFrame] = None
) -> FeatureMatrix:
    """
    Builds feature matrices for a given country for multi-task prediction.
    Returns
    -------
    X_cont : pd.DataFrame
        Continuous features (float64) with traffic data only, no attack information 
    X_cat : pd.DataFrame
        Categorical index features (int64 as required for embeddings in pytorch), columns in a FIXED order with temporal data.
    y_l3 : pd.Series
        L3 attack intensity for regression
    y_l7 : pd.Series
        L7 attack intensity for regression
    y_attack : pd.Series
        Attack classes for classification
    num_cont : int
        Number of continuous features.
    cat_dims : dict[str, int]
        Mapping from categorical column name -> cardinality.
        Keys match X_cat.columns exactly and order defines embedding order.
    """
    if df is None:
        df = build_country_dataframe(country, attack_label=True).copy()
    
    # ------------------------------------
    # Separate categorical vs continuous
    # ------------------------------------
    continuous_traffic_cols = [
        'http', 'http_auto', 'http_human', 'netflow', 'bots_total', 'ai_bots',
        'ratio_auto_human', 'ratio_bots_http', 'ratio_ai_bots_bots', 
        'ratio_netflow_http', 'http_roll3h', 'http_roll24h', 'http_auto_roll3h', 
        'http_auto_roll24h', 'http_human_roll3h', 'http_human_roll24h',
        'netflow_roll3h', 'netflow_roll24h', 'bots_total_roll3h', 
        'bots_total_roll24h', 'ai_bots_roll3h', 'ai_bots_roll24h', 'hour_sin', 
        'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'week_sin', 
        'week_cos'
    ]
    categorical_cols = [
        'weekday_idx', 'daytype_idx', 'daytime_idx', 'month_idx', 'week_idx'
    ]

    # ------------------------------------
    # 1. Input features
    # ------------------------------------
    df_cont = df[continuous_traffic_cols].astype("float64")
    df_cat = df[categorical_cols].astype("int64")

    # clamps extremely large raw values
    #df_cont = df_cont.clip(lower=df_cont.quantile(0.005), upper=df_cont.quantile(0.999), axis=1)
    
    # transform to log scale
    LOG_FEATURES = [
        "ratio_bots_http",
        "ratio_netflow_http",
        #"ratio_ai_bots_bots",
        #"ai_bots",
        #"bots_total",
    ]
    eps = 1e-6
    for feat in LOG_FEATURES:
        if feat in df_cont.columns:
            df_cont[feat] = np.log1p(df_cont[feat].clip(lower=0) + eps)

    # ------------------------------------
    # 2. Labels
    # ------------------------------------
    y_l3 = df["l3_origin"].astype("float64")
    y_l7 = df["l7_traffic"].astype("float64")
    y_at = df["attack_label"].astype("int64")

    # ------------------------------------
    # 3. Embedding metadata
    # ------------------------------------
    num_cont = df_cont.shape[1]
    # category cardinalities as a dict[col_name: cardinality]
    cat_dims = {
        col: int(df_cat[col].max()) + 1 for col in categorical_cols
    }

    print(f"[OK] Supervised feature matrix for {country} build!")

    return FeatureMatrix(
        X_cont=df_cont,
        X_cat=df_cat,
        y_l3=y_l3,
        y_l7=y_l7,
        y_at=y_at,
        num_cont=num_cont, 
        cat_dims=cat_dims
    )

def load_feature_matrix(
    country: str, 
    load_path: Path = FEATURE_DIR
) -> FeatureMatrix:
    fpath = load_path / f"features_{country}.pkl"
    if not fpath.exists():
        raise FileNotFoundError(f"[ERROR] Feature matrix does not exist: {fpath}")
    
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    df_cont = data.get("continuous")
    df_cat  = data.get("categorical")
    num_cont = data.get("num_cont")
    cat_dims = data.get("cat_dims")
    
    if df_cont is None or df_cat is None: 
        raise ValueError(f"[ERROR] Feature matrix file incomplete for {country}: missing feature component")
    elif not isinstance(df_cont, pd.DataFrame) or not isinstance(df_cat, pd.DataFrame):
        raise TypeError(f"[ERROR] Feature matrix file incompatible for {country}: wrong feature component type")
    if num_cont is None or cat_dims is None:
        raise ValueError(f"[ERROR] Feature matrix file incomplete for {country}: missing metadata component")
    elif not isinstance(num_cont, int) or not isinstance(cat_dims, dict):
        raise TypeError(f"[ERROR] Feature matrix file incompatible for {country}: wrong metadata component type")

    print(f"[OK] Feature matrix for {country} loaded!")

    return FeatureMatrix(
        X_cont=df_cont,
        X_cat=df_cat,
        num_cont=num_cont, 
        cat_dims=cat_dims
    )

def load_supervised_feature_matrix(
    country: str, 
    load_path: Path = FEATURE_DIR
) -> FeatureMatrix:
    fpath = load_path / f"super_features_{country}.pkl"
    if not fpath.exists():
        raise FileNotFoundError(f"[ERROR] Supervised feature matrix does not exist: {fpath}")
    
    with open(fpath, "rb") as f:
        data = pickle.load(f)

    df_cont = data.get("continuous")
    df_cat = data.get("categorical")
    y_l3 = data.get("label_l3")
    y_l7 = data.get("label_l7")
    y_at = data.get("label_type")
    num_cont = data.get("num_cont")
    cat_dims = data.get("cat_dims")
    
    if df_cont is None or df_cat is None: 
        raise ValueError(f"[ERROR] Supervised feature matrix file incomplete for {country}: missing feature component")
    elif not isinstance(df_cont, pd.DataFrame) or not isinstance(df_cat, pd.DataFrame):
        raise TypeError(f"[ERROR] Supervised feature matrix file incompatible for {country}: wrong feature component type")
    
    if y_l3 is None or y_l7 is None or y_at is None:
        raise ValueError(f"[ERROR] Supervised feature matrix file incomplete for {country}: missing feature component")
    elif not isinstance(y_l3, pd.Series) or not isinstance(y_l7, pd.Series) or not isinstance(y_at, pd.Series):
        raise TypeError(f"[ERROR] Supervised feature matrix file incompatible for {country}: wrong attack label component type")    
    
    if num_cont is None or cat_dims is None:
        raise ValueError(f"[ERROR] Supervised feature matrix file incomplete for {country}: missing metadata component")
    elif not isinstance(num_cont, int) or not isinstance(cat_dims, dict):
        raise TypeError(f"[ERROR] Supervised feature matrix file incompatible for {country}: wrong metadata component type")

    print(f"[OK] Supervised feature matrix for {country} loaded!")

    return FeatureMatrix(
        X_cont=df_cont,
        X_cat=df_cat,
        y_l3=y_l3,
        y_l7=y_l7,
        y_at=y_at,
        num_cont=num_cont, 
        cat_dims=cat_dims
    )
