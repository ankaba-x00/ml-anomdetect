#!/usr/bin/env python3
"""
Builds and saves feature matrix for all countries:
- loads processed time-series data
- constructs merged feature dataframe for each country
- splits into:
        a) continuous features (scaled)
        b) categorical index features
- saves feature matrices for use during training and inference

Outputs:
    datasets/featured/features_<COUNTRY_CODE>.pkl

Usage:
    python -m app.src.pipelines.build_features
"""

from pathlib import Path
import pickle, json, math
import pandas as pd
from app.src.data.feature_engineering import (
    COUNTRIES,
    build_feature_matrix,
    load_feature_matrix
)
from app.src.ml.analysis.analysis import plot_log_candidates


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
FEATURE_DIR = PROJECT_ROOT / "datasets" / "featured"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = PROJECT_ROOT.parent / "results" / "featured"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               HELPER                ##
#########################################

def _extract_log_candidates(
    stats: dict,
    skew_ratio_thresh: float = 20.,
    std_ratio_thresh: float = 3.0,
    tail_ratio_thresh: float = 2.0,
    order_mag_thresh: float = 2.0
):
    """
    Analyze feature stats and returns a list of features recommended
    for log-scaling based on
    - stats [dict] : mapping feature_name -> {min, max, mean, std, p99, p999}
    - skew_ratio_thresh [float] : if max/mean > threshold == right-skewed
    - std_ratio_thresh [float] : if std/mean > threshold == high variance
    - tail_ratio_thresh [float] : if p999/p99 > threshold == heavy tail
    - order_mag_thresh [float] : if log10(max+1) - log10(mean+1) > threshold == many orders of magnitude
    Returns
    -------
        dict:
            "log_candidates" : list of features that should be log-scaled
            "reasons" : per-feature explanation
    """
    log_candidates, reasons = [], {}

    for feat, s in stats.items():
        mn = s["min"]
        mx = s["max"]
        mean = s["mean"]
        std = s["std"]
        p99 = s["p99"]
        p999 = s["p999"]

        if feat.endswith("_sin") or feat.endswith("_cos"):
            continue

        # probabilities / fractions → skip
        if 0 <= mn and mx <= 1 and std <= 0.4:
            continue

        feat_reasons = []

        # -----------------------------
        # Criteria 1: max >> mean
        # -----------------------------
        if mean > 0:
            skew_ratio = mx / mean
            if skew_ratio > skew_ratio_thresh:
                feat_reasons.append(f"max/mean = {skew_ratio:.1f} (> {skew_ratio_thresh})")

        # -----------------------------
        # Criteria 2: std >> mean
        # -----------------------------
        if mean > 0:
            std_ratio = std / mean
            if std_ratio > std_ratio_thresh:
                feat_reasons.append(f"std/mean = {std_ratio:.1f} (> {std_ratio_thresh})")

        # -----------------------------
        # Criteria 3: heavy tail
        # -----------------------------
        if p99 > 0:
            tail_ratio = p999 / p99
            if tail_ratio > tail_ratio_thresh:
                feat_reasons.append(f"p999/p99 = {tail_ratio:.1f} (> {tail_ratio_thresh})")

        # -----------------------------
        # Criteria 4: orders of magnitude
        # -----------------------------
        if mx > 0 and mean > 0:
            orders = math.log10(mx + 1) - math.log10(mean + 1)
            if orders > order_mag_thresh:
                feat_reasons.append(f"order range = {orders:.1f} (> {order_mag_thresh})")

        if feat_reasons:
            log_candidates.append(feat)
            reasons[feat] = feat_reasons

    return {
        "log_candidates": log_candidates,
        "reasons": reasons,
    }

#########################################
##                MAIN                 ##
#########################################

def analyze_feature_matrix(country: str, X_cont: pd.DataFrame, show: bool) -> None:
    """Analyze continuous features numerically and visually for tailing and value distribution to determine which features can be clambed and/or transformed to logscale."""
    
    cont_keys = ['l3_origin', 'l3_target', 'l7_traffic', 'http', 'http_auto',
       'http_human', 'netflow', 'bots_total', 'ai_bots', 'l3_bitrate_avg',
       'l3_duration_avg', 'ratio_l3_l7', 'ratio_auto_human', 'ratio_bots_http', 'ratio_ai_bots_bots', 'ratio_netflow_http', 'l3_origin_roll3h', 'l3_origin_roll24h',
       'l3_target_roll3h', 'l3_target_roll24h', 'l7_traffic_roll3h',
       'l7_traffic_roll24h', 'http_roll3h', 'http_roll24h', 'http_auto_roll3h',
       'http_auto_roll24h', 'http_human_roll3h', 'http_human_roll24h',
       'netflow_roll3h', 'netflow_roll24h', 'bots_total_roll3h',
       'bots_total_roll24h', 'ai_bots_roll3h', 'ai_bots_roll24h',
       #'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
       #'week_sin', 'week_cos', 'udp_frac', 'tcp_frac', 'icmp_frac', 'gre_frac',
       # 'protocol_entropy'
    ]

    out_dir = ANALYSIS_DIR / f"{country.upper()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for key in cont_keys:
        plot_log_candidates(key, X_cont[key], out_dir, f"hist_{key}")
        results[key] = {
            "min": X_cont[key].min(),
            "max": X_cont[key].max(),
            "mean": X_cont[key].mean(),
            "std": X_cont[key].std(),
            "p99": X_cont[key].quantile(0.99),
            "p999": X_cont[key].quantile(0.999),
            "p001": X_cont[key].quantile(0.001),
        }
    stats_path = out_dir / f"stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)

    candidates = _extract_log_candidates(results)
    candidates_path = out_dir / f"candidates.json"
    with open(candidates_path, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"[INFO] Log-candidates for {country} = ", candidates["log_candidates"])

    print(f"[OK] Feature matrix for {country} processed!")

def save_feature_matrix(country: str, X_cont: pd.DataFrame, X_cat: pd.DataFrame, num_cont: int, cat_dims: list[int]) -> None:
    """
    Save:
      - continuous scaled features (float32)
      - categorical index features (int)
      - metadata: num_cont and cat_dims
    """
    fpath = FEATURE_DIR / f"features_{country}.pkl"
    with open(fpath, "wb") as f:
        pickle.dump(
            {
                "continuous": X_cont,
                "categorical": X_cat,
                "num_cont": num_cont,
                "cat_dims": cat_dims,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[OK] Feature matrix for {country} saved!")


#########################################
##                 RUN                 ##
#########################################

def build_single_country(country: str, BUILD: bool, SAVE: bool, ANALYZE: bool, show: bool):
    print(f"\n==============================")
    print(f"  FEATURES COUNTRY = {country}")
    print(f"==============================")

    try:
        if BUILD:
            X_cont, X_cat, num_cont, cat_dims = build_feature_matrix(country)
        else:
            X_cont, X_cat, num_cont, cat_dims = load_feature_matrix(country, FEATURE_DIR)
        
        if SAVE:
            save_feature_matrix(country, X_cont, X_cat, num_cont, cat_dims)
        if ANALYZE:
            analyze_feature_matrix(country, X_cont, show)
        
    except Exception as e:
        print(f"[ERROR] Could not build {country}: {e}")


def build_all_countries(BUILD: bool = False, SAVE: bool = False, ANALYZE: bool = False, show: bool = False):
    print("[INFO] Building feature matrices...")
    for c in COUNTRIES:
        build_single_country(c, BUILD, SAVE, ANALYZE, show)
    print("[DONE] All feature matrices build!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build, save and analyze feature matrices for country models"
    )

    parser.add_argument(
        "-B", "--build",
        action="store_true",
        help="build matrices"
    )

    parser.add_argument(
        "-S", "--save",
        action="store_true",
        help="save matrices"
    )

    parser.add_argument(
        "-A", "--analyze",
        action="store_true",
        help="analyze matrices"
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show visual output [recommended: false]"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to build US matrix, or 'all' to build all matrices"
    )

    args = parser.parse_args()

    if args.target.lower() == "all":
        build_all_countries(args.build, args.save, args.analyze, args.show)
    else:
        build_single_country(args.target.upper(), args.build, args.save, args.analyze, args.show)
