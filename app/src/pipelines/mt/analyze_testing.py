#!/usr/bin/env python3
"""
Visualize test data amulti-task prediction results:
- loads testing results and anomaly intervals
- generates plots 

Outputs:
    PATH : results/mt_ml/tested/analysis/<COUNTRY>
    FILES : <COUNTRY>_mt_anomaly_timeseries_<method>.png", 
            <COUNTRY>_attack_confidence_hist_<method>.png", 
            <COUNTRY>_attack_confusion_matrix_<method>.png", 
            <COUNTRY>_intervals_<method>.png",
            <COUNTRY>_hist_<method>.png", 
            <COUNTRY>_raw_l3_erroroverlay_<method>.png", 
            <COUNTRY>_raw_l7_erroroverlay_<method>.png", 
            <COUNTRY>_attack_timeline_<method>.png",
            <COUNTRY>_loss_timeseries_<method>.png"

Usage:
    python -m app.src.pipelines.mt.analyze_testing [-s] [-M] [-R] <COUNTRY|all|none>
"""

import pandas as pd
from pathlib import Path

from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis import (
    plot_intervals,
    plot_error_hist,
    plot_mt_anomaly_timeseries,
    plot_attack_confidence_hist,
    plot_attack_confusion_matrix,
    plot_true_pred_anomalies,
    plot_attack_timeline,
    plot_loss_components_timeseries
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
TESTED_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tested"
OUT_DIR = TESTED_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               LOAD DATA             ##
#########################################

def _load_results(
    country: str, 
    method: str
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    result_path = TESTED_DIR / f"{country}_errors_{method}.csv"
    intervals_path = TESTED_DIR / f"{country}_intervals_{method}.csv"

    if not result_path.exists():
        raise FileNotFoundError(f"Errors not found: {result_path}")
    if not intervals_path.exists():
        raise FileNotFoundError(f"Intervals not found: {intervals_path}")


    df_res = pd.read_csv(result_path, parse_dates=["ts"])
    df_int = pd.read_csv(
        intervals_path, 
        parse_dates=["start_ts", "end_ts"],
        date_format="ISO8601"
    )
    
    return df_res, df_int


#########################################
##                 MAIN                ##
#########################################

def analyze_country(
    country: str, 
    method: str, 
    show_plots: bool,
) -> None:
    """Runs full analysis pipeline of a country MT model testing."""
    print(f"[INFO] Analyzing {country} with {method}...")

    out_dir = OUT_DIR / country
    out_dir.mkdir(parents=True, exist_ok=True)

    df_res, df_int = _load_results(country, method)

    plot_mt_anomaly_timeseries(
        country,
        df_res,
        out_dir,
        f"{country}_mt_anomaly_timeseries_{method}.png",
        show=show_plots
    )
    plot_attack_confidence_hist(
        country,
        df_res,
        out_dir,
        f"{country}_attack_confidence_hist_{method}.png",
        show=show_plots
    )
    plot_attack_confusion_matrix(
        country,
        df_res,
        out_dir,
        f"{country}_attack_confusion_matrix_{method}.png",
        show=show_plots
    )
    plot_intervals(
        country, 
        df_res, 
        df_int, 
        method, 
        out_dir,
        f"{country}_intervals_{method}.png",
        show=show_plots
    )
    plot_error_hist(
        country, 
        df_res, 
        df_res["threshold"][0], 
        method, 
        out_dir, 
        f"{country}_hist_{method}.png",
        show=show_plots,
        MT=True
    )
    plot_true_pred_anomalies(
        "L3 intensities",
        df_res["ts"],
        df_res["l3_true"],
        df_res["l3_pred"],
        df_res["is_flagged"],
        df_int["start_idx"],
        df_int["end_idx"],
        out_dir,
        f"{country}_raw_l3_erroroverlay_{method}.png",
        show=show_plots
    )
    plot_true_pred_anomalies(
        "L7 intensities",
        df_res["ts"],
        df_res["l7_true"],
        df_res["l7_pred"],
        df_res["is_flagged"],
        df_int["start_idx"],
        df_int["end_idx"],
        out_dir,
        f"{country}_raw_l7_erroroverlay_{method}.png",
        show=show_plots
    )
    plot_attack_timeline(
        df_res,
        out_dir,
        f"{country}_attack_timeline_{method}.png",
        show=show_plots
    )
    plot_loss_components_timeseries(
        df_res,
        out_dir,
        f"{country}_loss_timeseries_{method}.png",
        show=show_plots   
    )

    print(f"[OK] Analysis for {country} completed!")


def analyze_all(
    method: str, 
    show_plots: bool
) -> None:
    """Runs full analysis pipeline of all country MT model testings."""
    print(f"\n[INFO] Analysis of all MT models starting...")

    for c in COUNTRIES:
        try:
            analyze_country(c, method, show_plots)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")
    
    print(f"\n[DONE] Analysis of all MT model testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MT model testing performance."
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show plots interactively when generated"
    )
    
    parser.add_argument(
        "-M", "--method", 
        choices=["p99", "p995", "mad"], 
        default="p99",
        help="threshold method [default: p99]"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all|none> e.g. 'US' to analyse US model, or 'all' to evaluate all country models"
    )
    
    args = parser.parse_args()

    target = args.target

    if target.lower() == "all":
        analyze_all(
            method=args.method, 
            show_plots=args.show,
        )
    else:
        analyze_country(
            country=target.upper(), 
            method=args.method, 
            show_plots=args.show,
        )
