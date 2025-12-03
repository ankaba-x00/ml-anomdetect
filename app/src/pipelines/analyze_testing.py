#!/usr/bin/env python3
"""
Visualize test-set anomaly detection results:
- loads testing errors, threshold and anomaly intervals
- plots raw signal of choice with an error overlay interactively if chosen
- generates plots (error curve with threshold and detected anomalies, anomaly intervals, error histogram, raw target signal with error overlay)

Outputs:
    plots: results/models/tested/analysis/<COUNTRY>_errorcurves_<METHOD>.png
           results/models/tested/analysis/<COUNTRY>_hist_<METHOD>.png
           results/models/tested/analysis/<COUNTRY>_intervals_<METHOD>.png
           results/models/tested/analysis/<COUNTRY>_raw_<SIGNAL>_erroroverlay_<METHOD>.png

Usage:
    python -m src.pipelines.analyze_testing [-s] [-M] [-R] <COUNTRY|all|none>
"""

import json, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from app.src.data.feature_engineering import build_country_dataframe
from app.src.data.split import timeseries_seq_split
from app.src.data.feature_engineering import COUNTRIES
from app.src.models.analysis import (
    plot_error_curve,
    plot_intervals,
    plot_error_hist,
    plot_raw_with_errors
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TESTED_DIR = PROJECT_ROOT / "results" / "models" / "tested"
ANALYSIS_TEST_DIR = TESTED_DIR / "analysis"
ANALYSIS_TEST_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               LOAD DATA             ##
#########################################

def _load_results(
        country: str, 
        method: str
    ) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    err_path = TESTED_DIR / f"{country}_errors_{method}.csv"
    thr_path = TESTED_DIR / f"{country}_threshold_{method}.json"
    int_path = TESTED_DIR / f"{country}_intervals_{method}.csv"

    if not err_path.exists():
        raise FileNotFoundError(f"Error not found: {err_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Threshold dict not found: {thr_path}")
    if not int_path.exists():
        raise FileNotFoundError(f"Intervals not found: {int_path}")

    df_err = pd.read_csv(err_path, parse_dates=["ts"])
    with open(thr_path, "r") as f:
        threshold = json.load(f)["threshold"]

    df_int = pd.read_csv(int_path, parse_dates=["start_ts", "end_ts", "duration_samples"])
    return df_err, threshold, df_int


#########################################
##                 MAIN                ##
#########################################

def analyze_raw(
        country: str, 
        method: str, 
        df_err, 
        show_plots: bool,
    ):

    TUNED_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
    scaler_path = TUNED_DIR / f"{country}_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load raw unscaled dataframe and filter out continuous cols
    df_raw = build_country_dataframe(country)
    cat_cols = ["weekday_idx", "daytype_idx", "daytime_idx", "month_idx", "week_idx"]
    cont_data = df_raw[df_raw.columns.difference(cat_cols)]
    raw_cont = cont_data.values
    ts = df_raw.index
    # extract test set
    (raw_train, _), (raw_val, _), (raw_test_cont, _) = timeseries_seq_split(
        raw_cont, np.zeros_like(raw_cont),
        train_ratio=0.75,
        val_ratio=0.15
    )
    ts_eval = ts[len(raw_train)+len(raw_val):]
    # extract computed testset errors and mask
    errors = df_err["error"].values
    mask = df_err["is_anomaly"].astype(bool).values
    # plot interactively
    while True:
        print("\nFeature signal options for plotting:")
        options = cont_data.keys()
        for i in range(0, len(options), 2):
            if i + 1 < len(options):
                print(f"{i} {options[i]:<25} {i+1} {options[i+1]}")
            else:
                print(f"{i} {options[i]:<25}")
        print("Enter index [int] or press [ENTER] to exit.")
        signal_idx = input(">>> ")
        if signal_idx == "":
            print("[INFO] No signal selected, exiting prompt.")
            break
        try:
            idx = int(signal_idx)
            name = options[int(signal_idx)]
            plot_raw_with_errors(
                name,
                ts_eval,
                raw_test_cont[:, idx],
                errors,
                mask,
                ANALYSIS_TEST_DIR,
                f"{country}_raw_{name}_erroroverlay_{method}.png",
                show_plots
            )
        except (ValueError, IndexError):
            print("[Error] Invalid index. Please enter valid integer from signal list.")

def analyze_country(
        country: str, 
        method: str, 
        show_plots: bool,
        plot_raw: bool
    ):
    """Runs full analysis pipeline of a country model testing."""
    print(f"[INFO] Analyzing {country} with {method}...")

    df_err, threshold, df_int = _load_results(country, method)
    plot_error_curve(
        country, 
        df_err, 
        threshold, 
        method, 
        ANALYSIS_TEST_DIR,
        f"{country}_errorcurve_{method}.png",
        show=show_plots
    )
    plot_intervals(
        country, 
        df_err, 
        df_int, 
        method, 
        ANALYSIS_TEST_DIR,
        f"{country}_intervals_{method}.png",
        show=show_plots
    )
    plot_error_hist(
        country, 
        df_err, 
        threshold, 
        method, 
        ANALYSIS_TEST_DIR, 
        f"{country}_hist_{method}.png",
        show=show_plots
    )

    if plot_raw:
        analyze_raw(country, method, df_err, show_plots)

    print(f"[OK] Analysis for {country} completed!")


def analyze_all(method: str, show_plots: bool, plot_raw: bool):
    """Runs full analysis pipeline of all country model testings."""
    print(f"\n[INFO] Analysis of all models starting...")

    for c in COUNTRIES:
        try:
            analyze_country(c, method, show_plots, plot_raw)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")
    print(f"\n[DONE] Analysis of all model testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model training & validation performance."
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show plots interactively when generated"
    )

    parser.add_argument(
        "-R", "--raw",
        action="store_true",
        help="plot raw signals with scaled error overlay"
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
            plot_raw=args.raw
        )
    else:
        analyze_country(
            target.upper(), 
            method=args.method, 
            show_plots=args.show,
            plot_raw=args.raw
        )
