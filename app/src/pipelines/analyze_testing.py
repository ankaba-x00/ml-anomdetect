#!/usr/bin/env python3
"""
Analyze model performance after testing for one or multiple countries:
- loads scores, threshold and anomaly intervals
- generates plots
- (optional) plots raw signal with error overlay interactively

Output:
    PATH : results/ml/tested/<MODEL>/analysis/<COUNTRY>
    FILES for AE/VAE: <COUNTRY>_errorcurves_<METHOD>.png
                      <COUNTRY>_hist_<METHOD>.png
                      <COUNTRY>_intervals_<METHOD>.png 
                      <COUNTRY>_raw_<SIGNAL>_erroroverlay_<METHOD>.png
    FILES for MTAE: <COUNTRY>_mt_anomaly_timeseries_<METHOD>.png
                    <COUNTRY>_attack_confidence_hist_<METHOD>.png 
                    <COUNTRY>_attack_confusion_matrix_<METHOD>.png
                    <COUNTRY>_l7_regression_scatter_<METHOD>.png
                    <COUNTRY>_l3_regression_scatter_<METHOD>.png
                    <COUNTRY>_intervals_<METHOD>.png
                    <COUNTRY>_hist_<METHOD>.png
                    <COUNTRY>_raw_l3_erroroverlay_<METHOD>.png 
                    <COUNTRY>_raw_l7_erroroverlay_<METHOD>.png 
                    <COUNTRY>_attack_timeline_<METHOD>.png
                    <COUNTRY>_loss_timeseries_<METHOD>.png

Usage:
    python -m app.src.pipelines.analyze_testing [-s] [-R] [-M <p99|p995|mad>] <MODEL> <COUNTRY|all>
"""

import json, pickle
import pandas as pd
import numpy as np
from pathlib import Path

from app.src.data.feature_engineering import load_feature_matrix, load_supervised_feature_matrix
from app.src.data import ISO_3166_alpha2, timeseries_seq_split
from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis import (
    plot_error_curve,
    plot_intervals,
    plot_error_hist,
    plot_raw_with_scores,
    plot_anomaly_timeseries,
    plot_attack_confidence_hist,
    plot_attack_confusion_matrix,
    plot_regression_scatter,
    plot_true_pred_anomalies,
    plot_attack_timeline,
    plot_loss_components_timeseries
)


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TESTED_DIR = PROJECT_ROOT / "results" / "ml" / "tested"


def _load_data(
    ae_type: str,
    country: str, 
    method: str
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    err_path = TESTED_DIR / f"{ae_type.upper()}" / f"{country}_scores_{method}.csv"
    thr_path = TESTED_DIR / f"{ae_type.upper()}" / f"{country}_threshold_{method}.json"
    int_path = TESTED_DIR / f"{ae_type.upper()}" / f"{country}_intervals_{method}.csv"

    if not err_path.exists():
        raise FileNotFoundError(f"[ERROR] Scores not found: {err_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"[ERROR] Threshold not found: {thr_path}")
    if not int_path.exists():
        raise FileNotFoundError(f"[ERROR] Intervals not found: {int_path}")

    df_err = pd.read_csv(err_path, parse_dates=["ts"])
    with open(thr_path, "r") as f:
        threshold = json.load(f)["threshold"]

    df_int = pd.read_csv(
        int_path, 
        parse_dates=["start_ts", "end_ts"],
        date_format="ISO8601"
    )

    return df_err, threshold, df_int

def analyze_raw(
    ae_type: str,
    country: str, 
    method: str, 
    df_err: pd.DataFrame, 
    out_path: Path,
    show_plots: bool,
) -> None:
    TESTED_DIR = PROJECT_ROOT / "results" / "ml" / "trained" / f"{ae_type.upper()}"
    scaler_path = TESTED_DIR / f"{country}_scaler.pkl"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    if ae_type in ["ae", "vae"]:
        X_cont, X_cat, num_cont, cat_dims = load_feature_matrix(country)
    else:
        X_cont, X_cat, y_l3, y_l7, y_at, num_cont, cat_dims = (
            load_supervised_feature_matrix(country)
        )

    (raw_tr, _), (raw_val, _), (raw_te, _) = timeseries_seq_split(
        X_cont.values, np.zeros_like(X_cont.values),
        0.75,
        0.15
    )

    raw_te_scald = scaler.transform(raw_te).astype(np.float32)
    ts_eval = X_cont.index[len(raw_tr)+len(raw_val):]
    
    scores = df_err["scores"].values
    mask = df_err["is_flagged"].astype(bool).values

    if len(scores) != raw_te_scald.shape[0]:
        raise ValueError(f"[ERROR] Mismatch of dataset split ratio between test run and analysis")
    
    print("\nFeature signal options for plotting:")
    options = X_cont.columns
    for i in range(0, len(options), 2):
        if i + 1 < len(options):
            print(f"{i} {options[i]:<25} {i+1} {options[i+1]}")
        else:
            print(f"{i} {options[i]:<25}")
    
    while True:
        print("Enter index [int] or press [ENTER] to exit")
        signal_idx = input(">>> ")
        
        if signal_idx == "":
            print("[INFO] No signal selected, exiting prompt")
            break
    
        try:
            idx = int(signal_idx)
            name = options[int(signal_idx)]
            plot_raw_with_scores(
                name,
                ts_eval,
                raw_te_scald[:, idx],
                scores,
                mask,
                out_path,
                f"{country}_raw_{name}_erroroverlay_{method}.png",
                True
            )
        except IndexError:
            print("[Error] Enter valid integer from signal list")

def analyze_testing(
    ae_type: str,
    country: str, 
    method: str, 
    show_plots: bool,
    plot_raw: bool
) -> None:
    """Runs full analysis pipeline for autoencoder after testing."""
    print(f"[INFO] Analyzing {country} with {method}...")

    # --------------------
    # Set config
    # --------------------
    out_path = TESTED_DIR / f"{ae_type.upper()}" / "analysis" / country
    out_path.mkdir(parents=True, exist_ok=True)

    df_err, threshold, df_int = _load_data(ae_type, country, method)

    if ae_type in ["mtae"]:
        pred_quantiles = [0.5, 0.75, 0.9]
        print(f"[INFO] Quantiles used for prediction: {pred_quantiles}")

    plot_error_curve(
        country, 
        df_err, 
        threshold, 
        method, 
        out_path,
        f"{country}_errorcurve_{method}.png",
        show_plots
    )
    plot_intervals(
        country, 
        df_err, 
        df_int, 
        method, 
        out_path,
        f"{country}_intervals_{method}.png",
        show_plots
    )
    plot_error_hist(
        country, 
        df_err, 
        threshold, 
        method, 
        out_path, 
        f"{country}_hist_{method}.png",
        show_plots
    )

    if ae_type in ["mtae"]:
        plot_anomaly_timeseries(
            country,
            df_err,
            out_path,
            f"{country}_mt_anomaly_timeseries_{method}.png",
            show_plots
        )
        plot_attack_confidence_hist(
            country,
            df_err,
            out_path,
            f"{country}_attack_confidence_hist_{method}.png",
            show_plots
        )
        plot_attack_confusion_matrix(
            country,
            df_err,
            out_path,
            f"{country}_attack_confusion_matrix_{method}.png",
            show_plots
        )
        for q in pred_quantiles:
            plot_regression_scatter(
                df_err["l3_true"],
                df_err[f"l3_pred_{q}"],
                "L3 Intensity",
                5000,
                out_path,
                f"{country}_l3_regression_scatter_{q}.png",
                show_plots
            )
            plot_regression_scatter(
                df_err["l7_true"],
                df_err[f"l7_pred_{q}"],
                "L7 Intensity",
                5000,
                out_path,
                f"{country}_l7_regression_scatter_{q}.png",
                show_plots
            )
            plot_true_pred_anomalies(
                "L3 intensities",
                df_err["ts"],
                df_err["l3_true"],
                df_err[f"l3_pred_{q}"],
                df_err["is_flagged"],
                df_int["start_idx"],
                df_int["end_idx"],
                out_path,
                f"{country}_l3_raw_erroroverlay_{q}.png",
                show_plots
            )
            plot_true_pred_anomalies(
                "L7 intensities",
                df_err["ts"],
                df_err["l7_true"],
                df_err[f"l7_pred_{q}"],
                df_err["is_flagged"],
                df_int["start_idx"],
                df_int["end_idx"],
                out_path,
                f"{country}_l7_raw_erroroverlay_{q}.png",
                show_plots
            )
        plot_attack_timeline(
            df_err,
            out_path,
            f"{country}_attack_timeline_{method}.png",
            show_plots
        )
        plot_intervals(
            country, 
            df_err, 
            df_int, 
            method, 
            out_path,
            f"{country}_intervals_{method}.png",
            show_plots
        )
        plot_loss_components_timeseries(
            df_err,
            out_path,
            f"{country}_loss_timeseries_{method}.png",
            show_plots   
        )

    if plot_raw:
        analyze_raw(
            ae_type, 
            country, 
            method, 
            df_err, 
            out_path, 
            show_plots
        )

    print(f"[DONE] Analysis for {country}")

def analyze_all(
    ae_type: str, 
    method: str, 
    show_plots: bool, 
    plot_raw: bool
) -> None:
    for c in COUNTRIES:
        try:
            analyze_testing(ae_type, c, method, show_plots, plot_raw)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")
            
    print(f"\n[DONE] All analysis completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model testing performance"
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
        "model",
        help="<ae|vae|mtae> model to train"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' or 'all' for all countries defined in config/models.yaml"
    )
    
    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae", "mtae"]:
        parser.print_help()
        exit(1)
    
    target = args.target.upper()
    if target not in ISO_3166_alpha2 and target.lower() != "all":
        parser.print_help()
        print(f"[ERROR] Invalid target: {target}")
        exit(1)

    if target.lower() == "all":
        analyze_all(
            ae_type,
            args.method, 
            args.show,
            args.raw
        )
    else:
        analyze_testing(
            ae_type,
            target, 
            args.method, 
            args.show,
            args.raw
        )
