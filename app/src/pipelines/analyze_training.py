#!/usr/bin/env python3
"""
Analyze model performance after training and validation for one or multiple countries:
- loads training history and validation errors
- generates plots and summary

Output:
    PATH: results/ml/trained/<MODEL>/analysis/<COUNTRY>
    FILES : <COUNTRY>_loss_curve.png 
            <COUNTRY>_lr_schedule.png
            <COUNTRY>_detailed_loss_curves.png
    PATH: results/ml/validated/<MODEL>/analysis/<COUNTRY>
    FILES for AE/VAE: <COUNTRY>_error_hist.png 
                      <COUNTRY>_error_timeseries.png 
                      <COUNTRY>_summary.json
    FILES for MTAE: <COUNTRY>_l7_regression_scatter_<QUANTILE>.png 
                    <COUNTRY>_l3_regression_scatter_<QUANTILE>.png
                    <COUNTRY>_attack_confidence_hist.png 
                    <COUNTRY>_attack_confusion_matrix.png 
                    <COUNTRY>_anomaly_timeseries.png
                    <COUNTRY>_summary.json

Usage:
    python -m app.src.pipelines.analyze_training [-s] [-M] <MODEL> <COUNTRY|all>
"""

import json
from pathlib import Path
import pandas as pd

from app.src.data import ISO_3166_alpha2
from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis import (
    plot_training_curves, 
    plot_detailed_loss_curves, 
    plot_error_histogram, 
    plot_error_timeseries, 
    summarize_validation,
    plot_detailed_mt_loss_curves,
    summarize_mt_validation, 
    plot_regression_scatter, 
    plot_attack_confusion_matrix, 
    plot_attack_confidence_hist, 
    plot_anomaly_timeseries
)
from app.src.ml.training.anomaly_utils import get_threshold


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TRAIN_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
VAL_DIR = PROJECT_ROOT / "results" / "ml" / "validated"


def analyze_training(
    ae_type: str, 
    country: str, 
    method: bool,
    show_plots: bool
) -> None:
    """Runs full analysis pipeline for autoencoder after training and validation."""
    print(f"[INFO] Analyzing {country}...")
   
    # --------------------
    # Load data
    # --------------------
    in_train = TRAIN_DIR / f"{ae_type.upper()}"
    out_train = in_train / "analysis" / country    
    out_train.mkdir(parents=True, exist_ok=True)

    in_val = VAL_DIR / f"{ae_type.upper()}" 
    out_val = in_val / "analysis" / country
    out_val.mkdir(parents=True, exist_ok=True)

    history_path = in_train / f"{country}_training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"[ERROR] Training history not found: {history_path}")
    with open(history_path, "r") as f:
        history = json.load(f)

    df_path =  in_val / f"{country}_validation.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"[ERROR] Validation CSV not found: {df_path}")
    val_df = pd.read_csv(df_path, parse_dates=["ts"])


    # --------------------
    # Set config
    # --------------------
    threshold = get_threshold(method, val_df["scores"])
    if ae_type in ["mtae"]:
        pred_quantiles = [0.5, 0.75, 0.9]
        print(f"[INFO] Quantiles used for prediction: {pred_quantiles}")

    # --------------------
    # Training
    # --------------------
    plot_training_curves(
        country,
        history,
        out_train,
        [f"{country}_loss_curve.png", f"{country}_lr_schedule.png"],
        show_plots,
    )
    if ae_type in ["ae", "vae"]:
        plot_detailed_loss_curves(
            ae_type,
            country,
            history,
            out_train,
            f"{country}_detailed_loss_curves.png",
            show_plots,
        )
    else:
        plot_detailed_mt_loss_curves(
            country,
            history,
            out_train,
            f"{country}_detailed_loss_curves.png",
            show_plots,
        )
    
    # --------------------
    # Validation
    # --------------------
    if ae_type in ["ae", "vae"]:
        plot_error_histogram(
            country,
            val_df,
            out_val,
            f"{country}_error_hist.png",
            show_plots,
        )
        plot_error_timeseries(
            country,
            val_df,
            threshold,
            out_val,
            f"{country}_error_timeseries.png",
            show_plots,
        )
        summarize_validation(
            country,
            val_df,
            out_val,
            f"{country}_summary.json"
        )
    else:
        for q in pred_quantiles:
            plot_regression_scatter(
                val_df["l3_true"],
                val_df[f"l3_pred_{q}"],
                f"L3 Intensity [{int(q*100)}th]",
                5000,
                out_val,
                f"{country}_l3_regression_scatter_{q}.png",
                show_plots
            )
            plot_regression_scatter(
                val_df["l7_true"],
                val_df[f"l7_pred_{q}"],
                f"L7 Intensity [{int(q*100)}th]",
                5000,
                out_val,
                f"{country}_l7_regression_scatter_{q}.png",
                show_plots
            )
        plot_attack_confusion_matrix(
            country,
            val_df,
            out_val,
            f"{country}_attack_confusion_matrix.png",
            show_plots
        )
        plot_attack_confidence_hist(
            country,
            val_df,
            out_val,
            f"{country}_attack_confidence_hist.png",
            show_plots
        )
        plot_anomaly_timeseries(
            country,
            val_df,
            out_val,
            f"{country}_anomaly_timeseries.png",
            show_plots
        )
        summarize_mt_validation(
            country,
            val_df,
            out_val,
            f"{country}_summary.json"
        )

    print(f"[DONE] Analysis for {country}")


def analyze_all(
    ae_type: str, 
    method: str,
    show_plots: bool
) -> None:    
    for c in COUNTRIES:
        try:
            analyze_training(
                ae_type, 
                c, 
                method, 
                show_plots
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    
    print(f"\n[DONE] All analysis completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model training and validation performance"
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
            args.show
        )
    else:
        analyze_training(
            ae_type, 
            target, 
            args.method,
            args.show
        )
