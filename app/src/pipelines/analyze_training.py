#!/usr/bin/env python3
"""
Analyze training and validation performance for one or all countries:
- loads training history
- loads validation errors
- generates plots (loss curves, LR schedules, error histograms, error time series) and validation summary

Outputs:
    PATH: results/ml/trained/<MODEL>
    FILES : <COUNTRY_CODE>_loss_curve.png, <COUNTRY_CODE>_detailed_loss_curves.png, <COUNTRY_CODE>_lr_schedule.png
    PATH: results/ml/validated/<MODEL>
    FILES : <COUNTRY_CODE>_error_hist.png, <COUNTRY_CODE>_error_timeseries.png, <COUNTRY_CODE>_summary.json

Usage:
    python -m app.src.pipelines.analyze_training [-s] <MODEL> <COUNTRY_CODE|all>
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis.analysis import (
    plot_training_curves, plot_detailed_loss_curves, plot_error_histogram, plot_error_timeseries, summarize_validation
)


#########################################
##                PARAMS               ##
#########################################
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TRAINED_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
VALIDATED_DIR = PROJECT_ROOT / "results" / "ml" / "validated"


#########################################
##               LOAD DATA             ##
#########################################

def load_training_history(ae_type: str, country: str, models_dir: Path) -> dict:
    """Load training history for country and returns train_loss, val_loss, learning_rates, best_epoch."""
    path = Path(models_dir) / f"{ae_type.upper()}" / f"{country}_training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Training history not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_validation_errors(ae_type:str, country: str, validated_dir: Path) -> pd.DataFrame:
    """Load validation CSV and returns ts, error."""
    path = Path(validated_dir) / f"{ae_type.upper()}" / f"{country}_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Validation CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["ts"])


#########################################
##                 MAIN                ##
#########################################

def analyze_country(ae_type: str, country: str, show_plots: bool):
    """Runs full analysis pipeline of a country model."""
    print(f"[INFO] Analyzing {country}...")
    out_train = TRAINED_DIR / f"{ae_type.upper()}" / "analysis"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val = VALIDATED_DIR / f"{ae_type.upper()}" / "analysis"
    out_val.mkdir(parents=True, exist_ok=True)

    try:
        history = load_training_history(ae_type, country, TRAINED_DIR)
        val_df = load_validation_errors(ae_type, country, VALIDATED_DIR)

        threshold = np.percentile(val_df["error"], 99)

        plot_training_curves(
            country,
            history,
            out_train,
            [f"{country}_loss_curve.png", f"{country}_lr_schedule.png"],
            show_plots,
        )
        plot_detailed_loss_curves(
            ae_type,
            country,
            history,
            out_train,
            f"{country}_detailed_loss_curves.png",
            show_plots,
        )
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
    except Exception as e:
        print(f"[ERROR] Failed analyzing {country}: {e}")

    print(f"[OK] Analysis for {country} completed!")

def analyze_all(ae_type: str, show_plots: bool):
    """Runs full analysis pipeline of all country models."""
    for c in COUNTRIES:
        analyze_country(ae_type, c, show_plots)
    print(f"\n[DONE] Analysis of all model trainings and validations completed!")


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
        "model",
        help="model to train: ae, vae"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to analyse US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    target = args.target.lower()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)

    if target == "all":
        analyze_all(ae_type, show_plots=args.show)
    else:
        analyze_country(ae_type, target.upper(), show_plots=args.show)
