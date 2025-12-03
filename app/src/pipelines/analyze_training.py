#!/usr/bin/env python3
"""
Analyze training and validation performance for one or all countries:
- loads training history
- loads validation errors
- generates plots (loss curves, LR schedules, error histograms, error time series) and validation summary

Outputs:
    plots : results/models/trained/<COUNTRY_CODE>_loss_curve.png
            results/models/trained/<COUNTRY_CODE>_lr_schedule.png
            results/models/validated/<COUNTRY_CODE>_error_hist.png
            results/models/validated/<COUNTRY_CODE>_error_timeseries.png
    summary : results/models/validated/<COUNTRY_CODE>_summary.json

Usage:
    python -m src.pipelines.analyze_training [-s] <COUNTRY_CODE|all>
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.src.data.feature_engineering import COUNTRIES
from app.src.models.analysis import (
    plot_training_curves, plot_detailed_loss_curves, plot_error_histogram, plot_error_timeseries, summarize_validation
)


#########################################
##                PARAMS               ##
#########################################
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TRAINED_DIR = PROJECT_ROOT / "results" / "models" / "trained"
VALIDATED_DIR = PROJECT_ROOT / "results" / "models" / "validated"
ANALYSIS_TRAIN_DIR = TRAINED_DIR / "analysis"
ANALYSIS_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_VAL_DIR = VALIDATED_DIR / "analysis"
ANALYSIS_VAL_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               LOAD DATA             ##
#########################################

def load_training_history(country: str, models_dir: Path) -> dict:
    """Load training history for country and returns train_loss, val_loss, learning_rates, best_epoch."""
    path = Path(models_dir) / f"{country}_training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Training history not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_validation_errors(country: str, validated_dir: Path) -> pd.DataFrame:
    """Load validation CSV and returns ts, error."""
    path = Path(validated_dir) / f"{country}_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Validation CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["ts"])


#########################################
##                 MAIN                ##
#########################################

def analyze_country(country: str, show_plots: bool):
    """Runs full analysis pipeline of a country model."""
    print(f"[INFO] Analyzing {country}...")

    try:
        history = load_training_history(country, TRAINED_DIR)
        val_df = load_validation_errors(country, VALIDATED_DIR)

        threshold = np.percentile(val_df["error"], 99)

        plot_training_curves(
            country,
            history,
            ANALYSIS_TRAIN_DIR,
            [f"{country}_loss_curve.png", f"{country}_lr_schedule.png"],
            show_plots,
        )
        plot_detailed_loss_curves(
            country,
            history,
            ANALYSIS_TRAIN_DIR,
            f"{country}_detailed_loss_curves.png",
            show_plots,
        )
        plot_error_histogram(
            country,
            val_df,
            ANALYSIS_VAL_DIR,
            f"{country}_error_hist.png",
            show_plots,
        )
        plot_error_timeseries(
            country,
            val_df,
            threshold,
            ANALYSIS_VAL_DIR,
            f"{country}_error_timeseries.png",
            show_plots,
        )
        summarize_validation(
            country,
            val_df,
            ANALYSIS_VAL_DIR,
            f"{country}_summary.json"
        )
    except Exception as e:
        print(f"[ERROR] Failed analyzing {country}: {e}")

    print(f"[OK] Analysis for {country} completed!")

def analyze_all(show_plots: bool):
    """Runs full analysis pipeline of all country models."""
    for c in COUNTRIES:
        analyze_country(c, show_plots)
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
        "target",
        help="<COUNTRY|all> e.g. 'US' to analyse US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    target = args.target.lower()

    if target == "all":
        analyze_all(show_plots=args.show)
    else:
        analyze_country(target.upper(), show_plots=args.show)
