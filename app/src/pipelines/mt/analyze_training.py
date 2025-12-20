#!/usr/bin/env python3
"""
Analyze multi-task training and validation performance for one or all countries:
- loads training history
- loads validation summary
- generates plots and summary

Outputs:
    PATH: results/mt_ml/trained/<COUNTRY>
    FILES : <COUNTRY>_loss_curve.png, 
            <COUNTRY>_detailed_loss_curves.png, 
            <COUNTRY>_lr_schedule.png
    PATH: results/mt_ml/validated/<COUNTRY>
    FILES : <COUNTRY>_l7_regression_scatter.png, 
            <COUNTRY>_l3_regression_scatter.png,
            <COUNTRY>_attack_confidence_hist.png, 
            <COUNTRY>_attack_confusion_matrix.png, 
            <COUNTRY>_mt_anomaly_timeseries.png, 
            <COUNTRY>_summary.json

Usage:
    python -m app.src.pipelines.mt.analyze_training [-s] <COUNTRY|all>
"""

import json
from pathlib import Path
import pandas as pd

from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis.analysis import plot_training_curves
from app.src.ml.analysis.analysis_mt import (
    plot_detailed_mt_loss_curves, 
    summarize_mt_validation, 
    plot_regression_scatter, 
    plot_attack_confusion_matrix, 
    plot_attack_confidence_hist, 
    plot_mt_anomaly_timeseries
)

#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
TRAINED_DIR = PROJECT_ROOT / "results" / "mt_ml" / "trained"
VALIDATED_DIR = PROJECT_ROOT / "results" / "mt_ml" / "validated"


#########################################
##               LOAD DATA             ##
#########################################

def load_training_history(
    country: str, 
    models_dir: Path
) -> dict:
    """Load training history for country."""
    path = Path(models_dir) / f"{country}_training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Training history not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_validation_errors(
    country: str, 
    validated_dir: Path
) -> pd.DataFrame:
    """Load validation CSV."""
    path = Path(validated_dir) / f"{country}_mt_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Validation CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["ts"])


#########################################
##                 MAIN                ##
#########################################

def analyze_country(
    country: str, 
    show_plots: bool
) -> None:
    """Runs full analysis pipeline of a country MT model."""
    print(f"[INFO] Analyzing {country}...")
    out_train = TRAINED_DIR / "analysis" / country
    out_train.mkdir(parents=True, exist_ok=True)
    out_val = VALIDATED_DIR / "analysis" / country
    out_val.mkdir(parents=True, exist_ok=True)

    try:
        history = load_training_history(country, TRAINED_DIR)
        val_df = load_validation_errors(country, VALIDATED_DIR)

        plot_training_curves(
            country,
            history,
            out_train,
            [f"{country}_loss_curve.png", f"{country}_lr_schedule.png"],
            show_plots,
            True
        )
        plot_detailed_mt_loss_curves(
            country,
            history,
            out_train,
            f"{country}_detailed_loss_curves.png",
            show_plots,
        )
        plot_regression_scatter(
            val_df["l3_true"],
            val_df["l3_pred"],
            "L3 Intensity",
            5000,
            out_val,
            f"{country}_l3_regression_scatter.png",
            show_plots
        )
        plot_regression_scatter(
            val_df["l7_true"],
            val_df["l7_pred"],
            "L7 Intensity",
            5000,
            out_val,
            f"{country}_l7_regression_scatter.png",
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
        plot_mt_anomaly_timeseries(
            country,
            val_df,
            out_val,
            f"{country}_mt_anomaly_timeseries.png",
            show_plots
        )
        summarize_mt_validation(
            country,
            val_df,
            out_val,
            f"{country}_summary.json"
        )
    except Exception as e:
        print(f"[ERROR] Failed analyzing {country}: {e}")

    print(f"[OK] Analysis for {country} completed!")


def analyze_all(show_plots: bool) -> None:
    """Runs full analysis pipeline of all country MT models."""
    print(f"\n[INFO] Analysis of all MT models starting...")

    for c in COUNTRIES:
        analyze_country(c, show_plots)
    
    print(f"\n[DONE] Analysis of all MT model trainings and validations completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MT model training & validation performance."
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
        analyze_all(
            show_plots=args.show
        )
    else:
        analyze_country(
            target.upper(), 
            show_plots=args.show
        )
