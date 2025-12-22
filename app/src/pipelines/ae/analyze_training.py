#!/usr/bin/env python3
"""
Analyze training and validation performance for one or all countries:
- loads training history
- loads validation errors
- generates plots and summary

Outputs:
    PATH without --tuned: results/ae_ml/trained/<MODEL>/analysis/<COUNTRY>
    PATH with --tuned: results/ae_ml/tuned/<MODEL>/analysis/<COUNTRY>
    FILES : <COUNTRY>_loss_curve.png, 
            <COUNTRY>_detailed_loss_curves.png, 
            <COUNTRY>_lr_schedule.png
    PATH without --tuned: results/ae_ml/validated/trained_model/<MODEL>/analysis/<COUNTRY>
    PATH with --tuned: results/ae_ml/validated/tuned_model/<MODEL>/analysis/<COUNTRY>
    FILES : <COUNTRY>_error_hist.png, 
            <COUNTRY>_error_timeseries.png, 
            <COUNTRY>_summary.json

Usage:
    python -m app.src.pipelines.ae.analyze_training [-s] [--tuned] <MODEL> <COUNTRY|all>
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis import (
    plot_training_curves, 
    plot_detailed_loss_curves, 
    plot_error_histogram, 
    plot_error_timeseries, 
    summarize_validation
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
TRAINED_DIR = PROJECT_ROOT / "results" / "ae_ml" / "trained"
TUNED_DIR = PROJECT_ROOT / "results" / "ae_ml" / "tuned"
VALIDATED_DIR = PROJECT_ROOT / "results" / "ae_ml" / "validated"
VAL_TRAIN_DIR = VALIDATED_DIR / "trained_model"
VAL_TUNE_DIR = VALIDATED_DIR / "tuned_model"

#########################################
##               LOAD DATA             ##
#########################################

def load_training_history(
    path: Path
) -> dict:
    """Load training history for country."""
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Training history not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_validation_errors(
    ae_type:str, 
    country: str, 
    validated_dir: Path
) -> pd.DataFrame:
    """Load validation CSV."""
    path = Path(validated_dir) / f"{ae_type.upper()}" / f"{country}_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Validation CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["ts"])


#########################################
##                 MAIN                ##
#########################################

def analyze_country(
    ae_type: str, 
    country: str, 
    tuned: bool,
    show_plots: bool
) -> None:
    """Runs full analysis pipeline of a country model."""
    print(f"[INFO] Analyzing {country}...")
    
    if not tuned:
        in_val = VAL_TRAIN_DIR
        in_train = TRAINED_DIR / f"{ae_type.upper()}" / f"{country}_training_history.json"
        out_train = TRAINED_DIR / f"{ae_type.upper()}" / "analysis" / country
    else:
        in_val = VAL_TUNE_DIR
        in_train = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_best_history.json"
        out_train = TUNED_DIR / f"{ae_type.upper()}" / "analysis" / country
    out_train.mkdir(parents=True, exist_ok=True)
    out_val = in_val / f"{ae_type.upper()}" / "analysis" / country
    out_val.mkdir(parents=True, exist_ok=True)

    try:
        history = load_training_history(in_train)
        val_df = load_validation_errors(ae_type, country, in_val)

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


def analyze_all(
    ae_type: str, 
    tuned: bool,
    show_plots: bool
) -> None:
    """Runs full analysis pipeline of all country models."""
    print(f"\n[INFO] Analysis of all models starting...")
    
    for c in COUNTRIES:
        analyze_country(ae_type, c, tuned, show_plots)
    
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
        "--tuned",
        action="store_true",
        help="use model after tuning stage"
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
        analyze_all(
            ae_type,
            args.tuned,
            show_plots=args.show
        )
    else:
        analyze_country(
            ae_type, 
            target.upper(), 
            args.tuned,
            show_plots=args.show
        )
