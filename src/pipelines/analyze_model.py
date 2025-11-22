#!/usr/bin/env python3
"""
Analyze training and validation performance for one or all countries.
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
    python -m src.pipelines.analyze_model <COUNTRY_CODE|all> --show
"""

from pathlib import Path
from src.data.feature_engineering import COUNTRIES
from src.models.analysis import analyze_country


#########################################
##                PARAMS               ##
#########################################
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]

TRAINED_DIR = PROJECT_ROOT / "results" / "models" / "trained"
VALIDATED_DIR = PROJECT_ROOT / "results" / "models" / "validated"


#########################################
##                 MAIN                ##
#########################################

def analyze_single(country: str, show_plots: bool):
    print(f"\n==============================")
    print(f"   ANALYZE MODEL ({country})")
    print(f"==============================")

    try:
        analyze_country(
            country=country,
            trained_dir=TRAINED_DIR,
            validated_dir=VALIDATED_DIR,
            show_plots=show_plots,
        )
    except Exception as e:
        print(f"[ERROR] Failed analyzing {country}: {e}")


def analyze_all(show_plots: bool):
    for c in COUNTRIES:
        analyze_single(c, show_plots)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model training & validation performance."
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to analyse US model, or 'all' to evaluate all country models."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively when generated."
    )

    args = parser.parse_args()

    target = args.target.lower()

    if target == "all":
        analyze_all(show_plots=args.show)
    else:
        analyze_single(target.upper(), show_plots=args.show)
