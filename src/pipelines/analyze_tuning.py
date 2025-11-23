#!/usr/bin/env python3
"""
Pipeline for analyzing Optuna tuning results.

Generates:
- Optuna standard plots
- trial correlation heatmap
- best trial learning curves
- loss curves for all trials
- 3D hyperparam landscape
- CSV + JSON reports
- (optional) multi-country comparison

Outputs:
    plots : results/models/tuned/analysis/<COUNTRY>/optimization_history.png + .html
            results/models/tuned/analysis/<COUNTRY>/parallel_coordinates.png + .html
            results/models/tuned/analysis/<COUNTRY>/param_importance.png + .html
            results/models/tuned/analysis/<COUNTRY>/contour.png + .html
            results/models/tuned/analysis/<COUNTRY>/slice.png + .html
            results/models/tuned/analysis/<COUNTRY>/3d_scatter.png
            results/models/tuned/analysis/<COUNTRY>/best_learning_curve.png
            results/models/tuned/analysis/<COUNTRY>/correlation_heatmap.png
            results/models/tuned/analysis/multi/best_losses.png
    summary : results/models/tuned/analysis/<COUNTRY>/trial_results.csv
              results/models/tuned/analysis/multi/best_losses.json

Usage:
    python -m src.pipelines.analyze_tuning [--multi] [--show] <COUNTRY|all>
"""

import json, optuna
import pandas as pd
from pathlib import Path
from src.data.feature_engineering import COUNTRIES
from src.models.analysis import (
    save_optuna_plots,
    plot_correlation_heatmap,
    plot_loss_curves_all_trials,
    plot_best_trial_learning_curve,
    plot_3d_scatter,
    plot_multi_country_overview,
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
TUNED_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
ANALYSIS_TUNE_DIR = TUNED_DIR / "analysis"
ANALYSIS_TUNE_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               LOAD DATA             ##
#########################################

def load_study(country: str, db_path: Path):
    """Load study from SQLite db."""
    return optuna.load_study(
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
    )


def trial_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Convert study trials into df."""
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        row = {"trial_id": t.number, "value": t.value}
        row.update(t.params)
        rows.append(row)
    return pd.DataFrame(rows)


#########################################
##                 MAIN                ##
#########################################

def analyze_multi_country(countries: list = COUNTRIES, show: bool = False):
    """Compare best validation losses across countries."""
    print(f"\n[INFO] Multi-country analysis...")
    summary = {}
    #countries = ["US","DE","GB","FR","JP","SG","NL","CA","AU","AT","BR","CH","TW","IN","ZA","KR","SE","IT","ES","PL"]
    for c in countries:
        cfg_path = TUNED_DIR / f"{c}_best_params.json"
        study_path = TUNED_DIR / f"{c}_study.db"
        if not cfg_path.exists() or not study_path.exists():
            continue
        study = load_study(c, study_path)
        summary[c] = study.best_value
    out_dir = ANALYSIS_TUNE_DIR / "multi"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_multi_country_overview(summary, out_dir / "best_losses.png", show)
    with open(out_dir / "best_losses.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Multi-country comparison completed!")

def analyze_single(country: str, multi: bool = True, all: bool = False, show: bool = False):
    """Runs full analysis pipeline of a country model tuning."""
    print(f"\n[INFO] Analyzing {country}...")
    
    out_dir = ANALYSIS_TUNE_DIR / country
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = TUNED_DIR / f"{country}_study.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No study DB for {country}")
    study = load_study(country, db_path)
    save_optuna_plots(study, out_dir)

    df = trial_dataframe(study)
    df.to_csv(out_dir / "trial_results.csv", index=False)
    plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png", show)
    plot_3d_scatter(df, out_dir / "3d_scatter.png", show)
    plot_loss_curves_all_trials(
        study,
        country,
        history_dir=TUNED_DIR / "trial_history",
        out_path=out_dir / "losses_all_trials.png",
        show=show
    )
    best_hist_path = TUNED_DIR / f"{country}_best_history.json"
    if best_hist_path.exists():
        with open(best_hist_path, "r") as f:
            best_history = json.load(f)
        plot_best_trial_learning_curve(best_history, out_dir / "best_learning_curve.png", show)
    print(f"[OK] Analysis for {country} completed!")

    if multi and not all:
        analyze_multi_country(show=show)

def analyze_all(multi: bool = True, show_plots: bool = False):
    """Runs full analysis pipeline of all country model tunings."""
    print(f"\n[INFO] Analysis of all models starting...")

    for c in COUNTRIES:
        analyze_single(country=c, multi=False, all=True, show=show_plots)

    if multi:
        analyze_multi_country(show=show_plots)

    print(f"\n[DONE] Analysis of all models completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to analyse US model, or 'all' to evaluate all country models."
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="Perform multi country analysis."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively when generated."
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        exit(1)

    target = args.target

    if target.lower() == "all":
        analyze_all(args.multi, args.show)
    else:
        analyze_single(
            country=target.upper(), 
            multi=args.multi, 
            all=False, 
            show=args.show
        )