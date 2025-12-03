#!/usr/bin/env python3
"""
Analyze Optuna tuning resultse for one or all countries:
- loads tuning history
- loads training error with best model
- performs multi-country comparison if chosen
- generates plots (Optuna standard plots as html and png, trial correlation heatmap, best trial learning curves, loss curves for all trials, 3D hyperparam landscape, csv and json reports, multi-country comparison)

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
            results/models/tuned/analysis/<COUNTRY_CODE>_latent_space_pca_coords.csv
            results/models/tuned/analysis/<COUNTRY_CODE>_latent_space.png
    summary : results/models/tuned/analysis/<COUNTRY>/trial_results.csv
              results/models/tuned/analysis/multi/best_losses.json

Usage:
    python -m src.pipelines.analyze_tuning [-s] [-M] <COUNTRY|all|none>
"""

import json, optuna, torch
import pandas as pd
from pathlib import Path
from app.src.data.feature_engineering import COUNTRIES
from app.src.models.analysis import (
    save_optuna_plots,
    plot_correlation_heatmap,
    plot_loss_curves_all_trials,
    plot_best_trial_learning_curve,
    plot_3d_scatter,
    plot_loss_component_analysis,
    plot_multi_loss_overview,
    plot_multi_weights_overview,
    plot_multi_weight_loss_correlation
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
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

def multi_analyze(countries: list = COUNTRIES, show: bool = False):
    """Compare best validation losses across countries."""
    print(f"\n[INFO] Multi-country analysis...")
    out_dir = ANALYSIS_TUNE_DIR / "_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    #countries.remove("KR")
    #countries.remove("TW")
    #countries.remove("AT")
    #countries.remove("GB")
    #countries.remove("CH")
    losses_data = {}
    for c in countries:
        cfg_path = TUNED_DIR / f"{c}_best_params.json"
        study_path = TUNED_DIR / f"{c}_study.db"
        if not cfg_path.exists() or not study_path.exists():
            continue
        study = load_study(c, study_path)
        losses_data[c] = study.best_value
    plot_multi_loss_overview(losses_data, out_dir, "best_losses.png", show)
    with open(out_dir / "best_losses.json", "w") as f:
        json.dump(losses_data, f, indent=2)

    weights_data = {}
    for c in countries:
        try:
            model_path = TUNED_DIR / f"{c}_best_model.pt"
            if model_path.exists():
                payload = torch.load(model_path, map_location="cpu")
                loss_weights = payload.get("additional_info", {}).get("loss_weights", {})
                weights_data[c] = {
                    "cont_weight": loss_weights.get("cont_weight", 1.0),
                    "cat_weight": loss_weights.get("cat_weight", 1.0),
                    "ratio": loss_weights.get("cat_weight", 1.0) / max(loss_weights.get("cont_weight", 1.0), 1e-8)
                }
        except Exception:
            continue
    plot_multi_weights_overview(weights_data, out_dir, "best_weights.png", show)
    with open(out_dir / "best_weights.json", "w") as f:
        json.dump(weights_data, f, indent=2)

    if weights_data and losses_data:
        common_countries = set(weights_data.keys()) & set(losses_data.keys())
        if len(common_countries) >= 3:
            plot_multi_weight_loss_correlation(
                    weights_data, 
                    losses_data, 
                    out_dir,
                    "weight_loss_correlation.png", 
                    show
                )
    print(f"[OK] Multi-country comparison completed!")

def analyze_country(country: str, multi: bool = True, all: bool = False, show: bool = False):
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
    plot_correlation_heatmap(df, out_dir, "correlation_heatmap.png", show)
    plot_3d_scatter(df, out_dir, "3d_scatter.png", show)
    plot_loss_curves_all_trials(
        study,
        country,
        history_dir=TUNED_DIR / "trial_history",
        folder=out_dir,
        fname="losses_all_trials.png",
        show=show
    )
    best_hist_path = TUNED_DIR / f"{country}_best_history.json"
    if best_hist_path.exists():
        with open(best_hist_path, "r") as f:
            best_history = json.load(f)
        plot_best_trial_learning_curve(best_history, out_dir, "best_learning_curve.png", show)
    if study:
        plot_loss_component_analysis(
            study,
            country, 
            history_dir = TUNED_DIR / "trial_history",
            folder=out_dir,
            fname="loss_component_analysis.png",
            show=show
        )

    print(f"[OK] Analysis for {country} completed!")

    if multi and not all:
        multi_analyze(show=show)

def analyze_all(multi: bool = True, show_plots: bool = False):
    """Runs full analysis pipeline of all country model tunings."""
    print(f"\n[INFO] Analysis of all models starting...")

    for c in COUNTRIES:
        analyze_country(country=c, multi=False, all=True, show=show_plots)

    if multi:
        multi_analyze(show=show_plots)
    print(f"\n[DONE] Analysis of all model tunings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model tuning performance."
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show plots interactively when generated"
    )

    parser.add_argument(
        "-M", "--multi",
        action="store_true",
        help="perform multi country analysis"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all|none> e.g. 'US' to analyse US model, or 'all' to evaluate all country models, or 'none' for switching off single-country analysis"
    )

    args = parser.parse_args()

    target = args.target

    if target.lower() == "all":
        analyze_all(args.multi, args.show)
    elif target.lower() == "none":
        multi_analyze(show=args.show)
    else:
        analyze_country(
            country=target.upper(), 
            multi=args.multi, 
            all=False, 
            show=args.show
        )