#!/usr/bin/env python3
"""
Analyze Optuna tuning resultse for one or all countries:
- loads tuning history
- loads training error with best model
- performs multi-country comparison if chosen
- generates plots (Optuna standard plots as html and png, trial correlation heatmap, best trial learning curves, loss curves for all trials, 3D hyperparam landscape, csv and json reports, multi-country comparison)

Outputs:
    PATH : results/ml/tuned/analysis/<MODEL>/<COUNTRY>
    FILES : optimization_history.png + .html, parallel_coordinates.png + .html, param_importance.png + .html, contour.png + .html, slice.png + .html, 3d_scatter.png, losses_all_trials, best_learning_curve.png, correlation_heatmap.png, loss_component_analysis.png
    PATH : results/ml/tuned/analysis/<MODEL>/multi
    FILES : best_losses.png, best_weights.png, weight_loss_correlation.png, trial_results.csv, best_losses.json, best_weights.json

Usage:
    python -m app.src.pipelines.analyze_tuning [-s] [-M] <MODEL> <COUNTRY|all|none>
"""

import json, optuna, torch, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from app.src.data.feature_engineering import COUNTRIES, load_feature_matrix
from app.src.data import timeseries_seq_split
from app.src.ml.training.train import load_autoencoder
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.analysis.analysis import (
    save_optuna_plots,
    plot_correlation_heatmap,
    plot_loss_curves_all_trials,
    plot_best_trial_learning_curve,
    plot_3d_scatter,
    plot_loss_component_analysis,
    plot_multi_loss_overview,
    plot_multi_weights_overview,
    plot_multi_weight_loss_correlation,
    plot_latent_space
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TUNED_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"

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

def plot_latent(ae_type: str, country: str, out_dir: Path, show: bool):
    """Plot latent space if flag was not set during tuning."""
    print(f"[INFO] Preparing latent space visualization...")

    model_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_best_model.pt"
    scaler_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")

    model, cfg = load_autoencoder(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    X_cont, X_cat, _, _ = load_feature_matrix(country)
    Xc_np = X_cont.values.astype(np.float64)
    Xk_np = X_cat.values.astype(np.int64)
    (train_cont, train_cat), _, _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=75/100,
        val_ratio=15/100,
    )
    train_cont_scald = scaler.transform(train_cont).astype(np.float32)
    
    plot_latent_space(
        country, 
        train_cont_scald, 
        train_cat,
        model,
        cfg.device,
        1000,
        out_dir,
        f"{country}_latent_space.png",
        show
    )
    print(f"[OK] Latent space plot for {country} generated.")
    
def multi_analyze(ae_type: str, countries: list = COUNTRIES, show: bool = False):
    """Compare best validation losses across countries."""
    print(f"\n[INFO] Multi-country analysis...")
    out_dir = TUNED_DIR / f"{ae_type.upper()}" / "analysis" / "_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    #countries.remove("KR")
    #countries.remove("TW")
    #countries.remove("AT")
    #countries.remove("GB")
    #countries.remove("CH")
    losses_data = {}
    for c in countries:
        cfg_path = TUNED_DIR / f"{ae_type.upper()}" / f"{c}_best_params.json"
        study_path = TUNED_DIR / f"{ae_type.upper()}" / f"{c}_study.db"
        if not cfg_path.exists() or not study_path.exists():
            print()
            continue
        study = load_study(c, study_path)
        losses_data[c] = study.best_value
    plot_multi_loss_overview(losses_data, out_dir, "best_losses.png", show)
    with open(out_dir / "best_losses.json", "w") as f:
        json.dump(losses_data, f, indent=2)

    weights_data = {}
    for c in countries:
        try:
            model_path = TUNED_DIR / f"{ae_type.upper()}" / f"{c}_best_model.pt"
            if model_path.exists():
                payload = torch.load(model_path, map_location="cpu")
                loss_weights = payload.get("additional_info", {}).get("loss_weights", {})
                weights_data[c] = {
                    "cont_weight": loss_weights.get("cont_weight", 1.0),
                    "cat_weight": loss_weights.get("cat_weight", 0.0),
                    "ratio": loss_weights["cat_weight"] / max(loss_weights["cont_weight"], 1e-8)
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

def analyze_country(ae_type: str, country: str, multi: bool = True, all: bool = False, latent: bool = False, show: bool = False):
    """Runs full analysis pipeline of a country model tuning."""
    print(f"\n[INFO] Analyzing {country}...")
    
    out_dir = TUNED_DIR / f"{ae_type.upper()}" / "analysis" / country
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_study.db"
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
        history_dir=TUNED_DIR / f"{ae_type.upper()}" / "trial_history",
        folder=out_dir,
        fname="losses_all_trials.png",
        show=show
    )
    best_hist_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_best_history.json"
    if best_hist_path.exists():
        with open(best_hist_path, "r") as f:
            best_history = json.load(f)
        plot_best_trial_learning_curve(best_history, out_dir, "best_learning_curve.png", show)
    if study:
        plot_loss_component_analysis(
            ae_type,
            study,
            country, 
            history_dir = TUNED_DIR / f"{ae_type.upper()}" / "trial_history",
            folder=out_dir,
            fname="loss_component_analysis.png",
            show=show
        )

    if latent:
        print("here")
        plot_latent(ae_type, country, out_dir, show)
    
    print(f"[OK] Analysis for {country} completed!")

    if multi and not all:
        multi_analyze(show=show)

def analyze_all(
        ae_type: str, 
        multi: bool = True, 
        latent: bool = False, 
        show_plots: bool = False
    ):
    """Runs full analysis pipeline of all country model tunings."""
    print(f"\n[INFO] Analysis of all models starting...")

    for c in COUNTRIES:
        analyze_country(
            ae_type=ae_type, 
            country=c, 
            multi=False, 
            all=True,
            latent=latent,
            show=show_plots
    )

    if multi:
        multi_analyze(ae_type=ae_type, show=show_plots)
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
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot for best model after tuning"
    )

    parser.add_argument(
        "model",
        help="model to train: ae, vae"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all|none> e.g. 'US' to analyse US model, or 'all' to evaluate all country models, or 'none' for switching off single-country analysis"
    )

    args = parser.parse_args()

    target = args.target

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)
    
    if target.lower() == "all":
        analyze_all(
            ae_type, 
            args.multi, 
            args.latent, 
            args.show
        )
    elif target.lower() == "none":
        if args.multi:
            multi_analyze(ae_type=ae_type, show=args.show)
        else:
            print(f"[INFO] No analysis selected [target=none and multi=False].")
    else:
        analyze_country(
            ae_type=ae_type,
            country=target.upper(), 
            multi=args.multi, 
            all=False,
            latent=args.latent,
            show=args.show
        )