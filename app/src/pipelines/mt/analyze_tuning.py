#!/usr/bin/env python3
"""
Analyze Optuna tuning results for one or all countries:
- loads tuning history
- loads training summary with best model
- performs multi-country comparison if chosen
- performs latent space analysis if specified
- generates plots and summary 

Outputs:
    PATH : results/mt_ml/tuned/analysis/<COUNTRY>
    FILES : optimization_history.png | .html, 
            parallel_coordinates.png | .html, 
            param_importance.png | .html, 
            contour.png | .html, 
            slice.png | .html, 
            3d_scatter.png, 
            losses_all_trials, 
            best_learning_curve.png, 
            correlation_heatmap.png,
            loss_component_analysis.png,
            attack_class_weights.png,
            attack_class_balance.png,
            trial_results.csv, 
            <COUNTRY>_latent_space.png, 
            <COUNTRY>_latent_space_pca_coords.csv
    PATH : results/mt_ml/tuned/analysis/_multi
    FILES : best_losses.png, 
            best_weights.png, 
            weight_loss_correlation.png, 
            best_losses.json, 
            best_weights.json,
            best_attack_class_weights.png,
            best_attack_class_weights.json

Usage:
    python -m app.src.pipelines.mt.analyze_tuning [-s] [-M] [-L] [--retune] <COUNTRY|all|none>
"""

import json, optuna, torch, pickle
import pandas as pd
import numpy as np
from pathlib import Path

from app.src.data.feature_engineering import COUNTRIES, load_supervised_feature_matrix
from app.src.data.attack_labelling import ATTACK_LABELS
from app.src.data import timeseries_seq_split
from app.src.ml.training.train_mt import load_multitask_model
from app.src.ml.analysis import (
    save_optuna_plots,
    plot_correlation_heatmap,
    plot_loss_curves_all_trials,
    plot_best_trial_learning_curve,
    plot_3d_scatter,
    plot_multi_loss_overview,
    plot_latent_space,
    plot_mt_loss_component_analysis,
    plot_attack_class_weights,
    plot_attack_class_balance,
    plot_multi_mt_weights_overview, 
    plot_multi_mt_weight_loss_correlation, 
    plot_multi_country_attack_weights
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
TUNED_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tuned"


#########################################
##               LOAD DATA             ##
#########################################

def load_study(country: str, db_path: Path) -> optuna.Study:
    """Load study from SQLite db."""
    return optuna.load_study(
        storage=f"sqlite:///{db_path}",
        study_name=f"mt_tuning_{country}",
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

def plot_latent(
    country: str,
    out_dir: Path, 
    show: bool
) -> None:
    """Plot latent space if flag was not set during tuning."""
    print(f"[INFO] Preparing latent space visualization...")

    model_path = TUNED_DIR / f"{country}_best_model.pt"
    scaler_path = TUNED_DIR / f"{country}_scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")

    model, cfg, model_num_cont, model_cat_dims = load_multitask_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    X_cont, X_cat, _, _, _, num_cont, cat_dims = load_supervised_feature_matrix(country)

    assert num_cont == model_num_cont, "[ERROR] num_cont mismatch between cfg and supervised matrix"
    assert cat_dims == model_cat_dims, "[ERROR] cat_dims mismatch between cfg and supervised matrix"

    Xc = X_cont.values.astype(np.float64)
    Xk = X_cat.values.astype(np.int64)

    (Xc_tr, Xk_tr), _, _ = timeseries_seq_split(
        Xc, Xk, 
        75/100, 
        15/100
    )
    Xc_tr_scald = scaler.transform(Xc_tr).astype(np.float32)
    plot_latent_space(
        country, 
        Xc_tr_scald, 
        Xk_tr,
        model,
        cfg.device,
        1000,
        out_dir,
        f"{country}_latent_space.png",
        show
    )
    print(f"[OK] Latent space plot for {country} generated.")


def multi_analyze(
    countries: list = COUNTRIES,
    tune_phase: str = base,
    show: bool = False
) -> None:
    """Compare best validation losses across countries."""
    print(f"\n[INFO] Multi-country analysis...")
    
    out_dir = TUNED_DIR / "analysis" / "_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    #countries.remove("KR")
    #countries.remove("TW")
    #countries.remove("AT")
    #countries.remove("GB")
    #countries.remove("CH")
    losses_data = {}
    attack_class_weights = {}
    for c in countries:
        cfg_path = TUNED_DIR / f"{c}_best_params.json"
        study_path = TUNED_DIR / f"{c}_study_{tune_phase}.db"

        if not cfg_path.exists() or not study_path.exists():
            print()
            continue
        
        study = load_study(c, study_path)
        losses_data[c] = study.best_value
        if c == "AT":
            attack_class_weights[c] = study.best_trial.user_attrs["attack_class_weights"]
    
    plot_multi_loss_overview(
        losses_data, 
        out_dir, 
        "best_losses.png", 
        show
    )
    with open(out_dir / "best_losses.json", "w") as f:
        json.dump(losses_data, f, indent=2)
    
    plot_multi_country_attack_weights(
        attack_class_weights,
        ATTACK_LABELS,
        out_dir, 
        "best_attack_class_weights.png", 
        show
    )
    with open(out_dir / "best_attack_class_weights.json", "w") as f:
        json.dump(attack_class_weights, f, indent=2)

    weights_data = {}
    for c in countries:
        try:
            model_path = TUNED_DIR / f"{c}_best_model.pt"
            if model_path.exists():
                payload = torch.load(model_path, map_location="cpu", weights_only=True)
                loss_weights = payload.get("additional_info", {}).get("loss_weights", {})
                reg_weights = loss_weights.get("l3", 0.0) + loss_weights.get("l7", 0.0)
                weights_data[c] = {
                    "l3": loss_weights.get("l3", 1.0),
                    "l7": loss_weights.get("l7", 1.0),
                    "class": loss_weights.get("attack", 1.0),
                    "reg": reg_weights,
                    "ratio": loss_weights["attack"] / max(reg_weights, 1e-8)
                }
        except Exception as e:
            print(f"[ERROR] Failed loading weights for {c}:", e)
            continue
    
    plot_multi_mt_weights_overview(
        weights_data, 
        out_dir, 
        "best_weights.png", 
        show
    )
    with open(out_dir / "best_weights.json", "w") as f:
        json.dump(weights_data, f, indent=2)

    if weights_data and losses_data:
        common_countries = set(weights_data.keys()) & set(losses_data.keys())
        if len(common_countries) >= 3:
            plot_multi_mt_weight_loss_correlation(
                weights_data, 
                losses_data, 
                out_dir,
                "weight_loss_correlation.png", 
                show
            )
    
    print(f"[OK] Multi-country comparison completed!")


def analyze_country(
    country: str, 
    multi: bool = True, 
    all: bool = False, 
    latent: bool = False, 
    show: bool = False,
    tune_phase: str = "base"
) -> None:
    """Runs full analysis pipeline of a country MT model tuning."""
    print(f"\n[INFO] Analyzing {country}...")
    
    out_dir = TUNED_DIR / "analysis" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = TUNED_DIR / f"{country}_study_{tune_phase}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No study DB for {country}")
    
    study = load_study(country, db_path)
    if study:
        save_optuna_plots(
            study, 
            out_dir, 
            html_out=True, 
            png_out=False
        )
        plot_mt_loss_component_analysis(
            study,
            country, 
            history_dir = TUNED_DIR / "trial_history",
            folder=out_dir,
            fname="loss_component_analysis.png",
            show=show
        )
        weights = study.best_trial.user_attrs["attack_class_weights"]
        plot_attack_class_weights(
            country,
            np.array(weights),
            ATTACK_LABELS,
            out_dir,
            f"attack_class_weights.png",
            show
        )
        frequ = study.best_trial.user_attrs["attack_class_frequencies"]
        plot_attack_class_balance(
            country,
            np.array(weights),
            np.array(frequ),
            ATTACK_LABELS,
            out_dir,
            f"attack_class_balance.png",
            show
        )

    df = trial_dataframe(study)
    df.to_csv(out_dir / "trial_results.csv", index=False)

    plot_correlation_heatmap(
        df, 
        out_dir, 
        "correlation_heatmap.png", 
        show
    )
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
        plot_best_trial_learning_curve(
            best_history, 
            out_dir, 
            "best_learning_curve.png", 
            show
        )

    if latent:
        plot_latent(country, out_dir, show)
    
    print(f"[OK] Analysis for {country} completed!")

    if multi and not all:
        multi_analyze(tune_phase=tune_phase, show=show)


def analyze_all(
    multi: bool = True, 
    latent: bool = False, 
    show_plots: bool = False,
    tune_phase: str = "base"
) -> None:
    """Runs full analysis pipeline of all country MT model tunings."""
    print(f"\n[INFO] Analysis of all MT models starting...")
    
    for c in COUNTRIES:
        analyze_country(
            country=c, 
            multi=False, 
            all=True,
            latent=latent,
            show=show_plots,
            tune_phase=tune_phase
    )

    if multi:
        multi_analyze(tune_phase=tune_phase, show=show_plots)
        
    print(f"\n[DONE] Analysis of all MT model tunings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze MT model tuning performance."
    )

    parser.add_argument(
        "-r", "--retune",
        default=0,
        type=int,
        help="retune study number [default: 0 = base]"
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
        "target",
        help="<COUNTRY|all|none> e.g. 'US' to analyse US model, or 'all' to evaluate all country models, or 'none' for switching off single-country analysis"
    )

    args = parser.parse_args()

    target = args.target

    tune_phase = "base" if args.retune == 0 else f"retune_{args.retune}"
    
    if target.lower() == "all":
        analyze_all(
            args.multi, 
            args.latent, 
            tune_phase,
            args.show
        )
    elif target.lower() == "none":
        if args.multi:
            multi_analyze(tune_phase=tune_phase, show=args.show)
        else:
            print(f"[INFO] No analysis selected [target=none and multi=False].")
    else:
        analyze_country(
            country=target.upper(), 
            multi=args.multi, 
            all=False,
            latent=args.latent,
            tune_phase=tune_phase,
            show=args.show
        )