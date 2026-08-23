#!/usr/bin/env python3
"""
Analyze model performance after tuning for one or multiple countries:
- loads study, tuning history, best trial
- generates plots and summary
- (optional) performs multi-country comparison

Output:
    PATH : results/ml/tuned/analysis/<MODEL>/<COUNTRY>
    FILES : <RETUNE_NO>_optimization_history.png | .html, 
            <RETUNE_NO>_param_importance.png | .html, 
            <RETUNE_NO>_parallel_coordinates.png | .html, 
            <RETUNE_NO>_slice.png | .html, 
            <RETUNE_NO>_contour.png | .html,
            <RETUNE_NO>_trial_results.csv, 
            <RETUNE_NO>_correlation_heatmap.png, 
            <RETUNE_NO>_3d_scatter.png, 
            <RETUNE_NO>_losses_all_trials.png, 
            <RETUNE_NO>_best_learning_curve.png, 
            <RETUNE_NO>_loss_component_analysis.png,
    Add. FILES for MTAE: <RETUNE_NO>attack_type_weights.png
                         <RETUNE_NO>attack_type_balance.png
    PATH : results/ae_ml/tuned/analysis/<MODEL>/_multi
    FILES : <RETUNE_NO>_best_losses.png, 
            <RETUNE_NO>_best_losses.json, 
            <RETUNE_NO>_best_weights.png, 
            <RETUNE_NO>_best_weights.json
            <RETUNE_NO>_weight_loss_correlation.png, 
    Add. FILES for MTAE: <RETUNE_NO>_best_attack_type_weights.png
                         <RETUNE_NO>_best_attack_type_weights.json

Usage:
    python -m app.src.pipelines.analyze_tuning [-s]  [--multi] [-r <int>] <MODEL> <COUNTRY|all|none>
"""

import json, optuna
import pandas as pd
import numpy as np
from pathlib import Path

from app.src.data import ATTACK_LABELS, ISO_3166_alpha2
from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.analysis import (
    save_optuna_plots,
    plot_correlation_heatmap,
    plot_loss_curves_all_trials,
    plot_best_trial_learning_curve,
    plot_3d_scatter,
    plot_loss_component_analysis,
    plot_multi_loss_overview,
    plot_multi_weights_overview,
    plot_multi_weight_loss_correlation,
    plot_mt_loss_component_analysis,
    plot_attack_type_weights,
    plot_attack_type_balance,
    plot_multi_mt_weights_overview, 
    plot_multi_mt_weight_loss_correlation, 
    plot_multi_attack_type_weights
)


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TUNED_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"


def load_study(country: str, db_path: Path) -> optuna.Study:
    """Loads study from SQLite db."""
    return optuna.load_study(
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
    )

def make_trial_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Converts study trials into df."""
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        row = {"trial_id": t.number, "value": t.value}
        row.update(t.params)
        rows.append(row)
    return pd.DataFrame(rows)

def multi_analyze(
    ae_type: str, 
    retune_no: int = 0,
    show_plots: bool = False
) -> None:
    """Compare best validation losses across countries."""
    print(f"\n[INFO] Multi-country analysis...")

    out_path = TUNED_DIR / f"{ae_type.upper()}" / "analysis" / "_multi"
    out_path.mkdir(parents=True, exist_ok=True)

    losses_data = {}
    if ae_type in ["mtae"]:
        attack_type_weights = {}
    for c in COUNTRIES:
        study_path = TUNED_DIR / f"{ae_type.upper()}" / f"{c}_{retune_no}_study.db"

        if not study_path.exists():
            print(f"[ERROR] No study data found at {study_path}. Skipping {c}...")
            continue
        
        study = load_study(c, study_path)
        losses_data[c] = study.best_value
        if ae_type in ["mtae"]:
            attack_type_weights[c] = study.best_trial.user_attrs["attack_type_weights"]

    plot_multi_loss_overview(
        losses_data, 
        out_path, 
        f"{retune_no}_best_losses.png", 
        show_plots
    )

    loss_fname = f"{retune_no}_best_losses.json" 
    with open(out_path / loss_fname, "w") as f:
        json.dump(losses_data, f, indent=2)
    print(f"[OK] Saved to {loss_fname}")

    if ae_type in ["mtae"]:
        plot_multi_attack_type_weights(
            attack_type_weights,
            ATTACK_LABELS,
            out_path, 
            f"{retune_no}_best_attack_type_weights.png", 
            show_plots
        )

        atf_fname = f"{retune_no}_best_attack_type_weights.json"
        with open(out_path / atf_fname, "w") as f:
            json.dump(attack_type_weights, f, indent=2)
        print(f"[OK] Saved to {atf_fname}")

    weights_data = {}
    for c in COUNTRIES:
        try:
            cfg_path = TUNED_DIR / f"{ae_type.upper()}" / f"{c}_{retune_no}_best_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg_dict = json.load(f)
                weights_data[c] = {
                    "cont_w": cfg_dict["cont_w"],
                    "cat_w": cfg_dict["cat_w"],
                    "ratio_ck": cfg_dict["cat_w"] / cfg_dict["cont_w"]
                }
                if ae_type in ["mtae"]:
                    weights_data[c] |= {
                        "l3_w": cfg_dict["lambda_l3"],
                        "l7_w": cfg_dict["lambda_l7"],
                        "at_w": cfg_dict["lambda_at"],
                        "reg": (cfg_dict["lambda_l3"] + cfg_dict["lambda_l7"]) / 2,
                        "cls": cfg_dict["lambda_at"],
                        "ratio_mt": cfg_dict["lambda_at"] / max((cfg_dict["lambda_l3"] + cfg_dict["lambda_l7"]) / 2, 1e-8)
                    }
        except Exception as e:
            print(f"[ERROR] Failed loading weights for {c}:", e)
            continue
    if ae_type in ["ae", "vae"]:
        plot_multi_weights_overview(
            weights_data, 
            out_path, 
            f"{retune_no}_best_weights.png", 
            show_plots
        )

        weights_fname = f"{retune_no}_best_weights.json"
        with open(out_path / weights_fname, "w") as f:
            json.dump(weights_data, f, indent=2)
        print(f"[OK] Saved to {weights_fname}")
    else:
        plot_multi_mt_weights_overview(
            weights_data, 
            out_path, 
            f"{retune_no}_best_weights.png", 
            show_plots
        )

        aw_fname = f"{retune_no}_best_weights.json"
        with open(out_path / aw_fname, "w") as f:
            json.dump(weights_data, f, indent=2)
        print(f"[OK] Saved to {aw_fname}")

    if weights_data and losses_data:
        common_countries = set(weights_data.keys()) & set(losses_data.keys())
        if len(common_countries) >= 3:
            if ae_type in ["ae", "vae"]:
                plot_multi_weight_loss_correlation(
                    weights_data, 
                    losses_data, 
                    out_path,
                    f"{retune_no}_weight_loss_correlation.png", 
                    show_plots
                )
            else:
                plot_multi_mt_weight_loss_correlation(
                    weights_data, 
                    losses_data, 
                    out_path,
                    f"{retune_no}_weight_loss_correlation.png", 
                    show_plots
                )

    print(f"[OK] Multi-country comparison completed!")

def analyze_tuning(
    ae_type: str, 
    country: str, 
    multi: bool, 
    all: bool, 
    retune_no: int,
    show_plots: bool
) -> None:
    """Runs full analysis pipeline for autoencoder after tuning."""
    print(f"\n[INFO] Analyzing {country}...")
    
    # --------------------
    # Load data
    # --------------------
    out_path = TUNED_DIR / f"{ae_type.upper()}" / "analysis" / country
    out_path.mkdir(parents=True, exist_ok=True)

    db_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_{retune_no}_study.db"
    if not db_path.exists():
        raise FileNotFoundError(f"[ERROR] Study DB not found: {db_path}")
    
    study = load_study(country, db_path)
    save_optuna_plots(
        study, 
        out_path,
        retune_no,
        html_out=True, 
        png_out=False
    )

    df = make_trial_dataframe(study)
    result_fname = f"{retune_no}_trial_results.csv"
    df.to_csv(out_path / result_fname, index=False)
    print(f"[OK] Saved to {result_fname}")
    
    plot_correlation_heatmap(
        df, 
        out_path, 
        f"{retune_no}_correlation_heatmap.png", 
        show_plots
    )
    plot_3d_scatter(df, out_path, f"{retune_no}_3d_scatter.png", show_plots)
    plot_loss_curves_all_trials(
        study,
        country,
        TUNED_DIR / f"{ae_type.upper()}" / "trial_history",
        out_path,
        f"{retune_no}_losses_all_trials.png",
        show_plots
    )

    best_hist_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_{retune_no}_best_history.json"
    if best_hist_path.exists():
        with open(best_hist_path, "r") as f:
            best_history = json.load(f)
        plot_best_trial_learning_curve(
            best_history, 
            out_path, 
            f"{retune_no}_best_learning_curve.png", 
            show_plots
        )

    if study:
        trial_hist_path = TUNED_DIR / f"{ae_type.upper()}" / "trial_history"
        if ae_type in ["ae", "vae"]:
            plot_loss_component_analysis(
                ae_type,
                study,
                country, 
                trial_hist_path,
                out_path,
                f"{retune_no}_loss_component_analysis.png",
                show_plots
            )
        else:
            plot_mt_loss_component_analysis(
                study,
                country, 
                trial_hist_path,
                out_path,
                f"{retune_no}_loss_component_analysis.png",
                show_plots
            )
            
            type_weights = study.best_trial.user_attrs["attack_type_weights"]
            plot_attack_type_weights(
                country,
                np.array(type_weights),
                ATTACK_LABELS,
                out_path,
                f"{retune_no}_attack_type_weights.png",
                show_plots
            )

            train_type_counts = study.best_trial.user_attrs["train_at_counts"]
            val_type_counts = study.best_trial.user_attrs["val_at_counts"]
            plot_attack_type_balance(
                country,
                np.array(type_weights),
                np.array(train_type_counts),
                np.array(val_type_counts),
                ATTACK_LABELS,
                out_path,
                f"{retune_no}_attack_type_balance.png",
                show_plots
            )
    
    print(f"[DONE] Analysis for {country}")

    if multi and not all:
        multi_analyze(
            ae_type, 
            retune_no, 
            show_plots
        )

def analyze_all(
    ae_type: str, 
    multi: bool, 
    retune_no: int,
    show_plots: bool
) -> None:
    for c in COUNTRIES:
        try:
            analyze_tuning(
                ae_type, 
                c, 
                False, 
                True,
                retune_no,
                show_plots,
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")

    if multi:
        multi_analyze(
            ae_type, 
            retune_no,
            show_plots
        )
        
    print(f"\n[DONE] All analysis completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze model tuning performance"
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show plots interactively when generated"
    )

    parser.add_argument(
        "-r", "--retune",
        default=0,
        type=int,
        help="retune number [default: 0]"
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="perform multi country analysis"
    )

    parser.add_argument(
        "model",
       help="<ae|vae|mtae> model to train"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all|none> e.g. 'US' or 'all' for all countries defined in config/models.yaml or 'none' for multi-country analysis"
    )

    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae", "mtae"]:
        parser.print_help()
        exit(1)

    target = args.target.upper()
    if target not in ISO_3166_alpha2 and target.lower() in ["all", "none"]:
        parser.print_help()
        print(f"[ERROR] Invalid target: {target}")
        exit(1)
    
    if target.lower() == "all":
        analyze_all(
            ae_type, 
            args.multi, 
            args.retune,
            args.show
        )
    elif target.lower() == "none":
        multi_analyze(
            ae_type,
            args.retune,
            args.show)
    else:
        analyze_tuning(
            ae_type,
            target, 
            args.multi, 
            False,
            args.retune,
            args.show
        )