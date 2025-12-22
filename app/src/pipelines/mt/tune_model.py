#!/usr/bin/env python3
"""
Hyperparameter tuning for TrafficAttackPredictor using Optuna.

Search space:
- hidden_dims (depth + width)
- latent_dim
- head_hidden_dim
- dropout
- learning rate
- weight decay
- batch size
- activation
- loss weights
- focal_gamma 

Outputs:
    PATH : results/mt_ml/tuned/<MODEL>
    FILES: <COUNTRY>_study.db, 
           <COUNTRY>_best_model.pt, 
           <COUNTRY>_best_params.json, 
           <COUNTRY>_best_config.json, 
           <COUNTRY>_best_history.json, 
           <COUNTRY>_scaler.pkl, 
           analysis/<COUNTRY>_latent_space_pca_coords.csv, 
           analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.mt.tune_model [-N <int>] [-P <median|halving|hyperband>] [-tr <int>] [-vr <int>] [-L] <COUNTRY|all>
"""

import json, pickle, torch, optuna
from optuna.pruners import (
    MedianPruner, 
    SuccessiveHalvingPruner, 
    HyperbandPruner
)
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from dataclasses import asdict

from app.src.data.feature_engineering import COUNTRIES, load_supervised_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.mte import MTEConfig
from app.src.ml.tuning.tune_mt import set_global_seeds, objective
from app.src.ml.training.train_mt import train_multitask_model, save_multitask_model
from app.src.ml.analysis import plot_latent_space


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def tune_country(
    country: str,
    n_trials: int = 40,
    pruner: str = "median",
    tr: int = 75,
    vr: int = 15,
    latent: bool = False
) -> None:
    print(f"\n==============================")
    print(f" OPTUNA MT TUNING FOR {country}")
    print(f"==============================\n")
    
    set_global_seeds(42)

    pr = {
        "median": MedianPruner(n_startup_trials=5),
        "halving": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
    }.get(pruner)
    if pr is None:
        raise ValueError(f"Unknown pruner: {pruner}")

    db_path = OUT_DIR / f"{country}_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"mt_tuning_{country}",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, country, tr, vr, OUT_DIR),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    print("\nBest Trial:")
    print(study.best_trial)
    print("\nBest Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # ------------------------------------
    # Retrain best model fully
    # ------------------------------------
    X_cont_df, X_cat_df, y3_s, y7_s, ya_s, num_cont, cat_dims = load_supervised_feature_matrix(country)
    Xc = X_cont_df.values.astype(np.float64)
    Xk = X_cat_df.values.astype(np.int64)
    y3 = y3_s.values.astype(np.float32)
    y7 = y7_s.values.astype(np.float32)
    ya = ya_s.values.astype(np.int64)

    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk,
        tr/100,
        vr/100
    )
    y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
    y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
    ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    p = study.best_trial.params
    depth = p["depth"]
    hidden_dims = [p[f"h{i}"] for i in range(depth)]
    lambda_l3 = p.get("lambda_l3", 1.0)
    lambda_l7 = p.get("lambda_l7", 1.0)
    lambda_attack = p.get("lambda_attack", 3.0)
    loss_weights = {
        "l3": lambda_l3,
        "l7": lambda_l7,
        "attack": lambda_attack,
    }

    best_cfg = MTEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        n_attack_types=8,
        hidden_dims=tuple(hidden_dims),
        #latent_dim=p["latent_dim"],
        dropout=p["dropout"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        batch_size=p["batch_size"],
        num_epochs=90,
        warmup_epochs=5,
        patience=p["patience"],
        activation=p["activation"],
        lambda_l3=lambda_l3,
        lambda_l7=lambda_l7,
        lambda_attack=lambda_attack,
        use_focal_loss=p["use_focal_loss"],
        focal_gamma=p["focal_gamma"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    best_model, best_history = train_multitask_model(
        Xc_train_scald, Xk_train, y3_tr, y7_tr, ya_tr,
        Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
        best_cfg,
        loss_weights
    )

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = OUT_DIR / f"{country}_best_model.pt"
    save_multitask_model(
        model=best_model, 
        config=best_cfg,
        cat_dims=cat_dims,
        num_cont=num_cont,
        path=out_model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "attack_class_weights": (
                best_model.attack_class_weights.cpu().tolist() if best_model.attack_class_weights is not None else None
            ),
            "total_samples": len(Xc_train_scald),
        }
    )

    with open(OUT_DIR / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(OUT_DIR / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(OUT_DIR / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(OUT_DIR / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[OK] Finished tuning for {country}")

    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = OUT_DIR / "analysis" / country
        out_path.mkdir(parents=True, exist_ok=True)
        plot_latent_space(
            country, 
            Xc_train_scald, 
            Xk_train,
            best_model,
            best_cfg.device,
            1000,
            out_path,
            f"{country}_best_latent_space.png"
        )
        
    print(f"[DONE] Saved best model to {out_model_path}")


def tune_all(
    trials: int, 
    pruner: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
    for c in COUNTRIES:
        try:
            tune_country(
                c, 
                n_trials=trials, 
                pruner=pruner, 
                tr=tr, 
                vr=vr,
                latent=latent
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")

    print(f"\n[DONE] All model tunings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune MT hyperparameters for single or for all countries."
    )

    parser.add_argument(
        "-N", "--ntrials",
        type=int, 
        default=40, 
        help="number of Optuna trials [default: 40]"
    )

    parser.add_argument(
        "-P", "--pruner",
        type=str, 
        default="median",
        help="<median|halving|hyperband> Optuna pruner strategy [default: median]"
    )

    parser.add_argument(
        "-tr",
        type=int,
        default=75,
        help="dataset ratio for training in %% [default: 75%%]"
    )

    parser.add_argument(
        "-vr",
        type=int,
        default=15,
        help="dataset ratio for validation in %% [default: 15%%]"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after tuning"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to tune US model, or 'all' to tune all country models"
    )

    args = parser.parse_args()

    target = args.target
    
    if args.pruner.lower() not in ["median", "halving", "hyperband"]:
        parser.print_help()
        exit(1)

    if target.lower() == "all":
        tune_all(
            args.ntrials, 
            args.pruner.lower(), 
            args.tr, 
            args.vr, 
            args.latent
        )
    else:
        tune_country(
            country=target.upper(),
            n_trials=args.ntrials,
            pruner=args.pruner.lower(),
            tr=args.tr, 
            vr=args.vr,
            latent=args.latent
        )