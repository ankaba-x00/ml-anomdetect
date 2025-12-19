#!/usr/bin/env python3
"""
Hyperparameter tuning for TrafficAttackPredictor using Optuna.

Search space:
- hidden_dims (depth + width)
- dropout
- learning rate
- weight decay
- batch size
- activation
- loss weights

Outputs:
    PATH : results/mt_ml/tuned/<MODEL>
    FILES: <COUNTRY>_study.db, <COUNTRY>_best_model.pt, <COUNTRY>_best_params.json, <COUNTRY>_best_config.json, <COUNTRY>_best_history.json, <COUNTRY>_scaler.pkl, <COUNTRY>_latent_space_pca_coords.csv, <COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.mt.tune_model [-N <int>] [-P <median|halving|hyperband>] [-tr <int>] [-vr <int>] [-L] <COUNTRY|all> [| tee stdout_tune.txt]
"""

from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.tuning.tune_mt import tune_country


#########################################
##                 RUN                 ##
#########################################

def tune_all(trials: int, pruner: str, tr: int, vr: int, latent: bool):
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

    parser = argparse.ArgumentParser(description="Tune MT hyperparameters for single or for all countries.")

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