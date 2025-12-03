#!/usr/bin/env python3
"""
Hyperparameter tuning for the TabularAutoencoder using Optuna.

Search space:
- latent_dim
- hidden_dims (depth + width)
- dropout
- learning rate
- batch size
- weight decay
- patience
- embedding_dim
- continuous noise_std
- residual_strength
- optimizer (Adam / AdamW)
- lr scheduler type

Outputs:
    study db : results/models/tuned/<COUNTRY>_study.db
    best model : results/models/tuned/<COUNTRY>_best_model.pt
    best hyperparams: results/models/tuned/<COUNTRY>_best_params.json
    best AEConfig : results/models/tuned/<COUNTRY>_best_config.json
    training hist : results/models/tuned/<COUNTRY>_best_history.json
    tuned scaler fitted on cont features: results/models/tuned/<COUNTRY>_scaler.pkl
    categorical dims : results/models/tuned/<COUNTRY>_cat_dims.json
    latent space : results/models/tuned/<COUNTRY>_latent_space_pca_coords.csv
                   results/models/tuned/<COUNTRY>_latent_space.png

Usage:
    python -m src.pipelines.tune_model [-n <int>] [-p <median|halving|hyperband>] [-tr <int>] [-vr <int>] [-L] <COUNTRY|all> [| tee stdout_tune.txt]
"""

from app.src.data.feature_engineering import COUNTRIES
from app.src.models.tune import tune_country


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

    parser = argparse.ArgumentParser(description="Tune AE hyperparameters for single or for all countries.")

    parser.add_argument(
        "-n", "--ntrials",
        type=int, 
        default=40, 
        help="number of Optuna trials [default: 40]"
    )

    parser.add_argument(
        "-p", "--pruner",
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
        help="generate latent space plot after training"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to tune US model, or 'all' to tune all country models"
    )

    args = parser.parse_args()

    target = args.target
    
    if args.pruner not in ["median", "halving", "hyperband"]:
        parser.print_help()
        exit(1)

    if target.lower() == "all":
        tune_all(args.ntrials, args.pruner, args.tr, args.vr, args.latent)
    else:
        tune_country(
            country=target.upper(),
            n_trials=args.ntrials,
            pruner=args.pruner,
            tr=args.tr, 
            vr=args.vr,
            latent=args.latent
        )
