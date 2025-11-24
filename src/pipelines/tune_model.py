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

Usage:
    python -m src.pipelines.tune_model [-n <int>] [-p <median|halving|hyperband>] <COUNTRY|all> [| tee stdout_tune.txt]
"""

import argparse
from src.data.feature_engineering import COUNTRIES
from src.models.tune import tune_country


#########################################
##                 RUN                 ##
#########################################

def tune_all(trials: int, pruner: str):
    for c in COUNTRIES:
        try:
            tune_country(c, n_trials=trials, pruner=pruner)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model tunings completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune AE hyperparameters for single or for all countries.")

    parser.add_argument(
        "-n", "--ntrials",
        nargs="?",
        type=int, 
        default=40, 
        help="number of Optuna trials [default: 40]"
    )

    parser.add_argument(
        "-p", "--pruner",
        nargs="?",
        type=str, 
        default="median",
        help="<median|halving|hyperband> Optuna pruner strategy [default: median]"
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to tune US model, or 'all' to tune all country models"
    )

    args = parser.parse_args()

    target = args.target

    if not target:
        parser.print_help()
        exit(1)
    
    if args.pruner not in ["median", "halving", "hyperband"]:
        parser.print_help()
        exit(1)

    if target.lower() == "all":
        tune_all(args.ntrials, args.pruner)
    else:
        tune_country(
            country=target.upper(),
            n_trials=args.ntrials,
            pruner=args.pruner
        )
