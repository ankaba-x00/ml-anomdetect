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

Outputs:
    study db : results/models/tuned/<COUNTRY>_study.db
    best model : results/models/tuned/<COUNTRY>_best_model.pt
    best hyperparams: results/models/tuned/<COUNTRY>_best_params.json
    best AEConfig : results/models/tuned/<COUNTRY>_best_config.json
    training hist : results/models/tuned/<COUNTRY>_best_history.json
    tuned scaler fitted on cont features: results/models/tuned/<COUNTRY>_scaler.pkl
    categorical dims : results/models/tuned/<COUNTRY>_cat_dims.json

Usage:
    python -m src.pipelines.tune_model <COUNTRY|all> <trail_no> <median|halving|hyperband> [>> stdout_tune.txt]
"""

import argparse
from src.data.feature_engineering import COUNTRIES
from src.models.tune import tune_country


def tune_all(trials: int, pruner: str):
    for c in COUNTRIES:
        try:
            tune_country(c, n_trials=trials, pruner=pruner)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune AE hyperparameters for single or for all countries.")

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to tune US model, or 'all' to tune all country models"
    )

    parser.add_argument(
        "trials",
        nargs="?",
        type=int, 
        default=40, 
        help="number of Optuna trials [default: 40]"
    )

    parser.add_argument(
        "pruner",
        nargs="?",
        type=str, 
        default="median",
        help="<median|halving|hyperband> Optuna pruner strategy [default: median]"
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
        tune_all(args.trials, args.pruner)
    else:
        tune_country(
            country=target.upper(),
            n_trials=args.trials,
            pruner=args.pruner
        )
