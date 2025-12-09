#!/usr/bin/env python3
"""
Hyperparameter tuning for the TabularAE using Optuna.

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
    PATH : results/ml/tuned/<MODEL>
    FILES: <COUNTRY>_study.db, <COUNTRY>_best_model.pt, <COUNTRY>_best_params.json, <COUNTRY>_best_config.json, <COUNTRY>_best_history.json, <COUNTRY>_scaler.pkl, <COUNTRY>_cat_dims.json, <COUNTRY>_latent_space_pca_coords.csv, <COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.tune_model [-N <int>] [-P <median|halving|hyperband>] [-M <elbo|recon|mixed>] [-tr <int>] [-vr <int>] [-L] <MODEL> <COUNTRY|all> [| tee stdout_tune.txt]
"""

from app.src.data.feature_engineering import COUNTRIES
from app.src.ml.tuning.tune import tune_country


#########################################
##                 RUN                 ##
#########################################

def tune_all(ae_type: str, trials: int, pruner: str, metric: str, tr: int, vr: int, latent: bool):
    for c in COUNTRIES:
        try:
            tune_country(
                ae_type,
                c, 
                n_trials=trials, 
                pruner=pruner, 
                metric=metric,
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
        "-M", "--metric",
        type=str, 
        default="elbo",
        help="<elbo|recon|mixed> Optuna tuning metric [default: elbo]"
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
        "model",
        help="model to train: ae, vae"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to tune US model, or 'all' to tune all country models"
    )

    args = parser.parse_args()

    target = args.target
    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)
    
    if args.pruner.lower() not in ["median", "halving", "hyperband"]:
        parser.print_help()
        exit(1)

    if args.metric.lower() not in ["elbo", "recon", "mixed"]:
        parser.print_help()
        exit(1)

    if target.lower() == "all":
        tune_all(ae_type, args.ntrials, args.pruner.lower(), args.metric.lower(), args.tr, args.vr, args.latent)
    else:
        tune_country(
            ae_type=ae_type,
            country=target.upper(),
            n_trials=args.ntrials,
            pruner=args.pruner.lower(),
            metric=args.metric.lower(),
            tr=args.tr, 
            vr=args.vr,
            latent=args.latent
        )
