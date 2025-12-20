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
- activation
- loss weights
- beta (for VAE)

Outputs:
    PATH : results/ae_ml/tuned/<MODEL>
    FILES: <COUNTRY>_study.db, 
           <COUNTRY>_best_model.pt, 
           <COUNTRY>_best_params.json, 
           <COUNTRY>_best_config.json, 
           <COUNTRY>_best_history.json, 
           <COUNTRY>_scaler.pkl, 
           <COUNTRY>_cat_dims.json, 
           <COUNTRY>_latent_space_pca_coords.csv, 
           <COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.ae.tune_model [-N <int>] [-P <median|halving|hyperband>] [-M <elbo|recon|mixed>] [-tr <int>] [-vr <int>] [-L] <MODEL> <COUNTRY|all>
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

from app.src.data.feature_engineering import COUNTRIES, load_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.tuning.tune import set_global_seeds, objective
from app.src.ml.training.train import train_autoencoder, save_autoencoder
from app.src.ml.analysis.analysis import plot_latent_space

#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "ae_ml" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def tune_country(
    ae_type: str,
    country: str, 
    n_trials: int = 40, 
    pruner: str = "median",
    metric: str = "elbo",
    tr: int = 75,
    vr: int = 15,
    latent: bool = False
) -> None:
    print(f"\n==============================")
    print(f"   OPTUNA TUNING FOR {country}")
    print(f"==============================\n")
    print(f"[INFO] Model {ae_type.upper()} selected")
    if ae_type == "vae":
        print(f"[INFO] Tuning metric {metric.upper()} selected")

    set_global_seeds(42)

    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    pr = {
        "median": MedianPruner(n_startup_trials=5),
        "halving": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
    }.get(pruner)
    if pr is None:
        raise ValueError(f"Unknown pruner: {pruner}")

    db_path = out_path / f"{country}_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )
    study.set_user_attr("tuning_metric", metric)
    study.optimize(
        lambda t: objective(ae_type, t, metric, country, tr, vr, out_path),
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
    X_cont_df, X_cat_df, num_cont, cat_dims = load_feature_matrix(country)
    Xc_np = X_cont_df.values.astype(np.float64)
    Xk_np = X_cat_df.values.astype(np.int64)

    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        tr/100,
        vr/100
    )

    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    p = study.best_trial.params
    depth = p["depth"]
    hidden_dims = [p[f"h{i}"] for i in range(depth)]
    cont_weight = p.get("cont_weight", 1.0)
    cat_weight = p.get("cat_weight", 0.0)
    loss_weights = {"cont_weight": cont_weight, "cat_weight": cat_weight}
    activation = p.get("activation", "relu")

    best_base_cfg = dict(
        num_cont=num_cont,
        cat_dims=cat_dims,
        latent_dim=p["latent_dim"],
        hidden_dims=tuple(hidden_dims),
        dropout=p["dropout"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        batch_size=p["batch_size"],
        num_epochs=90,
        patience=p["patience"],
        gradient_clip=1.0,
        use_lr_scheduler=True,
        embedding_dim=p["embedding_dim"],
        continuous_noise_std=p["noise_std"],
        residual_strength=p["residual_strength"],
        optimizer=p["optimizer"],
        lr_scheduler=p["lr_scheduler"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        activation=activation,
        temperature=1.0
    )
    if ae_type == "vae":
        best_base_cfg["beta"] = p.get("beta", 1.0)

    config_map = {
        "ae": AEConfig,
        "vae": VAEConfig
    }
    best_cfg = config_map[ae_type](**best_base_cfg)

    best_model, best_history = train_autoencoder(
        Xc_train_scald, Xk_train, 
        Xc_val_scald, Xk_val, 
        best_cfg,
        loss_weights
    )

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = out_path / f"{country}_best_model.pt"
    save_autoencoder(
        model=best_model, 
        config=best_cfg, 
        path=out_model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(Xc_train_scald),
            "tuning_metric": metric,
        }
    )

    with open(out_path / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(out_path / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(out_path / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(out_path / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(out_path / f"{country}_cat_dims.json", "w") as f:
        json.dump(cat_dims, f, indent=2)

    print(f"\n[OK] Finished tuning for {country}")

    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = out_path / "analysis" / country
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
    ae_type: str, 
    trials: int, 
    pruner: str, 
    metric: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
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
        tune_all(
            ae_type, 
            args.ntrials, 
            args.pruner.lower(), 
            args.metric.lower(), 
            args.tr, 
            args.vr, 
            args.latent
        )
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
