#!/usr/bin/env python3
"""
Hyperparameter tuning for the TabularAE using Optuna.

Currently implemented search space:
- latent dimensions
- hidden dimensions (base_dim + depth)
- dropout
- learning rate
- weight decay
- batch size
- patience for early stopping
- gradient clipping
- noise gaussian std and mask probability
- optimizer type
- optimizer-specific paramters like beta1, beta2 for adam, sgd_momentum for sgd
- lr scheduler type
- activation function for encoder and decoder
- loss weights (cont and cat)
- (for VAE) beta

To add parameters for tuning, check README_tuning.ms

Outputs:
    PATH : results/ae_ml/tuned/<MODEL>
    FILES: <COUNTRY>_study_<TUNE_PHASE>.db,
           <COUNTRY>_best_model.pt, 
           <COUNTRY>_best_params.json, 
           <COUNTRY>_best_config.json, 
           <COUNTRY>_best_history.json, 
           <COUNTRY>_scaler.pkl, 
           analysis/<COUNTRY>_latent_space_pca_coords.csv, 
           analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.ae.tune_model [-tr <int>] [-vr <int>] [-N <int>] [-P <median|halving|hyperband>] [-L] [-r <int>] <MODEL> <COUNTRY|all>
"""

import csv, json, pickle, torch, optuna, yaml, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from dataclasses import asdict
from optuna.pruners import (
    MedianPruner, 
    SuccessiveHalvingPruner, 
    HyperbandPruner
)

from app.src.data.feature_engineering import COUNTRIES, load_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.tuning.tune_ae import set_global_seeds, objective
from app.src.ml.training.train_ae import train_autoencoder, save_autoencoder
from app.src.ml.analysis import plot_latent_space

#########################################
##                PARAMS               ##
#########################################
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "ae_ml" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_params(param_type: str) -> dict[str, float | bool | list[float | int | str]]:
    PARAM_FILE = FILE_DIR.parent.parent / "config" / "tune_ae" / f"{param_type}.yml"
    if not PARAM_FILE.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {PARAM_FILE}")
    
    try:
        with open(PARAM_FILE, "r") as f:
            params = yaml.safe_load(f)
            _ = params.keys()
        print(f"[OK] YML params of {param_type} loaded!")
    except AttributeError:
        print(f"[ERROR] YML file empty!")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] YML params load failed for reason: {e}")
        sys.exit(0)
    
    return params

#########################################
##                 RUN                 ##
#########################################

def tune_country(
    ae_type: str,
    country: str, 
    n_trials: int = 40, 
    pruner: str = "median",
    tr: int = 75,
    vr: int = 15,
    latent: bool = False,
    retune_no: int = 0
) -> None:
    print(f"\n==============================")
    print(f"   OPTUNA TUNING FOR {country}")
    print(f"==============================\n")
    print(f"[INFO] Model {ae_type.upper()} selected")

    # -----------------------------
    # Load search space params
    # -----------------------------
    tune_phase = "base" if retune_no == 0 else "retune"
    params = load_params(tune_phase)
    required_params = set([
        "depth", "base_dim", "latent_dim", "dropout", "optimizer", "lr_scheduler", 
        "lr", "weight_decay", "adam_beta1", "adam_beta2", "sgd_momentum", 
        "gradient_clip", "batch_size", "patience", "noise_gauss_std", "noise_mask_prob",
        "activation_en", "activation_de", "cont_weight", "cat_weight"
    ])
    if ae_type == "vae":
        required_params = required_params | set(["beta"])
    if set(params.keys()) != required_params:
        raise KeyError(f"[ERROR] yml file is missing parameter: {set(params.keys()) ^ required_params}")

    # -----------------------------
    # Prepare study
    # -----------------------------
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

    if tune_phase == "retune":
        tune_phase = f"retune_{retune_no}"
    db_path = OUT_DIR / f"{ae_type}" / f"{country}_study_{tune_phase}.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(ae_type, t, country, tr, vr, out_path, params),
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
    print("\n[INFO] Retraining on best params.")
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
    hidden_dims = [max(32, int(p["base_dim"] / (2**i))) for i in range(p["depth"])]
    best_base_cfg = dict(
        num_cont=num_cont,
        cat_dims=cat_dims,
        use_embedding=False,
        hidden_dims=tuple(hidden_dims),
        latent_dim=p["latent_dim"],
        activation_en=p["activation_en"],
        activation_de=p["activation_de"],
        dropout=p["dropout"],
        optimizer=p["optimizer"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        adam_beta1=p["adam_beta1"],
        adam_beta2=p["adam_beta2"],
        sgd_momentum=p["sgd_momentum"],
        lr_scheduler=p["lr_scheduler"],
        gradient_clip=p["gradient_clip"],
        batch_size=p["batch_size"],
        allow_noise_injection=True,
        noise_gauss_std=p["noise_gauss_std"],
        noise_mask_prob=p["noise_mask_prob"],
        num_epochs=60,
        warmup_epochs=10,
        patience=p["patience"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if ae_type == "vae":
        base_cfg = base_cfg | dict(
            temperature=1.0,
            use_beta_annealing=True,
            beta_schedule="linear",
            beta=p.get["beta"],
            debug_kl_stats=True
        )
        del base_cfg["warmup_epochs"]
    
    loss_weights = {
        "cont_w": p["cont_w"], 
        "cat_w": p["cat_w"]
    }

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
        cat_dims=cat_dims,
        num_cont=num_cont,
        path=out_model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(Xc_train_scald),
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

    mode = "w" if tune_phase == "base" else "a"
    clean_params = {}
    clean_params["phase"] = tune_phase
    for k, v in params.items():
        if isinstance(v, dict):
            clean_params[k] = [v["start"], v["end"]]
        else:
            clean_params[k] = list(v)
    clean_params["pruner"] = pruner
    clean_params["best_trial"] = study.best_trial.number
    clean_params["best_val_loss"] = study.best_trial.values[0]
    with open(OUT_DIR / f"{ae_type}" /f"{country}_search_space.csv", mode=mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=clean_params.keys())
        if tune_phase == "base":
            writer.writeheader()
        writer.writerow(clean_params)

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
    tr: int, 
    vr: int, 
    latent: bool,
    retune_no: int = 0
) -> None:
    for c in COUNTRIES:
        try:
            tune_country(
                ae_type,
                c, 
                n_trials=trials, 
                pruner=pruner, 
                tr=tr, 
                vr=vr,
                latent=latent,
                retune_no=retune_no
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
            
    print(f"\n[DONE] All model tunings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune AE hyperparameters for single or for all countries.")

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
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after tuning"
    )

    parser.add_argument(
        "-r", "--retune",
        type=int,
        default=0,
        help="if set, parameter read from yml for retuning and retune phase number is assigned [default: 0 = base]"
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

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)
    
    if args.pruner.lower() not in ["median", "halving", "hyperband"]:
        parser.print_help()
        exit(1)
    
    tune_phase = "retune" if args.retune else "base"

    if args.target.lower() == "all":
        tune_all(
            ae_type, 
            args.ntrials, 
            args.pruner.lower(), 
            args.tr, 
            args.vr, 
            args.latent, 
            args.retune
        )
    else:
        tune_country(
            ae_type=ae_type,
            country=args.target.upper(),
            n_trials=args.ntrials,
            pruner=args.pruner.lower(),
            tr=args.tr, 
            vr=args.vr,
            latent=args.latent,
            retune_no=args.retune
        )
