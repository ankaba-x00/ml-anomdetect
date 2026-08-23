#!/usr/bin/env python3
"""
Hyperparameter tuning using Optuna.

Set search space parameters in config/tune/config_<MODEL>.yml.
Check README_tuning.md for additional documentation.

Output:
    PATH : results/aml/tuned/<MODEL>
    FILES: <COUNTRY>_<RETUNE_NO>_study.db
           <COUNTRY>_<RETUNE_NO>_best_config.json 
           <COUNTRY>_<RETUNE_NO>_best_history.json
           <COUNTRY>_search_space.csv
           analysis/<COUNTRY>_<RETUNE_NO>_latent_space_pca_coords.csv
           analysis/<COUNTRY>_<RETUNE_NO>_latent_space.png

Usage:
    python -m app.src.pipelines.tune_model [-tr <int>] [-vr <int>] [-N <int>] [-P <median|halving|hyperband>] [-S <random|tpe|gp|brute>] [-L] [-r <int>] <MODEL> <COUNTRY|all>
"""

import json, optuna, warnings
import numpy as np
from optuna.pruners import (
    MedianPruner, 
    SuccessiveHalvingPruner, 
    HyperbandPruner
)
from optuna.samplers import (
    BruteForceSampler,
    GPSampler, 
    RandomSampler, 
    TPESampler
)
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from app.src.data import ISO_3166_alpha2, timeseries_seq_split
from app.src.data.feature_engineering import (
    COUNTRIES, 
    load_feature_matrix,
    load_supervised_feature_matrix
)
from app.src.ml.analysis import plot_latent_space
from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig
from app.src.ml.training.train_ae import train_autoencoder
from app.src.ml.training.train_mt import train_mt_autoencoder
from app.src.ml.tuning.tune import objective
from app.src.ml.tuning.io_utils import TrialSummaryWriter

warnings.filterwarnings("ignore")


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tune_model(
    ae_type: str,
    country: str, 
    n_trials: int, 
    sampler_type: str,
    pruner_type: str,
    tr: int,
    vr: int,
    latent: bool,
    retune_no: int
) -> None:
    print(f"\n==============================")
    print(f"     TUNE {country} MODEL     ")
    print(f"==============================")
    print(f"\n[INFO] Model {ae_type.upper()} selected")

    # -----------------------------
    # Prepare study
    # -----------------------------
    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    sampler = {
        "random": RandomSampler(seed=42),
        "gp": GPSampler(seed=42),
        "tpe": TPESampler(seed=42),
        "brute": BruteForceSampler(seed=42)
    }.get(sampler_type)

    pruner = {
        "median": MedianPruner(n_startup_trials=5),
        "halving": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
        "none": None
    }.get(pruner_type)

    db_path = OUT_DIR / f"{ae_type}" / f"{country}_{retune_no}_study.db"

    # -----------------------------
    # Run study
    # -----------------------------
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(ae_type, t, country, tr, vr, out_path),
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
    print("\n[INFO] Retraining on best params")

    if ae_type in ["ae", "vae"]:
        X_cont, X_cat, num_cont, cat_dims = load_feature_matrix(country)
    else:
        X_cont, X_cat, y3_s, y7_s, ya_s, num_cont, cat_dims = load_supervised_feature_matrix(country)
    
    Xc = X_cont.values.astype(np.float32)
    Xk = X_cat.values.astype(np.int64)
    if ae_type in ["mtae"]:
        y3 = y3_s.values.astype(np.float32)
        y7 = y7_s.values.astype(np.float32)
        ya = ya_s.values.astype(np.int64)

    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk,
        tr/100,
        vr/100
    )
    if ae_type in ["mtae"]:
        y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
        y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
        ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)
    

    config_map = {
        "ae": AEConfig,
        "vae": VAEConfig,
        "mtae": MTAEConfig
    }
    cfg = study.best_trial.user_attrs["config"]
    best_cfg = config_map[ae_type](**cfg)

    loss_weights = {
        "cont_w": best_cfg.cont_w, 
        "cat_w": best_cfg.cat_w
    }
    if ae_type in ["mtae"]:
        loss_weights |= {
            "l3_w": best_cfg.lambda_l3,
            "l7_w": best_cfg.lambda_l7,
            "at_w": best_cfg.lambda_at
        }

    if ae_type in ["ae", "vae"]:
        best_model, best_history = train_autoencoder(
            Xc_train_scald, Xk_train, 
            Xc_val_scald, Xk_val, 
            best_cfg,
            loss_weights
        )
    else:
        best_model, best_history = train_mt_autoencoder(
            Xc_train_scald, Xk_train, y3_tr, y7_tr, ya_tr,
            Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
            best_cfg,
            loss_weights
        )

    # ------------------------------------
    # Save artefacts
    # ------------------------------------
    with open(out_path / f"{country}_{retune_no}_best_config.json", "w") as f:
        best_cfg.to_json(f)

    with open(out_path / f"{country}_{retune_no}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    writer = TrialSummaryWriter(
        retune_no,
        ae_type,
        sampler_type,
        pruner_type, 
        study.best_trial.number,
        study.best_trial.values[0]
    )
    writer.write(OUT_DIR / f"{ae_type}" /f"{country}_search_space.csv")

    print(f"[DONE] Saved best model to {out_path}")

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
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
            f"{country}_{retune_no}_best_latent_space.png"
        )

    print(f"\n[DONE] Tuned model for {country}")

def tune_all(
    ae_type: str, 
    n_trials: int, 
    sampler_type: str,
    pruner_type: str, 
    tr: int, 
    vr: int, 
    latent: bool,
    retune_no: int = 0
) -> None:
    for c in COUNTRIES:
        try:
            tune_model(
                ae_type,
                c, 
                n_trials, 
                sampler_type,
                pruner_type, 
                tr, 
                vr,
                latent,
                retune_no
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
            
    print(f"\n[DONE] All tunings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune single- or multi-task autoencoder for one or multiple countries")

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
        "-S", "--sampler",
        type=str, 
        default="tpe",
        help="random|gp|tpe|brute> Optuna sampling strategy [default: tpe]"
    )

    parser.add_argument(
        "-P", "--pruner",
        type=str, 
        default="median",
        help="<median|halving|hyperband|none> Optuna pruner strategy [default: median]"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="perform latent space analysis [default: false]"
    )

    parser.add_argument(
        "-r", "--retune",
        type=int,
        default=0,
        help="retune number [default: 0]"
    )

    parser.add_argument(
        "model",
        help="<ae|vae|mtae> model to train"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' or 'all' for all countries defined in config/models.yaml"
    )

    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae", "mtae"]:
        parser.print_help()
        exit(1)
    
    target = args.target.upper()
    if target not in ISO_3166_alpha2 and target.lower() != "all":
        parser.print_help()
        print(f"[ERROR] Invalid target: {target}")
        exit(1)
    
    if args.pruner.lower() not in ["median", "halving", "hyperband", "none"]:
        parser.print_help()
        exit(1)
    
    if args.sampler.lower() not in ["random", "tpe", "gp", "brute"]:
        parser.print_help()
        exit(1)

    if target.lower() == "all":
        tune_all(
            ae_type, 
            args.ntrials, 
            args.sampler.lower(),
            args.pruner.lower(), 
            args.tr, 
            args.vr, 
            args.latent, 
            args.retune
        )
    else:
        tune_model(
            ae_type,
            target,
            args.ntrials,
            args.sampler.lower(),
            args.pruner.lower(),
            args.tr, 
            args.vr,
            args.latent,
            args.retune
        )
