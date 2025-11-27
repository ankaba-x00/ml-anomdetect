#!/usr/bin/env python3
"""
Train a autoencoder for one or all countries:
- produces continuous + categorical feature matrices
- applies scaling to continuous features
- builds autoencoder configuration (AEConfig)
- trains TabularAutoencoder with early stopping

Outputs:
    for tuning : results/models/trained/<COUNTRY_CODE>_autoencoder.pt
                 results/models/trained/<COUNTRY_CODE>_scaler_cont.pkl
                 results/models/trained/<COUNTRY_CODE>_cat_dims.json
                 results/models/trained/<COUNTRY_CODE>_num_cont.json
                 results/models/trained/<COUNTRY_CODE>_training_history.json
    for inference: deployment/models/<COUNTRY_CODE>_autoencoder.pt
                   deployment/models/<COUNTRY_CODE>_scaler_cont.pkl
                   deployment/models/<COUNTRY_CODE>_cat_dims.json
                   deployment/models/<COUNTRY_CODE>_num_cont.json
                   deployment/models/<COUNTRY_CODE>_training_history.json
                   deployment/models/<COUNTRY_CODE>_cal_threshold.json"

Usage:
    python -m src.pipelines.train_model [-tr <int>] [-vr <int>] [-F] [-M <p99|p995|mad>] <COUNTRY_CODE|all> [| tee stdout_train.txt]
"""

import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import timeseries_seq_split
from src.data.feature_engineering import COUNTRIES, build_feature_matrix
from src.models.calibrate import calibrate_threshold
from src.models.autoencoder import AEConfig
from src.models.train import train_autoencoder, save_autoencoder


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "models" / "trained"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODELS_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
FULL_OUT_DIR = PROJECT_ROOT / "deployment" / "models"


#########################################
##                 RUN                 ##
#########################################

def train_country(
        country: str, 
        tr: int, 
        vr: int, 
        full: bool, 
        method: str, 
        cw: int
    ):
    print(f"\n==============================")
    print(f"  TRAIN AUTOENCODER ({country})")
    print(f"==============================")
    
    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    X_cont, X_cat, num_cont, cat_dims, scaler = build_feature_matrix(country)
    Xc_np = X_cont.values.astype(np.float32)
    Xk_np = X_cat.values.astype(np.int64)
      
    # ------------------------------------
    # AEConfig object: Load or construct
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Reading AEConfig from best tuning run.")
        tuned_cfg_path = BEST_MODELS_DIR / f"{country}_best_config.json"
        if not tuned_cfg_path.exists():
            raise FileNotFoundError("[ERROR] Best config not found. Run tuning first.")
        with open(tuned_cfg_path, "r") as f:
            cfg_dict = json.load(f)
            cfg = AEConfig(**cfg_dict)
    else:
        print(f"[INFO] Constructing AEConfig from inital params.")
        cfg = AEConfig(
            num_cont=num_cont,
            cat_dims=cat_dims,
            latent_dim=16,
            hidden_dims=(128, 64),
            dropout=0.1,
            embedding_dim=12,
            continuous_noise_std=0.01,
            residual_strength=0.10,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            num_epochs=60,
            patience=6,
            gradient_clip=1.0,
            optimizer="adam",
            lr_scheduler="plateau",
            use_lr_scheduler=True,
        )

    # ------------------------------------
    # Full OR Split : Train model
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Dataset not split: 100% train.")
        model, history = train_autoencoder(
            Xc_np, Xk_np,
            None, None,
            cfg
        )
        OUT_DIR = FULL_OUT_DIR
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test.")
        (train_cont, train_cat), (val_cont, val_cat), _ = timeseries_seq_split(
            Xc_np, Xk_np,
            train_ratio=tr/100,
            val_ratio=vr/100,
        )
        model, history = train_autoencoder(
            train_cont, train_cat, 
            val_cont, val_cat, 
            cfg
        )                 

    model_path = OUT_DIR / f"{country}_autoencoder.pt"
    save_autoencoder(model, cfg, model_path)

    scaler_path = OUT_DIR / f"{country}_scaler_cont.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved continuous scaler to {scaler_path}")
    
    cat_path = OUT_DIR / f"{country}_cat_dims.json"
    with open(cat_path, "w") as f:
        json.dump(cat_dims, f, indent=2)
    print(f"[OK] Saved categorical vocab sizes to {cat_path}")

    num_path = OUT_DIR / f"{country}_num_cont.json"
    with open(num_path, "w") as f:
        json.dump({"num_cont": num_cont}, f, indent=2)
    print(f"[OK] Saved num_cont to {num_path}")

    history_path = OUT_DIR / f"{country}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    print(f"[DONE] Trained model for {country}")

    # ------------------------------------
    # Error and threshold on calibration window
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Computing threshold on calibration window...")
        
        threshold = calibrate_threshold(
            country, model, scaler, cfg.device, method, cw
        )["threshold"]

        thr_path = FULL_OUT_DIR / f"{country}_cal_threshold.json"
        with open(thr_path, "w") as f:
            json.dump({"threshold": threshold}, f, indent=2)
        print(f"[OK] Saved threshold to {thr_path}")

        print(f"[DONE] Preparation for inference model for {country}")


def train_all(tr: int, vr: int, full: bool, method: str, cw: int):
    for c in COUNTRIES:
        try:
            train_country(c, tr, vr, full, method, cw)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model trainings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model for single country or all countries."
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
        "-F", "--full",
        action="store_true",
        help="train on full dataset, no validation (for inference); overwrites tr and vr"
    )

    parser.add_argument(
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method [default: p99]"
    )

    parser.add_argument(
        "-CW", "--calwindow",
        type=int,
        default=30,
        help="calibration window in days for computing threshold [default: 30]"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    target = args.target

    if target.lower() == "all":
        train_all(args.tr, args.vr, args.full, args.method, args.calwindow)
    else:
        train_country(target.upper(), args.tr, args.vr, args.full, args.method, args.calwindow)
