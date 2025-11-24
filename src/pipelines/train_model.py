#!/usr/bin/env python3
"""
Train a autoencoder for one or all countries:
- produces continuous + categorical feature matrices
- applies scaling to continuous features
- builds autoencoder configuration (AEConfig)
- trains TabularAutoencoder with early stopping

Outputs:
    model weights : results/models/trained/<COUNTRY_CODE>_autoencoder.pt
    continuous scaler : results/models/trained/<COUNTRY_CODE>_scaler_cont.pkl
    mapping of categorical cols : results/models/trained/<COUNTRY_CODE>_cat_dims.json
    continuous feature sizes : results/models/trained/<COUNTRY_CODE>_num_cont.json
    training history : results/models/trained/<COUNTRY_CODE>_training_history.json

Usage (required in <...>, optional in [...]):
    python -m src.pipelines.train_model <COUNTRY_CODE|all> [| tee stdout_train.txt]
"""

import json, pickle
from pathlib import Path
import numpy as np
from src.data import timeseries_seq_split
from src.data.feature_engineering import COUNTRIES, build_feature_matrix
from src.models.autoencoder import AEConfig
from src.models.train import train_autoencoder, save_autoencoder


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def train_country(country: str):
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
    # Split dataset
    # ------------------------------------
    (train_cont, train_cat), (val_cont, val_cat), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=0.75,
        val_ratio=0.15,
    )

    # ------------------------------------
    # AEConfig object
    # ------------------------------------
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
    # Train model
    # ------------------------------------
    model, history = train_autoencoder(
        train_cont, train_cat, 
        val_cont, val_cat, 
        cfg
    )

    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    save_autoencoder(model, cfg, model_path)

    scaler_path = MODELS_DIR / f"{country}_scaler_cont.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved continuous scaler → {scaler_path}")
    
    cat_path = MODELS_DIR / f"{country}_cat_dims.json"
    with open(cat_path, "w") as f:
        json.dump(cat_dims, f, indent=2)
    print(f"[OK] Saved categorical vocab sizes → {cat_path}")

    num_path = MODELS_DIR / f"{country}_num_cont.json"
    with open(num_path, "w") as f:
        json.dump({"num_cont": num_cont}, f, indent=2)
    print(f"[OK] Saved num_cont → {num_path}")

    history_path = MODELS_DIR / f"{country}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    print(f"[DONE] Trained model for {country}")

def train_all():
    for c in COUNTRIES:
        try:
            train_country(c)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model trainings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model for single country or all countries."
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        exit(1)

    target = args.target

    if target.lower() == "all":
        train_all()
    else:
        train_country(target.upper())
