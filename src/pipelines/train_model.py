#!/usr/bin/env python3

import json, pickle
from pathlib import Path
import numpy as np
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

def train_single(country: str):
    print(f"\n==============================")
    print(f"  TRAIN AUTOENCODER ({country})")
    print(f"==============================")

    X, scaler = build_feature_matrix(country)
    X_np = X.values.astype(np.float32)
    input_dim = X_np.shape[1]

    cfg = AEConfig(
        input_dim=input_dim,
        latent_dim=16,
        hidden_dims=(128, 64),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        num_epochs=60,
        patience=6,
        val_split=0.2,
        time_series_split=True,
        gradient_clip=1.0,
        use_lr_scheduler=True,
    )

    model, history = train_autoencoder(X_np, cfg)

    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    save_autoencoder(model, cfg, model_path)

    scaler_path = MODELS_DIR / f"{country}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved scaler to {scaler_path}")

    history_path = MODELS_DIR / f"{country}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    print(f"[DONE] Trained model for {country}")

def train_all():
    for c in COUNTRIES:
        try:
            train_single(c)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model for single country only or for all countries."
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
        train_single(target.upper())
