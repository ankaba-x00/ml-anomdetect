#!/usr/bin/env python3
"""
Validate a trained autoencoder:
 - loads model and features
 - computes reconstruction errors
 - prints summary stats
 - saves CSV

Usage:
    python -m src.pipelines.validate_model US
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.data.feature_engineering import build_feature_matrix
from src.models.evaluate import reconstruction_error
from src.models.train import load_autoencoder
from src.data.feature_engineering import COUNTRIES


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "results" / "models" / "validated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def validate_single(country: str):
    print(f"\n==============================")
    print(f"  VALIDATE MODEL ({country})")
    print(f"==============================")

    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    scaler_path = MODELS_DIR / f"{country}_scaler.pkl"

    model, cfg = load_autoencoder(model_path)

    # --------------------
    # Load feature matrix
    # --------------------
    # TODO: write load function later instead of being lazy and build again
    X, scaler = build_feature_matrix(country)
    X_np = X.values.astype(np.float32)

    # -------------------------
    # APPLY SAME SPLIT AS TRAIN
    # -------------------------
    # TODO: random sampling not yet implemented
    n_total = len(X_np)
    n_train = int(n_total * (1 - cfg.val_split))
    n_val = n_total - n_train

    X_val = X_np[n_train:]
    ts_val = X.index[n_train:]

    print(f"Total samples: {n_total}")
    print(f"Train samples: {n_train}")
    print(f"Val samples:   {n_val}")

    # -------------------------
    # Compute reconstruction error
    # -------------------------
    errors = reconstruction_error(model, X_val, cfg.device)

    print("\n--- Validation Error Summary ---")
    print(f"Min error:  {errors.min():.6f}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Median:     {np.median(errors):.6f}")
    print(f"95th pct:   {np.percentile(errors, 95):.6f}")
    print(f"99th pct:   {np.percentile(errors, 99):.6f}")

    df_out = pd.DataFrame({
        "ts": ts_val,
        "error": errors,
    })
    out_path = OUTPUT_DIR / f"{country}_validation.csv"
    df_out.to_csv(out_path, index=False)

    print(f"[OK] Validation CSV saved: {out_path}")

def validate_all():
    for c in COUNTRIES:
        try:
            validate_single(c)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate model predictions for single country or all countries."
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to validate US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        exit(1)

    target = args.target

    if target.lower() == "all":
        validate_all()
    else:
        validate_single(target.upper())
