#!/usr/bin/env python3
"""
Validate a trained autoencoder:
- loads model and features
- computes reconstruction errors
- prints summary stats

Outputs: 
    error per ts : results/models/validate/<COUNTRY_CODE>_validation.csv

Usage:
    python -m src.pipelines.validate_model [-tr <int>] [-vr <int>] <COUNTRY_CODE|all> [| tee stdout_val.txt]
"""

import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.feature_engineering import build_feature_matrix, COUNTRIES
from src.models.evaluate import reconstruction_error
from src.models.train import load_autoencoder
from src.data.split import timeseries_seq_split


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "trained"
OUT_DIR = PROJECT_ROOT / "results" / "models" / "validated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def validate_country(country: str, tr: int, vr: int):
    print(f"\n==============================")
    print(f"  VALIDATE MODEL ({country})")
    print(f"==============================")

    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    scaler_path = MODELS_DIR / f"{country}_scaler_cont.pkl"
    num_path = MODELS_DIR / f"{country}_num_cont.json"


    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")
    if not num_path.exists():
        raise FileNotFoundError(f"[ERROR] Num_cont not found: {num_path}")
    
    # --------------------
    # Load model + config, scaler, num_cont
    # --------------------
    model, cfg = load_autoencoder(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    num_cont = json.load(open(num_path))["num_cont"]

    # --------------------
    # Load feature matrix
    # --------------------
    X_cont_df, X_cat_df, num_cont_check, cat_dims, _ = build_feature_matrix(country, scaler=scaler)

    # Consistency check
    if num_cont != num_cont_check:
        raise ValueError("[ERROR] num_cont mismatch between scaler and feature matrix")

    Xc_np = X_cont_df.values.astype(np.float32)
    Xk_np = X_cat_df.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=tr/100,
        val_ratio=vr/100
    )
    ts_val = ts[len(Xc_train): len(Xc_train) + len(Xc_val)]
    
    # --------------------
    # Compute reconstruction error
    # --------------------
    errors = reconstruction_error(
        model=model,
        X_cont=Xc_val,
        X_cat=Xk_val,
        device=cfg.device,
    )

    # --------------------
    # Print summary
    # --------------------
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
    out_path = OUT_DIR / f"{country}_validation.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[DONE] Validation CSV saved: {out_path}")

def validate_all(tr: int, vr: int):
    for c in COUNTRIES:
        try:
            validate_country(c, tr, vr)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model validations completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate model for single country or all countries."
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
        "target",
        help="<COUNTRY|all> e.g. 'US' to validate US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    target = args.target

    if target.lower() == "all":
        validate_all(args.tr, args.vr)
    else:
        validate_country(target.upper(), args.tr, args.vr)
