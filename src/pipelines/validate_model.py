#!/usr/bin/env python3
"""
Validate a trained autoencoder:
- loads model and features
- computes reconstruction errors
- prints summary stats

Outputs: 
    error per ts : results/models/validate/<COUNTRY_CODE>_validation.csv

Usage (required in <...>, optional in [...]):
    python -m src.pipelines.validate_model <COUNTRY_CODE|all> [>> stdout_val.txt]
"""

import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.feature_engineering import build_feature_matrix, COUNTRIES
from src.models.evaluate import reconstruction_error
from src.models.train import load_autoencoder


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
    # Load feature matrix (unscaled)
    # --------------------
    X_cont_df, X_cat_df, num_cont_check, cat_dims, _ = build_feature_matrix(country)

    # Consistency check
    if num_cont != num_cont_check:
        raise ValueError("[ERROR] num_cont mismatch between scaler and feature matrix")

    # scale continuous features using training scaler
    Xc_np = X_cont_df.values.astype(np.float32)
    Xk_np = X_cat_df.values.astype(np.int64)

    # -------------------------
    # APPLY SAME SPLIT AS TRAIN
    # -------------------------
    # TODO: random sampling not implemented
    ts = X_cont_df.index
    n_total = len(ts)
    
    n_train = int(n_total * (1 - cfg.val_split))
    n_val = n_total - n_train

    Xc_val = Xc_np[n_train:]
    Xk_val = Xk_np[n_train:]
    ts_val = ts[n_train:]

    print(f"Total samples: {n_total}")
    print(f"Train samples: {n_train}")
    print(f"Val samples:   {n_val}")

    # -------------------------
    # Compute reconstruction error
    # -------------------------
    errors = reconstruction_error(
        model=model,
        X_cont=Xc_val,
        X_cat=Xk_val,
        device=cfg.device,
    )

    # -------------------------
    # Print summary
    # -------------------------
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

    print(f"[DONE] Validation CSV saved: {out_path}")

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
