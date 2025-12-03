#!/usr/bin/env python3
"""
Validate a trained autoencoder:
- loads model and features
- computes reconstruction errors
- prints summary stats
- performs latent space analysis if specified

Outputs: 
    error per ts : results/models/validate/<COUNTRY_CODE>_validation.csv

Usage:
    python -m src.pipelines.validate_model [-tr <int>] [-vr <int>] <COUNTRY_CODE|all> [| tee stdout_val.txt]
"""

import pickle, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from app.src.data.feature_engineering import build_feature_matrix, COUNTRIES
from app.src.models.evaluate import reconstruction_error
from app.src.models.train import load_autoencoder
from app.src.data.split import timeseries_seq_split
from app.src.models.analysis import plot_latent_space


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "trained"
OUT_DIR = PROJECT_ROOT / "results" / "models" / "validated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def validate_country(country: str, tr: int, vr: int, latent: bool):
    print(f"\n==============================")
    print(f"  VALIDATE MODEL ({country})")
    print(f"==============================")

    model_path = MODELS_DIR / f"{country}_autoencoder.pt"
    scaler_path = MODELS_DIR / f"{country}_scaler_cont.pkl"
    num_path = MODELS_DIR / f"{country}_num_cont.json"
    cat_path = MODELS_DIR / f"{country}_cat_dims.json"

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")
    if not num_path.exists():
        raise FileNotFoundError(f"[ERROR] Num_cont not found: {num_path}")
    if not cat_path.exists():
        raise FileNotFoundError(f"[ERROR] Cat_dims not found: {cat_path}")
    
    # --------------------
    # Load model + config, scaler, num_cont
    # --------------------
    model, cfg = load_autoencoder(model_path)
    model = model.to(cfg.device)
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(num_path, "r") as f:
        num_cont = json.load(f)["num_cont"]
    
    with open(cat_path, "r") as f:
        cat_dims = json.load(f)

    # --------------------
    # Load feature matrix
    # --------------------
    X_cont_df, X_cat_df, num_cont_check, cat_dims_check = build_feature_matrix(country)

    # Consistency check
    assert num_cont == num_cont_check, "[ERROR] num_cont mismatch between scaler and feature matrix"
    assert cat_dims == cat_dims_check, "[ERROR] cant_dims mismatch between scaler and feature matrix"

    Xc_np = X_cont_df.values.astype(np.float64)
    Xk_np = X_cat_df.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, _), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=tr/100,
        val_ratio=vr/100
    )
    ts_val = ts[len(Xc_train): len(Xc_train) + len(Xc_val)]
    
    # --------------------
    # Apply scaler on cont features and tranform data
    # --------------------
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)
    
    # --------------------
    # Load loss weights
    # --------------------
    payload = torch.load(model_path, map_location="cpu")
    loss_weights = payload.get("additional_info", {}).get("loss_weights", {"cont_weight": 1.0, "cat_weight": 1.0})

    cont_weight = loss_weights["cont_weight"]
    cat_weight = loss_weights["cat_weight"]

    print(f"[INFO] Using loss weights - Continuous: {cont_weight:.2f}, Categorical: {cat_weight:.2f}")

    # --------------------
    # Compute reconstruction error
    # --------------------
    errors = reconstruction_error(
        model=model,
        X_cont=Xc_val_scald,
        X_cat=Xk_val,
        device=cfg.device,
        cont_weight=cont_weight,
        cat_weight=cat_weight
    )

    # --------------------
    # Print summary
    # --------------------
    print("\n--- Validation Error Summary ---")
    print(f"Total samples: {len(errors)}")
    print(f"Min error:  {errors.min():.6f}")
    print(f"Max error:  {errors.max():.6f}")
    print(f"Mean error: {errors.mean():.6f}")
    print(f"Std error:  {errors.std():.6f}")
    print(f"Median:     {np.median(errors):.6f}")
    print(f"95th pct:   {np.percentile(errors, 95):.6f}")
    print(f"99.5th pct: {np.percentile(errors, 99.5):.6f}")
    print(f"99th pct:   {np.percentile(errors, 99):.6f}")

    df_out = pd.DataFrame({
        "ts": ts_val,
        "error": errors,
    })
    out_path = OUT_DIR / f"{country}_validation.csv"
    df_out.to_csv(out_path, index=False)

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        plot_latent_space(
            country, 
            Xc_val_scald,
            Xk_val,
            model,
            cfg.device,
            1000,
            OUT_DIR,
            f"{country}_latent_space.png"
        )
    
    print(f"[DONE] Validation CSV saved: {out_path}")

def validate_all(tr: int, vr: int, latent: bool):
    for c in COUNTRIES:
        try:
            validate_country(c, tr, vr, latent)
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
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after training"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to validate US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    target = args.target

    if target.lower() == "all":
        validate_all(args.tr, args.vr, args.latent)
    else:
        validate_country(target.upper(), args.tr, args.vr, args.latent)
