#!/usr/bin/env python3
"""
Validate a trained multi-task traffic model:
- loads model + scaler + supervised features
- computes per-sample loss_total and head losses
- prints summary stats
- saves validation CSV
- performs latent space analysis if specified

Outputs:
    PATH : results/mt_ml/validated
    FILES : <COUNTRY>_mt_validation.csv, 
            <COUNTRY>_latent_space_pca_coords.csv, 
            <COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.mt.validate_mtmodel [-tr <int>] [-vr <int>] <COUNTRY|all>
"""

import pickle, torch
import numpy as np
import pandas as pd
from pathlib import Path

from app.src.data.feature_engineering import COUNTRIES
from app.src.data.split import timeseries_seq_split
from app.src.ml.analysis.analysis import plot_latent_space
from app.src.ml.training.evaluate_mt import apply_multitask_model
from app.src.ml.training.train_mt import load_multitask_model
from app.src.data.feature_engineering import load_supervised_feature_matrix


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
MODELS_DIR = PROJECT_ROOT / "results" / "mt_ml" / "trained"
OUT_DIR = PROJECT_ROOT / "results" / "mt_ml" / "validated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def validate_country(
    country: str, 
    tr: int, 
    vr: int, 
    method: str, 
    latent: bool
) -> None:
    print(f"\n==============================")
    print(f"  VALIDATE MT MODEL ({country})")
    print(f"==============================")

    model_path = MODELS_DIR / f"{country}_multitask_model.pt"
    scaler_path = MODELS_DIR / f"{country}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")

    # --------------------
    # Load model + config, scaler
    # --------------------
    model, cfg, model_num_cont, model_cat_dims = load_multitask_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --------------------
    # Load supervised matrix
    # --------------------
    X_cont_df, X_cat_df, y3_s, y7_s, ya_s, num_cont, cat_dims = load_supervised_feature_matrix(country)

    assert num_cont == model_num_cont, "[ERROR] num_cont mismatch between cfg and supervised matrix"
    assert cat_dims == model_cat_dims, "[ERROR] cat_dims mismatch between cfg and supervised matrix"

    Xc = X_cont_df.values.astype(np.float64)
    Xk = X_cat_df.values.astype(np.int64)
    y3 = y3_s.values.astype(np.float32)
    y7 = y7_s.values.astype(np.float32)
    ya = ya_s.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_tr, _), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk, 
        tr/100, 
        vr/100
    )
    ts_val = ts[len(Xc_tr): len(Xc_tr) + len(Xc_val)]
    
    _, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
    _, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
    _, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)    

    # --------------------
    # Scale cont features
    # --------------------
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    # --------------------
    # Load loss weights (lambdas)
    # --------------------
    payload = torch.load(model_path, map_location="cpu")
    loss_weights = payload.get("additional_info", {}).get("loss_weights", {
        "l3": 1.0, "l7": 1.0, "attack": 1.0
    })

    l3_w = float(loss_weights.get("l3", 1.0))
    l7_w = float(loss_weights.get("l7", 1.0))
    att_w = float(loss_weights.get("attack", 1.0))

    print(f"[INFO] Using loss weights - L3: {l3_w:.2f}, L7: {l7_w:.2f}, Attack: {att_w:.2f}")

    # --------------------
    # Compute validation errors
    # --------------------
    res = apply_multitask_model(
        model=model,
        X_cont=Xc_val_scald,
        X_cat=Xk_val,
        y_l3=y3_val,
        y_l7=y7_val,
        y_attack=ya_val,
        method=method,
        device=cfg.device,
        l3_weight=l3_w,
        l7_weight=l7_w,
        min_length=1,
        merge_gap=0,
    )

    errors = res["loss_total"]
    thr = res["threshold"]
    mask = res["mask"]

    # --------------------
    # Print summary
    # --------------------
    print("\n--- MT Validation Summary ---")
    print(f"Total samples: {len(errors)}")
    print(f"Threshold ({method}): {thr:.6f}")
    print(f"Flagged samples: {int(mask.sum())}")
    print(f"Min:   {errors.min():.6f}")
    print(f"Mean:  {errors.mean():.6f}")
    print(f"Std:   {errors.std():.6f}")
    print(f"Median:{np.median(errors):.6f}")
    print(f"99th:  {np.percentile(errors, 99):.6f}")

    # --------------------
    # Save CSV
    # --------------------
    df_out = pd.DataFrame({
        "ts": ts_val,
        "loss_total": errors,
        "loss_l3": res["loss_l3"],
        "loss_l7": res["loss_l7"],
        "loss_attack": res["loss_attack"],
        "l3_true": y3_val,
        "l3_pred": res["l3_pred"],
        "l7_true": y7_val,
        "l7_pred": res["l7_pred"],
        "attack_true": ya_val,
        "attack_pred": res["attack_pred"],
        "attack_conf": res["attack_prob_max"],
        "threshold": thr,
        "is_flagged": mask.astype(int),
    })
    val_path = OUT_DIR / f"{country}_mt_validation.csv"
    df_out.to_csv(val_path, index=False)
    print(f"[OK] Validation CSV saved: {val_path}")

    # --------------------
    # Visualize latent space
    # --------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = OUT_DIR / "analysis" / country
        out_path.mkdir(parents=True, exist_ok=True)
        plot_latent_space(
            country,
            Xc_val_scald,
            Xk_val,
            model,
            cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png"
        )

    print(f"[DONE] Validated model for {country}")


def validate_all(
    tr: int, 
    vr: int, 
    method: str, 
    latent: bool
) -> None:
    for c in COUNTRIES:
        try:
            validate_country(c, tr, vr, method, latent)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    
    print(f"\n[DONE] All multi-task validations completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate multi-task attack predictor for single or all countries.")

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
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method on combined loss [default: p99]"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after training"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    if args.target.lower() == "all":
        validate_all(
            args.tr, 
            args.vr, 
            args.method, 
            args.latent
        )
    else:
        validate_country(
            args.target.upper(), 
            args.tr, 
            args.vr, 
            args.method, 
            args.latent
        )
