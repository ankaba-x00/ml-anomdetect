#!/usr/bin/env python3
"""
Test multi-task predictor on new data:
- loads trained MT model
- computes L3/L7 residuals
- computes anomaly score (weighted residuals)
- thresholds anomaly score
- outputs attack predictions
- performs latent space analysis if specified

Outputs:
    PATH : results/mt_ml/tested
    FILES : <COUNTRY>_errors_<method>.csv, 
            analysis/<COUNTRY>_latent_space_pca_coords.csv, 
            analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.ae.test_model [-M <p99|p995|mad>] [-tr <int>] [-vr <int>] [-L] <MODEL> <COUNTRY|all>
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from app.src.data.feature_engineering import load_supervised_feature_matrix, COUNTRIES
from app.src.data.split import timeseries_seq_split
from app.src.ml.training.train_mt import load_multitask_model
from app.src.ml.analysis import plot_latent_space
from app.src.ml.training.evaluate_mt import apply_multitask_model


#########################################
##               PARAMS                ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
MODELS_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tuned"
OUT_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tested"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def test_country(
    country: str, 
    method: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
    print(f"\n==============================")
    print(f"   TEST MT MODEL ({country})")
    print(f"==============================")

    model_path = MODELS_DIR / f"{country}_best_model.pt"
    scaler_path = MODELS_DIR / f"{country}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[Error] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[Error] Scaler not found: {scaler_path}")

    model, cfg, model_num_cont, model_cat_dims = load_multitask_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # -----------------------------
    # Load supervised feature matrix
    # -----------------------------
    Xc_df, Xk_df, y3_s, y7_s, ya_s, num_cont, cat_dims = (
        load_supervised_feature_matrix(country)
    )

    assert num_cont == model_num_cont, "[ERROR] num_cont mismatch between cfg and supervised matrix"
    assert cat_dims == model_cat_dims, "[ERROR] cat_dims mismatch between cfg and supervised matrix"

    Xc = Xc_df.values.astype(np.float32)
    Xk = Xk_df.values.astype(np.int64)
    y3 = y3_s.values.astype(np.float32)
    y7 = y7_s.values.astype(np.float32)
    ya = ya_s.values.astype(np.int64)
    ts = Xc_df.index

    # -----------------------------
    # Split dataset
    # -----------------------------
    (Xc_tr, _), (_, _), (Xc_te, Xk_te) = timeseries_seq_split(
        Xc, Xk, 
        tr/100, 
        vr/100
    )
    _, _, y3_te = timeseries_seq_split(y3, None, tr/100, vr/100)
    _, _, y7_te = timeseries_seq_split(y7, None, tr/100, vr/100)
    _, _, ya_te = timeseries_seq_split(ya, None, tr/100, vr/100)

    ts_te = ts[len(Xc_tr) + int(len(ts) * vr/100):]

    Xc_te = scaler.transform(Xc_te)

    # -----------------------------
    # Apply model
    # -----------------------------
    results = apply_multitask_model(
        model=model,
        X_cont=Xc_te,
        X_cat=Xk_te,
        y_l3=y3_te,
        y_l7=y7_te,
        y_attack=ya_te,
        method=method,
        device=cfg.device,
        l3_weight=cfg.lambda_l3,
        l7_weight=cfg.lambda_l7,
        min_length=1,
        merge_gap=0,
    )

    errors = results["loss_total"]
    thr = results["threshold"]
    mask = results["mask"]
    starts = results["anomaly_starts"]
    ends = results["anomaly_ends"]

    # --------------------
    # Print summary
    # --------------------
    print("\n--- MT Test Summary ---")
    print(f"Total samples: {len(errors)}")
    print(f"Threshold ({method}): {thr:.6f}")
    print(f"Flagged samples: {int(mask.sum())}")
    print(f"Min:   {errors.min():.6f}")
    print(f"Mean:  {errors.mean():.6f}")
    print(f"Std:   {errors.std():.6f}")
    print(f"Median:{np.median(errors):.6f}")
    print(f"99th:  {np.percentile(errors, 99):.6f}")
    
    # -----------------------------
    # Save CSV
    # -----------------------------
    df = pd.DataFrame({
        "ts": ts_te,
        "loss_total": errors,
        "loss_l3": results["loss_l3"],
        "loss_l7": results["loss_l7"],
        "loss_attack": results["loss_attack"],
        "l3_true": y3_te,
        "l3_pred": results["l3_pred"],
        "l7_true": y7_te,
        "l7_pred": results["l7_pred"],
        "attack_true": ya_te,
        "attack_pred": results["attack_pred"],
        "attack_conf": results["attack_prob_max"],
        "threshold": thr,
        "is_flagged": mask.astype(int),
    })
    val_path = OUT_DIR / f"{country}_errors_{method}.csv"
    df.to_csv(val_path, index=False)
    print(f"[OK] Saved results CSV to {val_path}")

    df_int = pd.DataFrame({
        "start_idx": starts,
        "end_idx": ends,
        "start_ts": ts_te[starts] if len(starts) else [],
        "end_ts": ts_te[ends - 1] if len(ends) else [],
        "duration_samples": ends - starts,
    })
    int_path = OUT_DIR / f"{country}_intervals_{method}.csv"
    df_int.to_csv(int_path, index=False)
    print(f"[OK] Saved intervals CSV to {int_path}")

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = OUT_DIR / "analysis" / country
        out_path.mkdir(parents=True, exist_ok=True)
        plot_latent_space(
            country,
            Xc_te,
            Xk_te,
            model,
            cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png",
        )

    print(f"[DONE] Tested MT model for {country}")


def test_all(
    method: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
    for c in COUNTRIES:
        try:
            test_country(c, method, tr, vr, latent)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    
    print(f"\n[DONE] All model testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test MT model for single country or all countries."
    )

    parser.add_argument(
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method [default: p99]"
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
        help="generate latent space plot after tuning"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    if args.target.lower() == "all":
        test_all(
            method=args.method, 
            tr=args.tr,
            vr=args.vr,
            latent=args.latent
        )
    else:
        test_country(
            country=args.target.upper(), 
            method=args.method, 
            tr=args.tr,
            vr=args.vr,
            latent=args.latent
        )
