#!/usr/bin/env python3
"""
Test model to detect anomalies in new data:
- loads model and features
- computes reconstruction errors
- computes threshold of choice (p99, p995, MAD)
- identifies anomaly intervals of certain min. sample length and sample gap 
- prints summary stats

Outputs:
    errors : results/models/tested/<COUNTRY>_errors_<method>.csv
    thresholds : results/models/tested/<COUNTRY>_threshold_<method>.json
    anomalies : results/models/tested/<COUNTRY>_intervals_<method>.csv

Usage:
    python -m src.pipelines.test_model [-M <p99|p995|mad>] [-tr <int>] [-vr <int>] <COUNTRY|all> [| tee stdout_test.txt]
"""

import pickle, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from app.src.data.feature_engineering import build_feature_matrix, COUNTRIES
from app.src.models.train import load_autoencoder
from app.src.models.evaluate import apply_model
from app.src.data.split import timeseries_seq_split


#########################################
##               PATHS                 ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

TUNED_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
OUT_DIR = PROJECT_ROOT / "results" / "models" / "tested"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def test_country(country: str, method: str, tr: int = 75, vr: int = 15):
    print(f"\n==============================")
    print(f"  DETECT ANOMALIES ({country})")
    print(f"==============================")

    # --------------------
    # Load model + config, scaler, cat_dims
    # --------------------
    model_path = TUNED_DIR / f"{country}_best_model.pt"
    scaler_path = TUNED_DIR / f"{country}_scaler.pkl"
    catdims_path = TUNED_DIR / f"{country}_cat_dims.json"

    if not model_path.exists():
        raise FileNotFoundError(f"[Error] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[Error] Scaler not found: {scaler_path}")
    if not catdims_path.exists():
        raise FileNotFoundError(f"[Error] Cat_dims not found: {catdims_path}")

    model, cfg = load_autoencoder(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(catdims_path, "r") as f:
        cat_dims = json.load(f)

    # --------------------
    # Load feature matrix
    # --------------------
    X_cont_df, X_cat_df, _, cat_dims2 = build_feature_matrix(country)

    # ensure consistent categorical structure
    assert cat_dims2 == cat_dims, "[Error] Saved cat_dims mismatch — rebuild features."

    X_cont = X_cont_df.values.astype(np.float64)
    X_cat = X_cat_df.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, _), (Xc_val, _), (Xc_test, Xk_test) = timeseries_seq_split(
        X_cont, X_cat,
        train_ratio=tr/100,
        val_ratio=vr/100
    )

    ts_eval = ts[len(Xc_train)+len(Xc_val):]
    
    # --------------------
    # Apply scaler on cont features and tranform data
    # --------------------
    Xc_test_scald = scaler.transform(Xc_test).astype(np.float32)

    # --------------------
    # Load loss weights
    # --------------------
    payload = torch.load(model_path, map_location="cpu")
    loss_weights = payload.get("additional_info", {}).get("loss_weights", {"cont_weight": 1.0, "cat_weight": 1.0})

    cont_weight = loss_weights["cont_weight"]
    cat_weight = loss_weights["cat_weight"]

    print(f"[INFO] Using loss weights - Continuous: {cont_weight:.2f}, Categorical: {cat_weight:.2f}")

    # --------------------
    # Run anomaly detection
    # --------------------
    results = apply_model(
        model=model,
        X_cont=Xc_test_scald,
        X_cat=Xk_test,
        method=method,
        device=cfg.device,
        cont_weight=cont_weight,
        cat_weight=cat_weight
    )

    errors = results["errors"]
    threshold = results["threshold"]
    mask = results["mask"]
    starts = results["anomaly_starts"]
    ends = results["anomaly_ends"]

    # -------------------------
    # Print summary
    # -------------------------
    print(f"\n--- Result threshold method: {method} ---")
    print(f"Total test samples = {len(errors)}")
    print(f"Threshold = {threshold:.6f}")
    print(f"Detected anomalous samples = {mask.sum()}")
    print(f"Detected anomaly intervals =  {len(starts)}\n")

    for s, e in zip(starts, ends):
        print(f"  > Interval {ts_eval[s]} - {ts_eval[e]} ({e-s} anomalies)")

    print(f"\nError Statistics:")
    print(f"Min:      {errors.min():.6f}")
    print(f"Mean:     {errors.mean():.6f}")
    print(f"Median:   {np.median(errors):.6f}")
    print(f"Max:      {errors.max():.6f}")
    print(f"Std:      {errors.std():.6f}")
    print(f"99th pct: {np.percentile(errors, 99):.6f}")    

    # --------------------
    # Save errors CSV
    # --------------------
    df_err = pd.DataFrame({
        "ts": ts_eval,
        "error": errors,
        "is_anomaly": mask.astype(int),
        "threshold": threshold,
    })

    err_path = OUT_DIR / f"{country}_errors_{method}.csv"
    df_err.to_csv(err_path, index=False)
    print(f"[OK] Saved error series to {err_path}")

    # --------------------
    # Save intervals
    # --------------------
    df_int = pd.DataFrame({
        "start_ts": ts_eval[starts] if len(starts) else [],
        "end_ts": ts_eval[ends - 1] if len(ends) else [],
        "duration_samples": (ends - starts)
    })
    int_path = OUT_DIR / f"{country}_intervals_{method}.csv"
    df_int.to_csv(int_path, index=False)
    print(f"[OK] Saved intervals to {int_path}")

    # --------------------
    # Save threshold
    # --------------------
    thr_path = OUT_DIR / f"{country}_threshold_{method}.json"
    threshold_data = {
        "country": country,
        "method": method,
        "threshold": float(threshold),
        "loss_weights": loss_weights,
        "test_samples": len(errors),
        "anomaly_count": int(mask.sum()),
        "interval_count": len(starts),
        "error_stats": {
            "min": float(errors.min()),
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "max": float(errors.max()),
            "std": float(errors.std()),
            "p99": float(np.percentile(errors, 99)),
        },
        "test_period": {
            "start": str(ts_eval[0].date()),
            "end": str(ts_eval[-1].date()),
        },
    }
    with open(thr_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"[OK] Saved threshold to {thr_path}")


def test_all(method: str, tr: int, vr: int):
    for c in COUNTRIES:
        try:
            test_country(c, method=method, tr=tr, vr=vr)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test model for single country or all countries."
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
        "target",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    if args.target.lower() == "all":
        test_all(
            method=args.method, 
            tr=args.tr,
            vr=args.vr
        )
    else:
        test_country(
            args.target.upper(), 
            method=args.method, 
            tr=args.tr,
            vr=args.vr
        )
