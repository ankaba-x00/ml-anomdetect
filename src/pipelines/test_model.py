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
    python -m src.pipelines.detect_anomalies US --method p99
    python -m src.pipelines.detect_anomalies all --method mad --full
"""

import pickle, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.feature_engineering import build_feature_matrix, COUNTRIES
from src.models.train import load_autoencoder
from src.models.evaluate import apply_model
from src.data.split import timeseries_seq_split


#########################################
##               PATHS                 ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]

TRAINED_DIR = PROJECT_ROOT / "results" / "models" / "trained"
TUNED_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
OUT_DIR = PROJECT_ROOT / "results" / "models" / "tested"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def test_country(
    country: str,
    method: str = "p99",
    use_full_data: bool = False,
):
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

    # scaler provides num_cont for splitting
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(catdims_path, "r") as f:
        cat_dims = json.load(f)

    device = torch.device(cfg.device)

    # --------------------
    # Load feature matrix
    # --------------------
    X_cont_df, X_cat_df, _, cat_dims2, _ = build_feature_matrix(country, scaler=scaler)

    # ensure consistent categorical structure
    assert cat_dims2 == cat_dims, "[Error] Saved cat_dims mismatch — rebuild features."

    X_cont = X_cont_df.values.astype(np.float32)
    X_cat = X_cat_df.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split or full dataset
    # --------------------
    if use_full_data:
        print(f"[INFO] Using FULL dataset ({len(X_cont)} samples).")
        X_eval_cont = X_cont
        X_eval_cat = X_cat
        ts_eval = ts
    else:
        (Xc_train, _), (Xc_val, _), (Xc_test, Xk_test) = timeseries_seq_split(
            X_cont, X_cat,
            train_ratio=0.75,
            val_ratio=0.15
        )
        X_eval_cont = Xc_test
        X_eval_cat = Xk_test
        ts_eval = ts[len(Xc_train)+len(Xc_val):]

    # --------------------
    # Run anomaly detection
    # --------------------
    results = apply_model(
        model=model,
        X_cont=X_eval_cont,
        X_cat=X_eval_cat,
        method=method,
        device=cfg.device,
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
    print(f"Threshold = {threshold:.6f}")
    print(f"Detected {mask.sum()} anomalous samples")
    print(f"Detected {len(starts)} anomaly intervals\n")

    for s, e in zip(starts, ends):
        print(f"  > Interval {ts_eval[s]} - {ts_eval[e-1]}  ({e-s} samples)")

    # --------------------
    # Save errors CSV
    # --------------------
    df_err = pd.DataFrame({
        "ts": ts_eval,
        "error": errors,
        "is_anomaly": mask.astype(int),
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
    with open(thr_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    print(f"[OK] Saved threshold to {thr_path}")


def test_all(method: str, use_full_data: bool):
    for c in COUNTRIES:
        try:
            test_country(c, method=method, use_full_data=use_full_data)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test model for single country or all countries."
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    parser.add_argument(
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method [default: p99]"
    )

    parser.add_argument(
        "-F", "--full",
        action="store_true",
        help="detect anomalies on full dataset instead of test split"
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        exit(1)

    if args.target.lower() == "all":
        test_all(method=args.method, use_full_data=args.full)
    else:
        test_country(args.target.upper(), method=args.method, use_full_data=args.full)
