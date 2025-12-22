#!/usr/bin/env python3
"""
Test model to detect anomalies in new data:
- loads model and features
- computes reconstruction errors
- computes threshold of choice (p99, p995, MAD)
- identifies anomaly intervals of certain min. sample length and sample gap 
- prints summary stats
- performs latent space analysis if specified

Outputs:
    PATH : results/ae_ml/tested/<MODEL>
    FILES : <COUNTRY>_errors_<method>.csv, 
            <COUNTRY>_threshold_<method>.json, 
            <COUNTRY>_intervals_<method>.csv, 
            analysis/<COUNTRY>_latent_space_pca_coords.csv, 
            analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.ae.test_model [-M <p99|p995|mad>] [-tr <int>] [-vr <int>] <MODEL> <COUNTRY|all>
"""

import pickle, json, torch
import numpy as np
import pandas as pd
from pathlib import Path

from app.src.data.feature_engineering import load_feature_matrix, COUNTRIES
from app.src.ml.training.train_ae import load_autoencoder
from app.src.ml.training.evaluate_ae import apply_model
from app.src.data.split import timeseries_seq_split
from app.src.ml.analysis import plot_latent_space


#########################################
##               PARAMS                ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
TUNED_DIR = PROJECT_ROOT / "results" / "ae_ml" / "tuned"
OUT_DIR = PROJECT_ROOT / "results" / "ae_ml" / "tested"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 RUN                 ##
#########################################

def test_country(
    ae_type: str, 
    country: str, 
    method: str, 
    tr: int = 75, 
    vr: int = 15, 
    use_mc_elbo: bool = False, 
    latent: bool = False
) -> None:
    print(f"\n==============================")
    print(f"  DETECT ANOMALIES ({country})")
    print(f"==============================")

    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load model + config, scaler
    # --------------------
    model_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_best_model.pt"
    scaler_path = TUNED_DIR / f"{ae_type.upper()}" / f"{country}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[Error] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[Error] Scaler not found: {scaler_path}")

    model, cfg, model_num_cont, model_cat_dims = load_autoencoder(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --------------------
    # Load feature matrix
    # --------------------
    X_cont_df, X_cat_df, num_cont, cat_dims = load_feature_matrix(country)

    # ensure consistent categorical structure
    assert model_cat_dims == cat_dims, "[Error] Saved cat_dims mismatch — rebuild features."
    assert model_num_cont == num_cont, "[Error] Saved num_cont mismatch — rebuild features."

    X_cont = X_cont_df.values.astype(np.float64)
    X_cat = X_cat_df.values.astype(np.int64)
    ts = X_cont_df.index

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, _), (Xc_val, _), (Xc_test, Xk_test) = timeseries_seq_split(
        X_cont, X_cat,
        tr/100,
        vr/100
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
    loss_weights = payload.get("additional_info", {}).get("loss_weights", {
        "cont_weight": 1.0, "cat_weight": 0.0
    })

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
        cat_weight=cat_weight,
        temperature=cfg.temperature, 
        use_mc_elbo=use_mc_elbo,
        beta=getattr(cfg, "beta", 1.0)
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
        print(f"  > Interval {ts_eval[s]} - {ts_eval[e-1]} ({e-s} anomalies)")

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

    err_path = out_path / f"{country}_errors_{method}.csv"
    df_err.to_csv(err_path, index=False)
    print(f"[OK] Saved error series to {err_path}")

    # --------------------
    # Save intervals
    # --------------------
    df_int = pd.DataFrame({
        "start_idx": starts,
        "end_idx": ends,
        "start_ts": ts_eval[starts] if len(starts) else [],
        "end_ts": ts_eval[ends - 1] if len(ends) else [],
        "duration_samples": ends - starts
    })
    int_path = out_path / f"{country}_intervals_{method}.csv"
    df_int.to_csv(int_path, index=False)
    print(f"[OK] Saved intervals to {int_path}")

    # --------------------
    # Save threshold
    # --------------------
    thr_path = out_path / f"{country}_threshold_{method}.json"
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

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = out_path / "analysis" / country
        out_path.mkdir(parents=True, exist_ok=True)
        plot_latent_space(
            country, 
            Xc_test_scald,
            Xk_test,
            model,
            cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png"
        )

    print(f"[DONE] Tested model for {country}")


def test_all(
    ae_type: str, 
    method: str, 
    tr: int, 
    vr: int, 
    use_mc_elbo: bool, 
    latent: bool
) -> None:
    for c in COUNTRIES:
        try:
            test_country(
                ae_type=ae_type, 
                country=c, 
                method=method, 
                tr=tr, 
                vr=vr, 
                use_mc_elbo=use_mc_elbo, 
                latent=latent
            )
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
        "-MC", "--MC-score",
        action="store_true",
        help="use Monte-Carlo scoring for reconstruction errors"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after tuning"
    )

    parser.add_argument(
        "model",
        help="model to train: ae, vae"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' to train US model, or 'all' to train all country models"
    )

    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)

    if args.target.lower() == "all":
        test_all(
            ae_type=ae_type,
            method=args.method, 
            tr=args.tr,
            vr=args.vr,
            use_mc_elbo=args.MC_score,
            latent=args.latent
        )
    else:
        test_country(
            ae_type=ae_type,
            country=args.target.upper(), 
            method=args.method, 
            tr=args.tr,
            vr=args.vr,
            use_mc_elbo=args.MC_score,
            latent=args.latent
        )
