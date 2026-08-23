#!/usr/bin/env python3
"""
Validate trained autoencoder for one or multiple countries:
A) AE/VAE/MTAE to predicts traffic anomalies
B) MTAE to predict L3/L7 intensities and attack types
- loads model, scaler, features
- prints summary stats
- (optional) performs latent space analysis

Output: 
    PATH : results/ml/validated/<MODEL>
    FILES : <COUNTRY>_validation.csv
            analysis/<COUNTRY>_latent_space_pca_coords.csv
            analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.validate_model [-tr <int>] [-vr <int>] [-L] <MODEL> <COUNTRY|all>
"""

import pickle, torch
import numpy as np
import pandas as pd
from pathlib import Path

from app.src.data import ID_TO_ATTACK, ISO_3166_alpha2, timeseries_seq_split
from app.src.data.feature_engineering import (
    COUNTRIES, 
    load_feature_matrix, 
    load_supervised_feature_matrix
)
from app.src.ml.analysis import plot_latent_space
from app.src.ml.models.helpers import load_autoencoder
from app.src.ml.training.anomaly_utils import get_threshold
from app.src.ml.training.evaluate_ae import apply_model
from app.src.ml.training.evaluate_mt import apply_mt_model


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TRAINED_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "validated"


def validate_model(
    ae_type: str, 
    country: str, 
    tr: int, 
    vr: int,
    method: str, 
    latent: bool
) -> None:
    print(f"\n==============================")
    print(f"    VALIDATE {country} MODEL  ")
    print(f"==============================")
    print(f"[INFO] Model {ae_type.upper()} selected")

    # --------------------
    # Set path params
    # --------------------
    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = TRAINED_DIR / f"{ae_type.upper()}" / f"{country}_autoencoder.pt"
    scaler_path = TRAINED_DIR / f"{ae_type.upper()}" / f"{country}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[ERROR] Scaler not found: {scaler_path}")

    # --------------------
    # Load model, weights, config, scaler
    # --------------------
    model, cfg, model_num_cont, model_cat_dims, metadata = load_autoencoder(model_path, metadata=True)

    try:
        loss_weights = metadata["loss_weights"]
        if ae_type in ["mtae"]:
            attack_type_weights = torch.Tensor(metadata["attack_type_weights"])
    except KeyError:
        print("[ERROR] No weights saved in model metadata")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --------------------
    # Load feature matrix
    # --------------------
    if ae_type in ["ae", "vae"]:
        X_cont, X_cat, num_cont, cat_dims = load_feature_matrix(country)
    else:
        X_cont, X_cat, y_l3, y_l7, y_at, num_cont, cat_dims = (
            load_supervised_feature_matrix(country)
        )

    assert model_num_cont == num_cont, "[ERROR] num_cont mismatch between scaler and feature matrix"
    assert model_cat_dims == cat_dims, "[ERROR] cant_dims mismatch between scaler and feature matrix"

    Xc = X_cont.values.astype(np.float32)
    Xk = X_cat.values.astype(np.int64)
    ts = X_cont.index
    if ae_type in ["mtae"]:
        y3 = y_l3.values.astype(np.float32)
        y7 = y_l7.values.astype(np.float32)
        ya = y_at.values.astype(np.int64)

    # --------------------
    # Split dataset
    # --------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_tr, _), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk,
        tr/100,
        vr/100,
    )
    ts_val = ts[len(Xc_tr): len(Xc_tr) + len(Xc_val)]

    if ae_type in ["mtae"]:
        _, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
        _, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
        _, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)
    
    # --------------------
    # Scale cont features
    # --------------------
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    # --------------------
    # Apply model
    # --------------------
    if ae_type in ["ae", "vae"]:
        result = apply_model(
            model=model,
            X_cont=Xc_val_scald,
            X_cat=Xk_val,
            loss_weights=loss_weights,
            device=cfg.device,
            method=method,
            temperature=cfg.temperature,
            beta=getattr(cfg, "beta", 1.0),
        )
    else:
        # --------------------
        # Set quantiles
        # --------------------
        med_q = cfg.quantiles.index(0.5)
        pred_quantiles = cfg.quantiles[med_q:]
        print(f"[INFO] Quantiles used for prediction: {pred_quantiles}")

        result = apply_mt_model(
            model=model,
            X_cont=Xc_val_scald,
            X_cat=Xk_val,
            y_l3=y3_val,
            y_l7=y7_val,
            y_attack=ya_val,
            loss_weights=loss_weights,
            attack_type_weights=attack_type_weights,
            pred_quantiles=pred_quantiles,
            method=method,
            min_length=1,
            merge_gap=0,
            device=cfg.device
        )

    scores = result["scores"]
    threshold = result["threshold"]
    mask = result["mask"]
    starts = result["anomaly_starts"]
    ends = result["anomaly_ends"]

    if ae_type in ["mtae"]:
        l3_preds = result["l3_pred"][0.5]
        l7_preds = result["l7_pred"][0.5]
        at_preds = result["at_pred"]

    # --------------------
    # Print summary
    # --------------------
    print("\n--- Validation Summary ---")
    print(f"Total samples: {len(scores)}")
    print(f"Threshold ({method}): {threshold:4f}")
    print(f"Flagged samples: {int(mask.sum())}")
    print(f"Flagged intervals = {len(starts)}\n")

    for s, e in zip(starts, ends):
        interval_str = f"{ts_val[s].strftime('%d/%m/%y,%H:%M')} - {ts_val[e-1].strftime('%d/%m/%y,%H:%M')} ({e-s} anomalies)"
        print(f"  > Interval {interval_str}")
        if ae_type in ["mtae"]:
            print(
                f"\tl3: {l3_preds[s:e]}"
                f"\n\tl7: {l7_preds[s:e]}"
                f"\n\ttype: {[ID_TO_ATTACK[t] for t in at_preds[s:e]]}"
            )

    print(f"\nScore Statistics:")
    print(f"Min:       {scores.min():.6f}")
    print(f"Max:       {scores.max():.6f}")
    print(f"Mean:      {scores.mean():.6f}")
    print(f"Std:       {scores.std():.6f}")
    print(f"Score {method}: {get_threshold(method, scores):4f}\n")

    

    # ------------------------------------
    # Save artefacts
    # ------------------------------------
    df_out = pd.DataFrame({
        "ts": ts_val,
        "scores": scores,
        "threshold": threshold,
        "is_flagged": mask.astype(int),
    })

    if ae_type in ["mtae"]:
        mt_dict_out = {
            "loss_l3": result["loss_l3"],
            "loss_l7": result["loss_l7"],
            "loss_at": result["loss_at"],
            "loss_total": result["loss_total"],
            "at_true": ya_val,
            "at_pred": result["at_pred"],
            "at_conf": result["at_conf"],
            "l3_true": y3_val,
            "l7_true": y7_val,
        }
        for q in pred_quantiles:
            mt_dict_out[f"l3_pred_{q}"] = result["l3_pred"][q]
            mt_dict_out[f"l7_pred_{q}"] = result["l7_pred"][q]
        df_out = pd.concat([df_out, pd.DataFrame(mt_dict_out)], ignore_index=True)

    val_path = out_path / f"{country}_validation.csv"
    df_out.to_csv(val_path, index=False)
    print(f"[OK] Validation CSV saved to {val_path}")

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space analysis...")
        out_path = out_path / "analysis" / country
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
    ae_type: str, 
    tr: int, 
    vr: int, 
    method: str,
    latent: bool
) -> None:
    for c in COUNTRIES:
        try:
            validate_model(
                ae_type, 
                c, 
                tr, 
                vr, 
                method, 
                latent)
            
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")

    print(f"\n[DONE] All validations completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate single- or multi-task autoencoder for one or multiple countries"
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
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method [default: p99]"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="perform latent space analysis [default: false]"
    )

    parser.add_argument(
        "model",
        help="<ae|vae|mtae> model to train"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all> e.g. 'US' or 'all' for all countries defined in config/models.yaml"
    )

    args = parser.parse_args()

    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae", "mtae"]:
        parser.print_help()
        exit(1)

    target = args.target.upper()
    if target not in ISO_3166_alpha2 and target.lower() != "all":
        parser.print_help()
        print(f"[ERROR] Invalid target: {target}")
        exit(1)

    if target.lower() == "all":
        validate_all(
            ae_type,
            args.tr, 
            args.vr,
            args.method,
            args.latent
        )
    else:
        validate_model(
            ae_type, 
            target,
            args.tr, 
            args.vr, 
            args.method,
            args.latent
        )
