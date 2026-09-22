#!/usr/bin/env python3
"""
Test autoencoder for one or multiple countries
A) AE/VAE/MTAE to predict traffic anomalies
B) MTAE to predict L3/L7 intensities and attack types
- loads model and features
- computes reconstruction errors and identifies anomaly intervals
- prints summary stats
- (optional) performs latent space analysis

Output:
    PATH: results/ml/tested/<MODEL>
    FILES : <COUNTRY>_scores_<method>.csv
            <COUNTRY>_threshold_<method>.json
            <COUNTRY>_intervals_<method>.csv
            analysis/<COUNTRY>_latent_space_pca_coords.csv
            analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.test_model [-tr <int>] [-vr <int>] [-M <p99|p995|mad>] [-L] <MODEL> <COUNTRY|all>
"""

import json, pickle, torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from typing import Literal

from app.src.data.building import ID_TO_ATTACK
from app.src.data.fetching import ISO_3166_alpha2
from app.src.data.processing import timeseries_seq_split
from app.src.data.building.feature_engineering import (
    COUNTRIES,
    load_feature_matrix, 
    load_supervised_feature_matrix
)
from app.src.ml.analysis import plot_latent_space
from app.src.ml.models.ae import TabularAE
from app.src.ml.models.configs import MTAEConfig
from app.src.ml.models.helpers import load_autoencoder
from app.src.ml.models.vae import TabularVAE
from app.src.ml.training.anomaly_utils import get_threshold
from app.src.ml.training.evaluate_ae import apply_model
from app.src.ml.training.evaluate_mt import apply_mt_model


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
TRAINED_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "tested"


def test_model(
    ae_type: Literal["ae", "vae", "mtae"], 
    country: str, 
    method: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
    """Runs testing pipeline for selected model and country."""
    print(f"\n==============================")
    print(f"     TEST {country} MODEL     ")
    print(f"==============================")
    print(f"[INFO] Model {ae_type.upper()} selected")

    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = TRAINED_DIR / f"{ae_type.upper()}" / f"{country}_autoencoder.pt"
    scaler_path = TRAINED_DIR / f"{ae_type.upper()}" / f"{country}_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[Error] Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"[Error] Scaler not found: {scaler_path}")

    # ------------------------
    # Load model, weights, config, scaler
    # ------------------------
    model_bundle = load_autoencoder(model_path)

    if model_bundle.metadata is not None:
        loss_weights = model_bundle.metadata["loss_weights"]
        if ae_type in ["mtae"]:
            attack_type_weights = torch.Tensor(model_bundle.metadata["attack_type_weights"])

    with open(scaler_path, "rb") as f:
        scaler: RobustScaler = pickle.load(f)

    # ------------------------
    # Load feature matrix
    # ------------------------
    if ae_type in ["ae", "vae"]:
        fmatrix = load_feature_matrix(country)
    else:
        fmatrix = (
            load_supervised_feature_matrix(country)
        )

    assert model_bundle.num_cont == fmatrix.num_cont, "[ERROR] num_cont mismatch between scaler and feature matrix"
    assert model_bundle.cat_dims == fmatrix.cat_dims, "[ERROR] cant_dims mismatch between scaler and feature matrix"
    
    Xc = fmatrix.X_cont.to_numpy(dtype=np.float32)
    Xk = fmatrix.X_cat.to_numpy(dtype=np.int64)
    ts = pd.to_datetime(fmatrix.X_cont.index)
    if (
        ae_type in ["mtae"]
        and fmatrix.y_l3 is not None
        and fmatrix.y_l7 is not None
        and fmatrix.y_at is not None
    ):
        y3 = fmatrix.y_l3.to_numpy(dtype=np.float32)
        y7 = fmatrix.y_l7.to_numpy(dtype=np.float32)
        ya = fmatrix.y_at.to_numpy(dtype=np.int64)

    # ------------------------
    # Split dataset
    # ------------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    [Xc_tr, _], _, [Xc_te, Xk_te] = timeseries_seq_split(
        [Xc, Xk],
        tr/100,
        vr/100
    )
    ts_te = ts[len(Xc_tr) + int(len(ts) * vr/100):]

    if ae_type in ["mtae"]:
        _, _, [y3_te] = timeseries_seq_split([y3], tr/100, vr/100)
        _, _, [y7_te] = timeseries_seq_split([y7], tr/100, vr/100)
        _, _, [ya_te] = timeseries_seq_split([ya], tr/100, vr/100)

    # ------------------------
    # Scale cont features
    # ------------------------
    Xc_te_scald = scaler.transform(Xc_te).astype(np.float32)

    # ------------------------
    # Apply model
    # ------------------------
    if isinstance(model_bundle.model, (TabularAE, TabularVAE)):
        result = apply_model(
            model=model_bundle.model,
            X_cont=Xc_te_scald,
            X_cat=Xk_te,
            loss_weights=loss_weights,
            device=model_bundle.cfg.device,
            method=method,
            temperature=model_bundle.cfg.temperature
        )
    else:
        # ------------------------
        # Set quantiles
        # ------------------------
        if isinstance(model_bundle.cfg, MTAEConfig):
            med_q = model_bundle.cfg.quantiles.index(0.5)
            pred_quantiles = model_bundle.cfg.quantiles[med_q:]
            print(f"[INFO] Quantiles used for prediction: {pred_quantiles}")

        result = apply_mt_model(
            model=model_bundle.model,
            X_cont=Xc_te_scald,
            X_cat=Xk_te,
            y_l3=y3_te,
            y_l7=y7_te,
            y_attack=ya_te,
            loss_weights=loss_weights,
            attack_type_weights=attack_type_weights,
            pred_quantiles=pred_quantiles,
            method=method,
            min_length=1,
            merge_gap=0,
            device=model_bundle.cfg.device
        )

        l3_preds = result["l7_pred_0.5"]
        l7_preds = result["l7_pred_0.5"]
        at_preds = result["at_pred"]

    # ------------------------
    # Print summary
    # ------------------------
    print("\n--- Test Summary ---")
    print(f"Total samples = {len(result.scores)}")
    print(f"Threshold = {result.threshold:.6f}")
    print("Quantile: 0.5")
    print(f"Flagged samples = {result.mask.sum()}")
    print(f"Flagged intervals = {len(result.anom_starts)}\n")

    for s, e in zip(result.anom_starts, result.anom_ends):
        start = ts_te[s].strftime("%d/%m/%y,%H:%M")
        end = ts_te[e-1].strftime("%d/%m/%y,%H:%M")
        interval_str = f"{start} - {end} ({e-s} anomalies)"
        print(f"  > Interval {interval_str}")
        if ae_type in ["mtae"]:
            print(
                f"\tl3: {l3_preds[s:e]}"
                f"\n\tl7: {l7_preds[s:e]}"
                f"\n\ttype: {[ID_TO_ATTACK[t] for t in at_preds[s:e]]}"
            )

    print(f"\nScore Statistics:")
    print(f"Min:       {result.scores.min():.6f}")
    print(f"Max:       {result.scores.max():.6f}")
    print(f"Mean:      {result.scores.mean():.6f}")
    print(f"Std:       {result.scores.std():.6f}")
    print(f"Score {method}: {get_threshold(method, result.scores):4f}\n")

    # ------------------------
    # Save artefacts
    # ------------------------
    df_out = pd.DataFrame({
        "ts": ts_te,
        "scores": result.scores,
        "threshold": result.threshold,
        "is_flagged": result.mask.astype(int),
    })

    if ae_type in ["mtae"]:
        mt_dict_out = {
            "loss_l3": result["loss_l3"],
            "loss_l7": result["loss_l7"],
            "loss_at": result["loss_at"],
            "loss_total": result["loss_total"],
            "l3_true": y3_te,
            "l7_true": y7_te,
            "at_true": ya_te,
            "at_pred": result["at_pred"],
            "at_conf": result["at_conf"]
        }
        for q in pred_quantiles:
            mt_dict_out[f"l3_pred_{q}"] = result[f"l3_pred_{q}"]
            mt_dict_out[f"l7_pred_{q}"] = result[f"l7_pred_{q}"]
        df_out = pd.concat([df_out, pd.DataFrame(mt_dict_out)], axis=1)

    score_path = out_path / f"{country}_scores_{method}.csv"
    df_out.to_csv(score_path, index=False)
    print(f"[OK] Saved test scores to {score_path}")

    df_int = pd.DataFrame({
        "start_idx": result.anom_starts,
        "end_idx": result.anom_ends,
        "start_ts": ts_te[result.anom_starts] if len(result.anom_starts) else [],
        "end_ts": ts_te[result.anom_ends - 1] if len(result.anom_ends) else [],
        "duration_samples": result.anom_ends - result.anom_starts
    })
    int_path = out_path / f"{country}_intervals_{method}.csv"
    df_int.to_csv(int_path, index=False)
    print(f"[OK] Saved intervals to {int_path}")

    thr_path = out_path / f"{country}_threshold_{method}.json"
    threshold_data = {
        "country": country,
        "train_ratio": tr,
        "val_ratio": vr,
        "method": method,
        "threshold": float(result.threshold),
        "loss_weights": loss_weights,
        "test_samples": len(result.scores),
        "anomaly_count": int(result.mask.sum()),
        "interval_count": len(result.anom_starts),
        "error_stats": {
            "min": float(result.scores.min()),
            "mean": float(result.scores.mean()),
            "median": float(np.median(result.scores)),
            "max": float(result.scores.max()),
            "std": float(result.scores.std()),
            "p995": float(np.percentile(result.scores, 99.5)),
        },
        "test_period": {
            "start": str(ts_te[0].date()),
            "end": str(ts_te[-1].date()),
        },
    }
    if ae_type in ["mtae"]:
        threshold_data["attack_type_weights"] = attack_type_weights.tolist()

    with open(thr_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"[OK] Saved threshold to {thr_path}")

    # ------------------------
    # Visualize latent space
    # ------------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        out_path = out_path / "analysis" / country
        out_path.mkdir(parents=True, exist_ok=True)

        plot_latent_space(
            country, 
            Xc_te_scald,
            Xk_te,
            model_bundle.model,
            model_bundle.cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png"
        )

    print(f"[DONE] Tested model for {country}")

def test_all(
    ae_type: Literal["ae", "vae", "mtae"], 
    method: str, 
    tr: int, 
    vr: int, 
    latent: bool
) -> None:
    """
    Runs testing pipeline for selected model and all pre-defined countries.
    """
    for c in COUNTRIES:
        try:
            test_model(
                ae_type, 
                c, 
                method, 
                tr, 
                vr, 
                latent
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")

    print(f"\n[DONE] All testings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test single- or multi-task autoencoder for one or multiple countries"
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
        test_all(
            ae_type,
            args.method, 
            args.tr,
            args.vr,
            args.latent
        )
    else:
        test_model(
            ae_type,
            target, 
            args.method, 
            args.tr,
            args.vr,
            args.latent
        )
