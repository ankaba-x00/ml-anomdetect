#!/usr/bin/env python3
"""
Train multi-task attack predictor for one or all countries to predict L3 and L7 intensity and classify attack types:
- loads feature matrices and labels
- build model configuration (MTEConfig)
- trains model (TrafficAttackPredictor)
- performs latent space analysis if specified

Outputs:
    PATH without -F: results/mt_ml/trained
    PATH with -F: app/deployment/models/MTP
    FILES: <COUNTRY>_model.pt, <COUNTRY>_scaler.pkl, <COUNTRY>_training_history.json, <COUNTRY>_latent_space_pca_coords.csv, <COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.mt.train_multitask_model [-tr <int>] [-vr <int>] [-F] [-L] <COUNTRY|all>
"""

import json, pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.preprocessing import RobustScaler
from app.src.data.split import timeseries_seq_split
from app.src.data.feature_engineering import COUNTRIES
from app.src.data.feature_engineering import load_supervised_feature_matrix
from app.src.ml.models.mte import MTEConfig
from app.src.ml.analysis.analysis import plot_latent_space
from app.src.ml.training.train_mt import train_multitask_model, save_multitask_model


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "mt_ml" / "trained"
BEST_MODELS_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tuned"
FULL_OUT_DIR = PROJECT_ROOT / "app" / "deployment" / "models" / "MTP"


#########################################
##                 RUN                 ##
#########################################

def train_country(
    country: str,
    tr: int,
    vr: int,
    full: bool,
    latent: bool,
    loss_weights: Optional[dict] = None,
):

    print(f"\n==============================")
    print(f"   TRAIN MT MODEL ({country}) ")
    print(f"==============================")

    # ------------------------------------
    # Load supervised feature matrix
    # ------------------------------------ 
    X_cont, X_cat, y_l3, y_l7, y_attack, num_cont, cat_dims = load_supervised_feature_matrix(country)
    Xc = X_cont.values.astype(np.float64)
    Xk = X_cat.values.astype(np.int64)
    y3 = y_l3.values.astype(np.float32)
    y7 = y_l7.values.astype(np.float32)
    ya = y_attack.values.astype(np.int64)

    # ------------------------------------
    # MTEConfig  object: Load or construct
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Reading MTEConfig from best tuning run.")
        tuned_cfg_path = BEST_MODELS_DIR / f"{country}_best_config.json"
        tuned_params_path = BEST_MODELS_DIR / f"{country}_best_params.json"
        if not tuned_cfg_path.exists():
            raise FileNotFoundError("[ERROR] Best config not found. Run tuning first.")
        if not tuned_params_path.exists():
            raise FileNotFoundError("[ERROR] Best params not found. Run tuning first.")
        with open(tuned_cfg_path, "r") as f:
            cfg_dict = json.load(f)
            cfg = MTEConfig(**cfg_dict)
        with open(tuned_params_path, "r") as f:
            best_params = json.load(f)
        try:
            loss_weights = {
                "l3": best_params.get("l3", 1.0),
                "l7": best_params.get("l7", 1.0),
                "attack": best_params.get("attack", 1.0),
            }
            print(f"[INFO] Using tuned loss weights: {loss_weights}")
        except Exception:
            loss_weights = {
                "l3": 1.0,
                "l7": 1.0,
                "attack": 1.0,
            }
            print(f"[INFO] Using default loss weights: {loss_weights}")
    else:
        print(f"[INFO] Constructing MTEConfig from inital params.")
        cfg = MTEConfig(
            num_cont=num_cont,
            cat_dims=cat_dims,
            n_attack_types=8,
            hidden_dims=(128, 64),
            latent_dim=32,
            dropout=0.1,
            head_hidden_dim=32,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            num_epochs=60,
            patience=6,
            gradient_clip=1.0,
            use_lr_scheduler=True,
            device="cpu",
        )

    if loss_weights is None:
        loss_weights = {
            "l3": 1.0,
            "l7": 1.0,
            "attack": 1.0,
        }

    # ------------------------------------
    # Full OR split
    # ------------------------------------
    if full or tr == 100:
        scaler = RobustScaler()
        Xc_scald = scaler.fit_transform(Xc).astype(np.float32)

        print("[INFO] Training on full dataset")
        model, history = train_multitask_model(
            Xc_scald, Xk, y3, y7, ya,
            None, None, None, None, None,
            cfg,
            loss_weights,
        )
        out_path = FULL_OUT_DIR
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test.")
        (Xc_tr, Xk_tr), (Xc_val, Xk_val), _ = timeseries_seq_split(
            Xc, Xk,
            train_ratio=tr/100,
            val_ratio=vr/100,
        )
        y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
        y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
        ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

        scaler = RobustScaler()
        Xc_tr_scald = scaler.fit_transform(Xc_tr).astype(np.float32)
        Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

        model, history = train_multitask_model(
            Xc_tr_scald, Xk_tr, y3_tr, y7_tr, ya_tr,
            Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
            cfg,
            loss_weights,
        )
        out_path = OUT_DIR
        out_path.mkdir(parents=True, exist_ok=True)
    
    model_path = out_path / f"{country}_multitask_model.pt"
    if full:
        tr, vr = 100, 0
    save_multitask_model(
        model=model, 
        config=cfg, 
        cat_dims=cat_dims,
        num_cont=num_cont,
        path=model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(X_cont),
        }
    )
    scaler_path = out_path / f"{country}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved continuous scaler to {scaler_path}")

    history_path = out_path / f"{country}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space visualization...")
        if full or tr == 100:
            Xc_tr_scald, Xk_tr = Xc_scald, Xk
        else: 
            out_path = OUT_DIR / "analysis" / country
            out_path.mkdir(parents=True, exist_ok=True)
        plot_latent_space(
            country, 
            Xc_tr_scald, 
            Xk_tr,
            model,
            cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png"
        )

    print(f"[DONE] Trained model for {country}")


def train_all(tr: int, vr: int, full: bool, latent: bool):
    for c in COUNTRIES:
        try:
            train_country(c, tr, vr, full, latent)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print("\n[DONE] All multi-task trainings completed!")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train multi-task attack predictor for single or all countries"
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
        "-F", "--full",
        action="store_true",
        help="train on full dataset, no validation (for inference); overwrites tr and vr"
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
        train_all(args.tr, args.vr, args.full, args.latent)
    else:
        train_country(
            args.target.upper(),
            args.tr,
            args.vr,
            args.full,
            args.latent,
        )
