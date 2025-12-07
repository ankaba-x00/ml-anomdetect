#!/usr/bin/env python3
"""
Train a autoencoder for one or all countries:
- produces continuous + categorical feature matrices
- applies scaling to continuous features
- builds autoencoder configuration (AEConfig)
- trains TabularAE with early stopping
- performs latent space analysis if specified

Outputs: 
    PATH without -F: results/ml/trained/<MODEL>
    PATH with -F: app/deployment/models/<MODEL
    FILES: <COUNTRY_CODE>_autoencoder.pt, <COUNTRY_CODE>_scaler_cont.pkl, <COUNTRY_CODE>_cat_dims.json, <COUNTRY_CODE>_num_cont.json, <COUNTRY_CODE>_training_history.json, <COUNTRY_CODE>_latent_space_pca_coords.csv, <COUNTRY_CODE>_latent_space.png

Usage:
    python -m app.src.pipelines.train_model [-tr <int>] [-vr <int>] [-F] [-M <p99|p995|mad>] [-L] <MODEL> <COUNTRY_CODE|all> [| tee stdout_train.txt]
"""

import json, pickle
from pathlib import Path
import numpy as np
from typing import Optional
from sklearn.preprocessing import RobustScaler
from app.src.data import timeseries_seq_split
from app.src.data.feature_engineering import COUNTRIES, build_feature_matrix
from app.src.ml.training.calibrate import calibrate_threshold
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.analysis.analysis import plot_latent_space
from app.src.ml.training.train import train_autoencoder, save_autoencoder


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
FULL_OUT_DIR = PROJECT_ROOT / "app" / "deployment" / "models"
BEST_MODELS_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"


#########################################
##                 RUN                 ##
#########################################

def train_country(
        ae_type: str,
        country: str, 
        tr: int, 
        vr: int, 
        full: bool, 
        method: str, 
        cw: int,
        latent: bool,
        loss_weights: Optional[dict] = None,
    ):

    print(f"\n==============================")
    print(f"  TRAIN AUTOENCODER ({country})")
    print(f"==============================")
    print(f"[INFO] Model {ae_type.upper()} selected")
    
    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    X_cont, X_cat, num_cont, cat_dims = build_feature_matrix(country)
    Xc_np = X_cont.values.astype(np.float64)
    Xk_np = X_cat.values.astype(np.int64)
      
    # ------------------------------------
    # AEConfig object: Load or construct
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Reading {ae_type.upper()}Config from best tuning run.")
        tuned_cfg_path = BEST_MODELS_DIR / f"{ae_type.upper()}" / f"{country}_best_config.json"
        tuned_params_path = BEST_MODELS_DIR / f"{ae_type.upper()}" / f"{country}_best_params.json"
        if not tuned_cfg_path.exists():
            raise FileNotFoundError("[ERROR] Best config not found. Run tuning first.")
        if not tuned_params_path.exists():
            raise FileNotFoundError("[ERROR] Best params not found. Run tuning first.")
        with open(tuned_cfg_path, "r") as f:
            cfg_dict = json.load(f)
            if ae_type == "ae":
                cfg = AEConfig(**cfg_dict)
            elif ae_type == "vae":
                cfg = VAEConfig(**cfg_dict)
        with open(tuned_params_path, "r") as f:
            best_params = json.load(f)
        try:
            loss_weights = {
                "cont_weight": best_params.get("cont_weight", 1.0),
                "cat_weight": best_params.get("cat_weight", 0.0),
            }
            print(f"[INFO] Using tuned loss weights: {loss_weights}")
        except Exception:
            loss_weights = {"cont_weight": 1.0, "cat_weight": 0.0}
            print(f"[INFO] Using default loss weights: {loss_weights}")

    else:
        print(f"[INFO] Constructing {ae_type.upper()}Config from inital params.")
        base_cfg = dict(
            num_cont=num_cont,
            cat_dims=cat_dims,
            latent_dim=16,
            hidden_dims=(128, 64),
            dropout=0.1,
            embedding_dim=12,
            continuous_noise_std=0.01,
            residual_strength=0.10,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=256,
            num_epochs=60,
            patience=6,
            gradient_clip=1.0,
            optimizer="adam",
            lr_scheduler="plateau",
            use_lr_scheduler=True,
            anomaly_threshold=None,
            activation="relu",
            temperature=1.0
        )
        config_map = {
            "ae": AEConfig,
            "vae": VAEConfig
        }
        cfg = config_map[ae_type](**base_cfg)
            
    
    # Default loss weights if not provided
    if loss_weights is None:
        loss_weights = {"cont_weight": 1.0, "cat_weight": 0.0}

    # ------------------------------------
    # Full OR Split : Train model
    # ------------------------------------
    if full or tr == 100:
        # ------------------------------------
        # Fit scaler on cont features and tranform data
        # ------------------------------------
        scaler = RobustScaler()
        Xc_np_scald = scaler.fit_transform(Xc_np).astype(np.float32)

        # ------------------------------------
        # Train model
        # ------------------------------------
        print(f"[INFO] Dataset not split: 100% train.")
        model, history = train_autoencoder(
            Xc_np_scald, Xk_np,
            None, None,
            cfg,
            loss_weights
        )
        out_path = FULL_OUT_DIR / f"{ae_type.upper()}"
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        # ------------------------------------
        # Split dataset
        # ------------------------------------
        print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test.")
        (train_cont, train_cat), (val_cont, val_cat), _ = timeseries_seq_split(
            Xc_np, Xk_np,
            train_ratio=tr/100,
            val_ratio=vr/100,
        )

        # ------------------------------------
        # Fit scaler on cont features and tranform data
        # ------------------------------------
        scaler = RobustScaler()
        train_cont_scald = scaler.fit_transform(train_cont).astype(np.float32)
        val_cont_scald = scaler.transform(val_cont).astype(np.float32)

        # ------------------------------------
        # Train model
        # ------------------------------------
        model, history = train_autoencoder(
            train_cont_scald, train_cat, 
            val_cont_scald, val_cat, 
            cfg,
            loss_weights
        )
        out_path = OUT_DIR / f"{ae_type.upper()}"
        out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / f"{country}_autoencoder.pt"
    save_autoencoder(
        model=model, 
        config=cfg, 
        path=model_path,
        additional_info={
            "country": country,
            "training_mode": "full" if full else "split",
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(Xc_np),
        }
    )

    scaler_path = out_path / f"{country}_scaler_cont.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved continuous scaler to {scaler_path}")
    
    cat_path = out_path / f"{country}_cat_dims.json"
    with open(cat_path, "w") as f:
        json.dump(cat_dims, f, indent=2)
    print(f"[OK] Saved categorical vocab sizes to {cat_path}")

    num_path = out_path / f"{country}_num_cont.json"
    with open(num_path, "w") as f:
        json.dump({"num_cont": num_cont}, f, indent=2)
    print(f"[OK] Saved num_cont to {num_path}")

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
            train_cont_scald, train_cat, = Xc_np_scald, Xk_np
        plot_latent_space(
            country, 
            train_cont_scald, 
            train_cat,
            model,
            cfg.device,
            1000,
            out_path,
            f"{country}_latent_space.png"
        )

    print(f"[DONE] Trained model for {country}")

    # ------------------------------------
    # Error and threshold on calibration window
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Computing threshold on calibration window...")
        
        threshold_dict, _ = calibrate_threshold(
            country, 
            model, 
            scaler, 
            cfg.device, 
            method, 
            cw,
            cont_weight=loss_weights["cont_weight"],
            cat_weight=loss_weights["cat_weight"],
            tune_temperature=True,
            temperature_range=[0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
        )

        thr_path = out_path / f"{country}_cal_threshold.json"
        with open(thr_path, "w") as f:
            json.dump(threshold_dict, f, indent=2)
        print(f"[OK] Saved threshold to {thr_path}")

        print(f"[DONE] Preparation for inference model for {country}")


def train_all(ae_type: str, tr: int, vr: int, full: bool, method: str, cw: int, latent: bool):
    for c in COUNTRIES:
        try:
            train_country(ae_type, c, tr, vr, full, method, cw, latent)
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")
    print(f"\n[DONE] All model trainings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model for single country or all countries."
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
        "-M", "--method",
        choices=["p99", "p995", "mad"],
        default="p99",
        help="threshold method [default: p99]"
    )

    parser.add_argument(
        "-CW", "--calwindow",
        type=int,
        default=30,
        help="calibration window in days for computing threshold [default: 30]"
    )

    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="generate latent space plot after training"
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
   
    target = args.target
    ae_type = args.model.lower() 
    if ae_type not in ["ae", "vae"]:
        parser.print_help()
        print(f"[Error] Model can either be ae or vae!")
        exit(1)

    if target.lower() == "all":
        train_all(ae_type, args.tr, args.vr, args.full, args.method, args.calwindow, args.latent)
    else:
        train_country(ae_type, target.upper(), args.tr, args.vr, args.full, args.method, args.calwindow, args.latent)
