#!/usr/bin/env python3
"""
Train autoencoder for one or multiple countries
A) AE/VAE/MTAE to predict traffic anomalies
B) MTAE to predict L3/L7 intensities and attack types
- loads feature matrices
- applies scaling on cont features
- builds autoencoder configuration
- trains model with early stopping
- (optional) performs latent space analysis
- (with -F) prepares model and helper files for inference

Output: 
    PATH without -F: results/ml/trained/<MODEL>
    PATH with -F: app/deployment/models/<MODEL>
    FILES: <COUNTRY>_autoencoder.pt 
           <COUNTRY>_config.json
           <COUNTRY>_scaler.pkl 
           <COUNTRY>_training_history.json 
           <COUNTRY>_cal_threshold.json
           analysis/<COUNTRY>_latent_space_pca_coords.csv
           analysis/<COUNTRY>_latent_space.png

Usage:
    python -m app.src.pipelines.train_model [-tr <int>] [-vr <int>] [-F] [-M <p99|p995|mad>] [-CW] [-L] <CONFIG_PATH> <MODEL> <COUNTRY|all>
"""

import json, pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

from app.src.data import ISO_3166_alpha2, timeseries_seq_split
from app.src.data.feature_engineering import (
    COUNTRIES,
    load_feature_matrix, 
    load_supervised_feature_matrix
)
from app.src.ml.analysis import plot_latent_space
from app.src.ml.models.configs import AEConfig, VAEConfig, MTAEConfig
from app.src.ml.models.helpers import load_autoencoder, save_autoencoder
from app.src.ml.training.calibrate import calibrate_threshold
from app.src.ml.training.train_ae import train_autoencoder
from app.src.ml.training.train_mt import train_mt_autoencoder


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "app" / "src" / "config" / "train"
BEST_MODELS_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "trained"
FULL_OUT_DIR = PROJECT_ROOT / "app" / "deployment" / "models"


def train_model(
    ae_type: str,
    country: str, 
    config_path: str,
    tr: int, 
    vr: int,
    full: bool,
    latent: bool,
    method: str, 
    cw: int
) -> None:
    print(f"\n==============================")
    print(f"     TRAIN {country} MODEL    " )
    print(f"==============================")
    print(f"[INFO] Model {ae_type.upper()} selected")
    
    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    if ae_type in ["ae", "vae"]:
        X_cont, X_cat, num_cont, cat_dims = load_feature_matrix(country)
    else:
        X_cont, X_cat, y_l3, y_l7, y_at, num_cont, cat_dims = (
            load_supervised_feature_matrix(country)
        )
    Xc = X_cont.values.astype(np.float32)
    Xk = X_cat.values.astype(np.int64)
    if ae_type in ["mtae"]:
        y3 = y_l3.values.astype(np.float32)
        y7 = y_l7.values.astype(np.float32)
        ya = y_at.values.astype(np.int64)

    # ------------------------------------
    # Load Config
    # ------------------------------------
    if config_path == "config":
        cfg_path = CONFIG_DIR / f"config_{ae_type}.json"
    elif config_path == "tuned":
        print("[INFO] Reading config from tuning run...")
        retune_no = int(input("[SELECT] Which tuning run? [0 = base|int] "))
        tune_phase = "base" if retune_no == 0 else f"retune_{retune_no}"
        cfg_path = BEST_MODELS_DIR / f"{ae_type.upper()}" / f"{country}_best_config_{tune_phase}.json"
    if config_path == "trained":
        cfg_path = OUT_DIR / f"{ae_type.upper()}" / f"{country}_autoencoder.pt"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"[ERROR] Best config not found in {cfg_path}")
    print(f"[INFO] Reading config from {cfg_path}")
    
    if config_path in ["config", "tuned"]:
        config_map = {
            "ae": AEConfig,
            "vae": VAEConfig,
            "mtae": MTAEConfig
        }
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = config_map[ae_type](**cfg_dict)
    else:
        _, cfg, _, _ = load_autoencoder(cfg_path)

    loss_weights = {
        "cont_w": cfg.cont_w,
        "cat_w": cfg.cat_w
    }
    if ae_type in ["mtae"]:
        loss_weights["l3_w"] = cfg.lambda_l3
        loss_weights["l7_w"] = cfg.lambda_l7
        loss_weights["at_w"] = cfg.lambda_at

    # ------------------------------------
    # Training
    # ------------------------------------
    if full or tr == 100:
        print("[INFO] Training on full dataset")
        # ------------------------------------
        # Scale cont features
        # ------------------------------------
        scaler = RobustScaler()
        Xc_scald = scaler.fit_transform(Xc).astype(np.float32)

        # ------------------------------------
        # Train model
        # ------------------------------------
        if ae_type in ["ae", "vae"]:
            model, history = train_autoencoder(
                Xc_scald, Xk,
                None, None,
                cfg,
                loss_weights
            )
        else:
            model, history = train_mt_autoencoder(
            Xc_scald, Xk, y3, y7, ya,
            None, None, None, None, None,
            cfg,
            loss_weights,
        )

        out_path = FULL_OUT_DIR / f"{ae_type.upper()}"
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        # ------------------------------------
        # Split dataset
        # ------------------------------------
        print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
        (Xc_tr, Xk_tr), (Xc_val, Xk_val), _ = timeseries_seq_split(
            Xc, Xk,
            tr/100,
            vr/100,
        )

        if ae_type in ["mtae"]:
            y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
            y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
            ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

        # ------------------------------------
        # Scale cont features
        # ------------------------------------
        scaler = RobustScaler()
        Xc_tr_scald = scaler.fit_transform(Xc_tr).astype(np.float32)
        Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

        # ------------------------------------
        # Train model
        # ------------------------------------
        if ae_type in ["ae", "vae"]:
            model, history = train_autoencoder(
                Xc_tr_scald, Xk_tr, 
                Xc_val_scald, Xk_val, 
                cfg,
                loss_weights
            )
        else:
            model, history = train_mt_autoencoder(
                Xc_tr_scald, Xk_tr, y3_tr, y7_tr, ya_tr,
                Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
                cfg,
                loss_weights,
            )

        out_path = OUT_DIR / f"{ae_type.upper()}"
        out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # Save artefacts
    # ------------------------------------
    metadata = {
        "country": country,
        "train_ratio": 100 if full else tr,
        "val_ratio": 0 if full else vr,
        "loss_weights": loss_weights,
        "total_samples": len(Xc),
    }
    if ae_type in ["mtae"]:
        metadata["attack_type_weights"] = history["attack_type_weights"]
    
    model_path = out_path / f"{country}_autoencoder.pt"
    save_autoencoder(
        model=model, 
        config=cfg,
        cat_dims=cat_dims,
        num_cont=num_cont,
        path=model_path,
        metadata=metadata
    )

    scaler_path = out_path / f"{country}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved scaler to {scaler_path}")

    history_path = out_path / f"{country}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    # ------------------------------------
    # Visualize latent space
    # ------------------------------------
    if latent:
        print(f"[INFO] Preparing latent space analysis...")
        if full or tr == 100:
            Xc_tr_scald, Xk_tr = Xc_scald, Xk
        else: 
            out_path = out_path / "analysis" / country
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

    # ------------------------------------
    # Calculate threshold on calibration window
    # ------------------------------------
    if full or tr == 100:
        print(f"[INFO] Computing threshold on calibration window...")

        if ae_type in ["mte"]:
            attack_type_weights = history["attack_type_weights"]
        else:
            attack_type_weights = None

        threshold_dict, _ = calibrate_threshold(
            country=country, 
            model=model, 
            scaler=scaler,
            device=cfg.device,
            loss_weights=loss_weights,
            attack_type_weights=attack_type_weights,
            pred_quantiles=getattr(cfg, "pred_quantiles", None),
            beta=getattr(cfg, "beta", 1.0),
            cw=cw,
            method=method, 
            tune_temperature=True
        )

        thr_path = out_path / f"{country}_cal_threshold.json"
        with open(thr_path, "w") as f:
            json.dump(threshold_dict, f, indent=2)
        print(f"[OK] Saved threshold to {thr_path}")

        print(f"[DONE] Preparation for inference model for {country}")

def train_all(
    ae_type: str, 
    config_path: str,
    tr: int, 
    vr: int, 
    full: bool, 
    latent: bool,
    method: str, 
    cw: int
) -> None:
    for c in COUNTRIES:
        try:
            train_model(
                ae_type, 
                c, 
                config_path, 
                tr, 
                vr, 
                full, 
                latent, 
                method, 
                cw
            )
        except Exception as e:
            print(f"[ERROR] Failed for {c}: {e}")

    print(f"\n[DONE] All trainings completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train single- or multi-task autoencoder for one or multiple countries"
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
        help="full dataset for training, no validation (overwrites -tr/-vr); use for inference [default: false]"
    )


    parser.add_argument(
        "-L", "--latent",
        action="store_true",
        help="perform latent space analysis [default: false]"
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
        help="calibration window in days for anomaly threshold [default: 30]"
    )

    parser.add_argument(
        "path",
        help="<config|trained|tuned> path to model config"
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

    config_path = args.path.lower()
    if config_path not in ["config", "trained", "tuned"]:
        parser.print_help()
        print(f"""
            [Error] Config path {config_path} not recognised; valid options:
            \t[config]  for app/src/config
            \t[trained] for results/ae_ml/trained
            \t[tuned]   for results/ae_ml/tuned
            """)
        exit(1)

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
        train_all(
            ae_type,
            config_path,
            args.tr, 
            args.vr, 
            args.full, 
            args.latent,
            args.method, 
            args.calwindow
        )
    else:
        train_model(
            ae_type, 
            target,
            config_path,
            args.tr, 
            args.vr, 
            args.full, 
            args.latent,
            args.method, 
            args.calwindow
        )
