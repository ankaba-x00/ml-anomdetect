import json, optuna
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Literal

from . import set_global_seeds, YMLReader
from app.src.data.building.feature_engineering import load_feature_matrix, load_supervised_feature_matrix
from app.src.data.processing.split import timeseries_seq_split
from app.src.ml.models.configs import AEConfig, MTAEConfig, VAEConfig
from app.src.ml.training.train_ae import train_autoencoder
from app.src.ml.training.train_mt import train_mt_autoencoder


def objective(
    ae_type: Literal["ae", "vae", "mtae"],
    trial: optuna.Trial,
    country: str, 
    tr: int, 
    vr: int,
    path: Path
) -> float:
    print(f"\n[INFO] Setting up new trial")
    # -----------------------------
    # Prepare trial
    # -----------------------------
    set_global_seeds(42)
    
    trial_path = path / "trial_history"
    trial_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    if ae_type in ["ae", "vae"]:
        fmatrix = load_feature_matrix(country)
    else:
        fmatrix = (
            load_supervised_feature_matrix(country)
        )
    Xc = fmatrix.X_cont.to_numpy(dtype=np.float32)
    Xk = fmatrix.X_cat.to_numpy(dtype=np.int64)
    if (
        ae_type in ["mtae"]
        and fmatrix.y_l3 is not None
        and fmatrix.y_l7 is not None
        and fmatrix.y_at is not None
    ):
        y3 = fmatrix.y_l3.to_numpy(dtype=np.float32)
        y7 = fmatrix.y_l7.to_numpy(dtype=np.float32)
        ya = fmatrix.y_at.to_numpy(dtype=np.int64)

    # ------------------------------------
    # Split dataset
    # ------------------------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    [Xc_tr, Xk_tr], [Xc_val, Xk_val], _ = timeseries_seq_split(
        [Xc, Xk],
        tr/100,
        vr/100
    )
    if ae_type in ["mtae"]:
        [y3_tr], [y3_val], _ = timeseries_seq_split([y3], tr/100, vr/100)
        [y7_tr], [y7_val], _ = timeseries_seq_split([y7], tr/100, vr/100)
        [ya_tr], [ya_val], _ = timeseries_seq_split([ya], tr/100, vr/100)

    # ------------------------------------
    # Scale cont features
    # ------------------------------------
    scaler = RobustScaler()
    Xc_tr_scald = scaler.fit_transform(Xc_tr).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    # ------------------------------------
    # Load search space config
    # ------------------------------------
    cfg = YMLReader(ae_type, trial).load_search_space(fmatrix.num_cont, fmatrix.cat_dims)

    loss_weights = {
        "cont_w": cfg.cont_w, 
        "cat_w": cfg.cat_w
    }
    if isinstance(cfg, MTAEConfig):
        loss_weights |= {
            "l3_w": cfg.lambda_l3,
            "l7_w": cfg.lambda_l7,
            "at_w": cfg.lambda_at
        }

    # ------------------------------------
    # Train model
    # ------------------------------------
    if isinstance(cfg, (AEConfig, VAEConfig)):
        _, hist_tracker = train_autoencoder(
            Xc_tr_scald, Xk_tr,
            Xc_val_scald, Xk_val, 
            cfg,
            loss_weights
        )
    else:
        _, hist_tracker = train_mt_autoencoder(
            Xc_tr_scald, Xk_tr, y3_tr, y7_tr, ya_tr,
            Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
            cfg,
            loss_weights
        )

    # ------------------------------------
    # Score trial
    # ------------------------------------
    best_epoch: int = hist_tracker.best_epoch
    final_val_loss: float = hist_tracker["val_loss"][best_epoch-1]

    # ------------------------------------
    # Report trial
    # ------------------------------------
    trial.report(final_val_loss, step=0)
    # Store additional metrics
    trial.set_user_attr("config", cfg.__dict__)
    trial.set_user_attr("best_epoch", best_epoch)
    if ae_type in ["mtae"]:
        trial.set_user_attr("train_at_counts", hist_tracker["train_at_counts"])
        trial.set_user_attr("val_at_counts", hist_tracker["val_at_counts"])
        trial.set_user_attr("attack_type_weights", hist_tracker["attack_type_weights"])

    # ------------------------------------
    # Store trial
    # ------------------------------------
    trial_history_path = trial_path / f"{country}_trial_{trial.number:04d}_history.json"
    with open(trial_history_path, "w") as f:
        hist_tracker.to_json(f)

    # ------------------------------------
    # Prune
    # ------------------------------------
    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return final_val_loss
