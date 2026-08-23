import json, optuna
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

from . import set_global_seeds, YMLReader
from app.src.data.feature_engineering import load_feature_matrix, load_supervised_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.training.train_ae import train_autoencoder
from app.src.ml.training.train_mt import train_mt_autoencoder


def objective(
    ae_type: str,
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
        X_cont, X_cat, num_cont, cat_dims, = load_feature_matrix(country)
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
    # Split dataset
    # ------------------------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_tr, Xk_tr), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk,
        tr/100,
        vr/100
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
    # Load search space config
    # ------------------------------------
    cfg = YMLReader(ae_type, trial).load_search_space(num_cont, cat_dims)

    loss_weights = {
        "cont_w": cfg.cont_w, 
        "cat_w": cfg.cat_w
    }
    if ae_type == "mtae":
        loss_weights |= {
            "l3_w": cfg.lambda_l3,
            "l7_w": cfg.lambda_l7,
            "at_w": cfg.lambda_at
        }

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
            loss_weights
        )

    # ------------------------------------
    # Score trial
    # ------------------------------------
    best_epoch = np.argmin(history[f"val_loss"])
    final_val_loss = history[f"val_loss"][best_epoch]

    # ------------------------------------
    # Report trial
    # ------------------------------------
    trial.report(final_val_loss, step=0)
    # Store additional metrics
    trial.set_user_attr("config", cfg.__dict__)
    trial.set_user_attr("best_epoch", int(best_epoch) if 'best_epoch' in locals() else -1)
    if ae_type in ["mtae"]:
        trial.set_user_attr("train_at_counts", history["train_at_counts"])
        trial.set_user_attr("val_at_counts", history["val_at_counts"])
        trial.set_user_attr("attack_type_weights", history["attack_type_weights"])

    # ------------------------------------
    # Store trial
    # ------------------------------------
    trial_history_path = trial_path / f"{country}_trial_{trial.number:04d}_history.json"
    with open(trial_history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ------------------------------------
    # Prune
    # ------------------------------------
    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return final_val_loss
