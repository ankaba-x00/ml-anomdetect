import json, pickle, torch, optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from pathlib import Path
import numpy as np
from dataclasses import asdict
from src.data.feature_engineering import build_feature_matrix
from src.models.autoencoder import AEConfig
from src.models.train import train_autoencoder, save_autoencoder


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##                 SETUP               ##
#########################################

def set_global_seeds(seed: int = 42):
    """Ensures reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#########################################
##          OPTUNA OBJECTIVE           ##
#########################################

def objective(trial: optuna.Trial, country: str) -> float:
    """Defines objective for Optuna incl. train AE with trial hyperparameters and return validation loss."""
    set_global_seeds(42)

    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    X_cont_df, X_cat_df, num_cont, cat_dims, scaler = build_feature_matrix(country)
    Xc_np = X_cont_df.values.astype(np.float32)
    Xk_np = X_cat_df.values.astype(np.int64)

    # ------------------------------------
    # Hyperparameter search space
    # ------------------------------------
    latent_dim = trial.suggest_categorical("latent_dim", [8, 12, 16, 24, 32])

    depth = trial.suggest_int("depth", 1, 3)
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", [64, 128, 256])
        for i in range(depth)
    ]

    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    patience = trial.suggest_int("patience", 4, 10)

    # ============================================================
    # AEConfig object
    # ============================================================

    cfg = AEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=45,
        patience=patience,
        val_split=0.2,
        gradient_clip=1.0,
        use_lr_scheduler=True,
        time_series_split=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ------------------------------------
    # Train model
    # ------------------------------------
    model, history = train_autoencoder(Xc_np, Xk_np, cfg)

    final_val_loss = history["val_loss"][-1]

    trial.report(final_val_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return final_val_loss


#########################################
##        TUNING WRAPPER (API)         ##
#########################################

def tune_country(country: str, n_trials: int = 40, pruner: str = "median"):
    """Full Optuna tuning incl. creating study, running optimization, retraining best model fully"""
    set_global_seeds(42)

    print(f"\n==============================")
    print(f"   OPTUNA TUNING FOR {country}")
    print(f"==============================\n")

    pr = {
        "median": MedianPruner(n_startup_trials=5),
        "halving": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
    }.get(pruner)

    if pr is None:
        raise ValueError(f"Unknown pruner: {pruner}")

    db_path = RESULTS_DIR / f"{country}_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, country),
        n_trials=n_trials,
        n_jobs=1,
    )
    print("\nBest Trial:")
    print(study.best_trial)
    print("\nBest Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # ------------------------------------
    # Retrain best model fully
    # ------------------------------------
    X_cont_df, X_cat_df, num_cont, cat_dims, scaler = build_feature_matrix(country)
    Xc_np = X_cont_df.values.astype(np.float32)
    Xk_np = X_cat_df.values.astype(np.int64)

    p = study.best_trial.params
    depth = p["depth"]
    hidden_dims = [p[f"h{i}"] for i in range(depth)]

    best_cfg = AEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        latent_dim=p["latent_dim"],
        hidden_dims=tuple(hidden_dims),
        dropout=p["dropout"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        batch_size=p["batch_size"],
        num_epochs=120,
        patience=p["patience"],
        val_split=0.2,
        gradient_clip=1.0,
        use_lr_scheduler=True,
        time_series_split=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    best_model, best_history = train_autoencoder(Xc_np, Xk_np, best_cfg)

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = RESULTS_DIR / f"{country}_best_model.pt"
    save_autoencoder(best_model, best_cfg, out_model_path)

    with open(RESULTS_DIR / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(RESULTS_DIR / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(RESULTS_DIR / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(RESULTS_DIR / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(RESULTS_DIR / f"{country}_cat_dims.json", "w") as f:
        json.dump(cat_dims, f, indent=2)

    print(f"\n[OK] Finished tuning for {country}")
    print(f"[DONE] Saved best model → {out_model_path}")

    return study