import json, pickle, torch, optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from dataclasses import asdict
from app.src.data.feature_engineering import build_feature_matrix
from app.src.models.autoencoder import AEConfig
from app.src.models.train import train_autoencoder, save_autoencoder
from app.src.data.split import timeseries_seq_split


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "models" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_HIST_DIR = OUT_DIR / "trial_history"
TRIAL_HIST_DIR.mkdir(parents=True, exist_ok=True)


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

def objective(
        trial: optuna.Trial, 
        country: str, 
        tr: int, 
        vr: int
    ) -> float:
    """Defines objective for Optuna incl. train AE with trial hyperparameters and return validation loss."""
    set_global_seeds(42)

    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    X_cont_df, X_cat_df, num_cont, cat_dims, = build_feature_matrix(country)
    Xc_np = X_cont_df.values.astype(np.float64)
    Xk_np = X_cat_df.values.astype(np.int64)

    # ------------------------------------
    # Split dataset
    # ------------------------------------
    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=tr/100,
        val_ratio=vr/100
    )

    # ------------------------------------
    # Fit scaler on cont features and tranform data
    # ------------------------------------
    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    # ------------------------------------
    # Hyperparameter search space
    # ------------------------------------
    latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192])
    depth = trial.suggest_int("depth", 1, 4)
    possible_h = [int(latent_dim * m) for m in [2, 3, 4, 6, 8, 12]]
    possible_h = [h for h in possible_h if 32 <= h <= 1024]
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", possible_h)
        for i in range(depth)
    ]
    dropout = trial.suggest_float("dropout", 0.0, 0.35)
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    patience = trial.suggest_int("patience", 4, 10)
    embedding_dim = trial.suggest_categorical("embedding_dim", [4, 8, 12, 16, 24])
    noise_std = trial.suggest_float("noise_std", 0.0, 0.20)
    residual_strength = trial.suggest_float("residual_strength", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr_scheduler = trial.suggest_categorical(
        "lr_scheduler",
        ["none", "plateau", "cosine", "onecycle"]
    )

    # ------------------------------------
    # AEConfig object
    # ------------------------------------
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
        gradient_clip=1.0,
        use_lr_scheduler=True,
        embedding_dim=embedding_dim,
        continuous_noise_std=noise_std,
        residual_strength=residual_strength,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ------------------------------------
    # Train model
    # ------------------------------------
    _, history = train_autoencoder(
        Xc_train_scald, Xk_train,
        Xc_val_scald, Xk_val, 
        cfg
    )

    # save per-trial history (optional)
    trial_history_path = TRIAL_HIST_DIR / f"{country}_trial_{trial.number:04d}_history.json" # TODO: added :04d so that the studies do not overwrite themselves. Maybe use f"{country}_study_{study_name}_trial_{trial.number}.json" instead?!?
    with open(trial_history_path, "w") as f:
        json.dump(history, f, indent=2)

    best_epoch = np.argmin(history["val_loss"])
    final_val_loss = history["val_loss"][best_epoch]

    trial.report(final_val_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return final_val_loss


#########################################
##        TUNING WRAPPER (API)         ##
#########################################

def tune_country(
        country: str, 
        n_trials: int = 40, 
        pruner: str = "median",
        tr: int = 75,
        vr: int = 15
    ):
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

    db_path = OUT_DIR / f"{country}_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, country, tr, vr),
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
    X_cont_df, X_cat_df, num_cont, cat_dims = build_feature_matrix(country)
    Xc_np = X_cont_df.values.astype(np.float64)
    Xk_np = X_cat_df.values.astype(np.int64)

    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, Xk_np,
        train_ratio=tr/100,
        val_ratio=vr/100
    )

    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

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
        num_epochs=90,
        patience=p["patience"],
        gradient_clip=1.0,
        use_lr_scheduler=True,
        embedding_dim=p["embedding_dim"],
        continuous_noise_std=p["noise_std"],
        residual_strength=p["residual_strength"],
        optimizer=p["optimizer"],
        lr_scheduler=p["lr_scheduler"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    best_model, best_history = train_autoencoder(
        Xc_train_scald, Xk_train, 
        Xc_val_scald, Xk_val, 
        best_cfg
    )

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = OUT_DIR / f"{country}_best_model.pt"
    save_autoencoder(best_model, best_cfg, out_model_path)

    with open(OUT_DIR / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(OUT_DIR / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(OUT_DIR / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(OUT_DIR / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(OUT_DIR / f"{country}_cat_dims.json", "w") as f:
        json.dump(cat_dims, f, indent=2)

    print(f"\n[OK] Finished tuning for {country}")
    print(f"[DONE] Saved best model to {out_model_path}")

    return study