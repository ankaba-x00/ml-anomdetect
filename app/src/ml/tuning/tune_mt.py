import json, pickle, torch, optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from dataclasses import asdict
from app.src.data.feature_engineering import load_supervised_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.mte import MTEConfig
from app.src.ml.training.train_mt import train_multitask_model, save_multitask_model
from app.src.ml.analysis.analysis import plot_latent_space


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "mt_ml" / "tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    vr: int,
    out_path: Path,
) -> float:
    set_global_seeds(42)

    trial_path = out_path / "trial_history"
    trial_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load supervised feature matrix
    # -----------------------------
    Xc, Xk, y_l3, y_l7, y_attack, num_cont, cat_dims = (
        load_supervised_feature_matrix(country)
    )
    Xc_np = Xc.values.astype(np.float64)
    Xk_np = Xk.values.astype(np.int64)

    # -----------------------------
    # Split dataset
    # -----------------------------
    (Xc_tr, Xk_tr), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, 
        Xk_np, 
        tr/100, 
        vr/100
    )
    y3_tr, y3_val, _ = timeseries_seq_split(y_l3.values, None, tr/100, vr/100)
    y7_tr, y7_val, _ = timeseries_seq_split(y_l7.values, None, tr/100, vr/100)
    ya_tr, ya_val, _ = timeseries_seq_split(y_attack.values, None, tr/100, vr/100)

    # -----------------------------
    # Fit scaler on cont features and tranform data
    # -----------------------------
    scaler = RobustScaler()
    Xc_tr = scaler.fit_transform(Xc_tr).astype(np.float32)
    Xc_val = scaler.transform(Xc_val).astype(np.float32)

    # -----------------------------
    # Hyperparameter search space
    # -----------------------------
    depth = trial.suggest_int("depth", 1, 4)
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", [64, 128, 256, 384, 512])
        for i in range(depth)
    ]
    latent_dim = 32
    #latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64, 96])
    head_hidden_dim = 32
    #head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    patience = trial.suggest_int("patience", 4, 10)
    activation = trial.suggest_categorical(
        "activation", ["relu", "gelu", "leaky_relu"]
    )
    lambda_l3 = trial.suggest_float("lambda_l3", 0.5, 2.0)
    lambda_l7 = trial.suggest_float("lambda_l7", 0.5, 2.0)
    lambda_attack = trial.suggest_float("lambda_attack", 0.5, 2.0)
    loss_weights = {
        "l3": lambda_l3,
        "l7": lambda_l7,
        "attack": lambda_attack,
    }

    # -----------------------------
    # Config object
    # -----------------------------
    cfg = MTEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        n_attack_types=8,
        hidden_dims=tuple(hidden_dims),
        latent_dim=latent_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=50,
        patience=patience,
        activation=activation,
        lambda_l3=lambda_l3,
        lambda_l7=lambda_l7,
        lambda_attack=lambda_attack,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # -----------------------------
    # Train
    # -----------------------------
    model, history = train_multitask_model(
        Xc_tr, Xk_tr, y3_tr, y7_tr, ya_tr,
        Xc_val, Xk_val, y3_val, y7_val, ya_val,
        cfg,
        loss_weights
    )
    # save per-trial history
    trial_history_path = trial_path / f"{country}_trial_{trial.number:04d}_history.json"
    with open(trial_history_path, "w") as f:
        json.dump(history, f, indent=2)

    # -----------------------------
    # Objective: best validation loss
    # -----------------------------
    for epoch, val_loss in enumerate(history["val_loss"]):
        trial.report(val_loss, step=epoch)
        if np.isnan(val_loss) or np.isinf(val_loss):
            raise optuna.TrialPruned()
        if epoch >= 5 and trial.should_prune():
            raise optuna.TrialPruned()

    best_epoch = int(np.argmin(history["val_loss"]))
    final_val_loss = float(history["val_loss"][best_epoch])
    trial.set_user_attr(
        "loss_weights",
        {
            "l3": lambda_l3,
            "l7": lambda_l7,
            "attack": lambda_attack,
        }
    )
    trial.set_user_attr("best_epoch", best_epoch)

    return final_val_loss


#########################################
##        TUNING WRAPPER (API)         ##
#########################################

def tune_country(
        country: str,
        n_trials: int = 40,
        pruner: str = "median",
        tr: int = 75,
        vr: int = 15,
        latent: bool = False
    ):
    """Full Optuna tuning incl. creating study, running optimization, retraining best model fully"""
    set_global_seeds(42)

    print(f"\n==============================")
    print(f" OPTUNA MT TUNING FOR {country}")
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
        study_name=f"mt_tuning_{country}",
        load_if_exists=True,
    )
    study.optimize(
        lambda t: objective(t, country, tr, vr, OUT_DIR),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )
    print("\nBest Trial:")
    print(study.best_trial)
    print("\nBest Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # ------------------------------------
    # Retrain best model fully
    # ------------------------------------
    X_cont_df, X_cat_df, y3_s, y7_s, ya_s, num_cont, cat_dims = load_supervised_feature_matrix(country)
    Xc = X_cont_df.values.astype(np.float64)
    Xk = X_cat_df.values.astype(np.int64)
    y3 = y3_s.values.astype(np.float32)
    y7 = y7_s.values.astype(np.float32)
    ya = ya_s.values.astype(np.int64)

    print(f"[INFO] Dataset split ratio: {tr}% train | {vr}% val | {100-tr-vr}% test")
    (Xc_train, Xk_train), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc, Xk,
        train_ratio=tr/100,
        val_ratio=vr/100
    )
    y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
    y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
    ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

    scaler = RobustScaler()
    Xc_train_scald = scaler.fit_transform(Xc_train).astype(np.float32)
    Xc_val_scald = scaler.transform(Xc_val).astype(np.float32)

    p = study.best_trial.params
    depth = p["depth"]
    hidden_dims = [p[f"h{i}"] for i in range(depth)]

    lambda_l3 = p.get("lambda_l3", 1.0)
    lambda_l7 = p.get("lambda_l7", 1.0)
    lambda_attack = p.get("lambda_attack", 1.0)
    loss_weights = {
        "l3": lambda_l3,
        "l7": lambda_l7,
        "attack": lambda_attack,
    }

    best_cfg = MTEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        n_attack_types=8,
        hidden_dims=tuple(hidden_dims),
        #latent_dim=p["latent_dim"],
        dropout=p["dropout"],
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        batch_size=p["batch_size"],
        num_epochs=90,
        patience=p["patience"],
        activation=p["activation"],
        lambda_l3=lambda_l3,
        lambda_l7=lambda_l7,
        lambda_attack=lambda_attack,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    best_model, best_history = train_multitask_model(
        Xc_train_scald, Xk_train, y3_tr, y7_tr, ya_tr,
        Xc_val_scald, Xk_val, y3_val, y7_val, ya_val,
        best_cfg,
        loss_weights
    )

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = OUT_DIR / f"{country}_best_model.pt"
    save_multitask_model(
        model=best_model, 
        config=best_cfg,
        cat_dims=cat_dims,
        num_cont=num_cont,
        path=out_model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(Xc_train_scald),
        }
    )

    with open(OUT_DIR / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(OUT_DIR / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(OUT_DIR / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(OUT_DIR / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[OK] Finished tuning for {country}")

    if latent:
        plot_latent_space(
            country, 
            Xc_train_scald, 
            Xk_train,
            best_model,
            best_cfg.device,
            1000,
            OUT_DIR,
            f"{country}_best_latent_space.png"
        )
        
    print(f"[DONE] Saved best model to {out_model_path}")

    return study