import json, pickle, torch, optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
from dataclasses import asdict
from app.src.data.feature_engineering import load_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.training.train import train_autoencoder, save_autoencoder
from app.src.ml.analysis.analysis import plot_latent_space


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[3]
OUT_DIR = PROJECT_ROOT / "results" / "ml" / "tuned"
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
        ae_type: str,
        trial: optuna.Trial,
        metric: str,
        country: str, 
        tr: int, 
        vr: int,
        path: Path
    ) -> float:
    """Defines objective for Optuna incl. train model with trial hyperparameters and return validation loss."""
    set_global_seeds(42)
    
    trial_path = path / "trial_history"
    trial_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    X_cont_df, X_cat_df, num_cont, cat_dims, = load_feature_matrix(country)
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
    possible_h = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", possible_h)
        for i in range(depth)
    ]
    hidden_dims = [max(h, latent_dim * 2) for h in hidden_dims]
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
    cont_weight = trial.suggest_float("cont_weight", 0.0, 2.0)
    cat_weight = trial.suggest_float("cat_weight", 0.0, 2.0)
    loss_weights = {"cont_weight": cont_weight, "cat_weight": cat_weight}
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu", "tanh", "elu"])

    if ae_type == "vae":
        beta = trial.suggest_float("beta", 0.1, 5.0, log=True)
    else:
        beta = None

    # ------------------------------------
    # Config object
    # ------------------------------------
    base_cfg = dict(
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
        activation=activation,
        temperature=1.0
    )
    if ae_type == "vae":
        base_cfg["beta"] = beta

    config_map = {
        "ae": AEConfig,
        "vae": VAEConfig
    }
    cfg = config_map[ae_type](**base_cfg)

    # ------------------------------------
    # Train model
    # ------------------------------------
    model, history = train_autoencoder(
        Xc_train_scald, Xk_train,
        Xc_val_scald, Xk_val, 
        cfg,
        loss_weights=loss_weights
    )

    cont_loss_name = "cont_loss" if ae_type == "ae" else "recon_loss"
    cat_loss_name = "cat_loss" if ae_type == "ae" else "kl_loss"
    # save per-trial history
    trial_history_path = trial_path / f"{country}_trial_{trial.number:04d}_history.json"
    with open(trial_history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ------------------------------------
    # Optimization metric: ELBO, recon-only, mixed scoring
    # ------------------------------------
    if ae_type == "vae":
        if metric == "elbo":
        # OPTION 1 : tune using ELBO : ELBO=ReconLoss+β⋅KL
        # = to find smoothest latent distributeion; for true generative model
            best_epoch = np.argmin(history[f"val_loss"])
            final_val_loss = history[f"val_loss"][best_epoch]
        elif metric == "recon":
        # OPTION 2 : tune using recon only 
        # = then VAE behaves like a regularized AE, sharper recon, KL important for taining stability not for selection
        # CAREFUL: no best epoch, model selected based on full validation set after training finishes
            model.eval()
            Xc_val_t = torch.tensor(Xc_val_scald, dtype=torch.float32, device=cfg.device)
            Xk_val_t = torch.tensor(Xk_val,       dtype=torch.int64,  device=cfg.device)
            with torch.no_grad():
                rec_errors = model.reconstruction_error_per_sample(Xc_val_t, Xk_val_t)
                final_val_loss = rec_errors.mean().item()
        elif metric == "mixed":
        # OPTION 3 : tune using mixed scoring : Recon + λ·KL or Recon + α·CatLoss
        # when both recon and regularization matter 
        # when recon-only gives too unstable latent representation, but ELBO is too strict
            lambda_kl = cfg.beta if isinstance(cfg, VAEConfig) else 0.1
            mixed_scores = history[f"val_{cont_loss_name}"] + lambda_kl * history[f"val_{cat_loss_name}"]
            best_epoch = np.argmin(mixed_scores)
            final_val_loss = mixed_scores[best_epoch]
        else:
            raise ValueError("[ERROR] Tuning metric nor recognize, aborting!")
    else:
        best_epoch = np.argmin(history[f"val_loss"])
        final_val_loss = history[f"val_loss"][best_epoch]

    trial.report(final_val_loss, step=0)
    # Store additional metrics
    trial.set_user_attr("tuning_metric", metric)
    trial.set_user_attr("cont_weight", cont_weight)
    trial.set_user_attr("cat_weight", cat_weight)
    trial.set_user_attr("best_epoch", int(best_epoch) if 'best_epoch' in locals() else -1)

    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return final_val_loss


#########################################
##        TUNING WRAPPER (API)         ##
#########################################

def tune_country(
        ae_type: str,
        country: str, 
        n_trials: int = 40, 
        pruner: str = "median",
        metric: str = "elbo",
        tr: int = 75,
        vr: int = 15,
        latent: bool = False
    ):
    """Full Optuna tuning incl. creating study, running optimization, retraining best model fully"""
    set_global_seeds(42)

    out_path = OUT_DIR / f"{ae_type.upper()}"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n==============================")
    print(f"   OPTUNA TUNING FOR {country}")
    print(f"==============================\n")
    print(f"[INFO] Model {ae_type.upper()} selected")
    print(f"[INFO] Tuning metric {metric.upper()} selected")

    pr = {
        "median": MedianPruner(n_startup_trials=5),
        "halving": SuccessiveHalvingPruner(),
        "hyperband": HyperbandPruner(),
    }.get(pruner)

    if pr is None:
        raise ValueError(f"Unknown pruner: {pruner}")

    db_path = out_path / f"{country}_study.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=pr,
        storage=f"sqlite:///{db_path}",
        study_name=f"ae_tuning_{country}",
        load_if_exists=True,
    )
    study.set_user_attr("tuning_metric", metric)
    study.optimize(
        lambda t: objective(ae_type, t, metric, country, tr, vr, out_path),
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
    X_cont_df, X_cat_df, num_cont, cat_dims = load_feature_matrix(country)
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

    cont_weight = p.get("cont_weight", 1.0)
    cat_weight = p.get("cat_weight", 0.0)
    loss_weights = {"cont_weight": cont_weight, "cat_weight": cat_weight}

    activation = p.get("activation", "relu")

    best_base_cfg = dict(
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
        activation=activation,
        temperature=1.0
    )
    if ae_type == "vae":
        best_base_cfg["beta"] = p.get("beta", 1.0)

    config_map = {
        "ae": AEConfig,
        "vae": VAEConfig
    }
    best_cfg = config_map[ae_type](**best_base_cfg)

    best_model, best_history = train_autoencoder(
        Xc_train_scald, Xk_train, 
        Xc_val_scald, Xk_val, 
        best_cfg,
        loss_weights
    )

    # ------------------------------------
    # Save output
    # ------------------------------------
    out_model_path = out_path / f"{country}_best_model.pt"
    save_autoencoder(
        model=best_model, 
        config=best_cfg, 
        path=out_model_path,
        additional_info={
            "country": country,
            "train_ratio": tr,
            "val_ratio": vr,
            "loss_weights": loss_weights,
            "total_samples": len(Xc_train_scald),
            "tuning_metric": metric,
        }
    )

    with open(out_path / f"{country}_best_params.json", "w") as f:
        json.dump(p, f, indent=2)

    with open(out_path / f"{country}_best_config.json", "w") as f:
        cfg_for_save = asdict(best_cfg)
        json.dump(cfg_for_save, f, indent=2)

    with open(out_path / f"{country}_best_history.json", "w") as f:
        json.dump(best_history, f, indent=2)

    with open(out_path / f"{country}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(out_path / f"{country}_cat_dims.json", "w") as f:
        json.dump(cat_dims, f, indent=2)

    print(f"\n[OK] Finished tuning for {country}")

    if latent:
        plot_latent_space(
            country, 
            Xc_train_scald, 
            Xk_train,
            best_model,
            best_cfg.device,
            1000,
            out_path,
            f"{country}_best_latent_space.png"
        )
        
    print(f"[DONE] Saved best model to {out_model_path}")

    return study