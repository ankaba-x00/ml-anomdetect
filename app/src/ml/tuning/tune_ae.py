import json, torch, optuna
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

from app.src.data.feature_engineering import load_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.ae import AEConfig
from app.src.ml.models.vae import VAEConfig
from app.src.ml.training.train_ae import train_autoencoder


#########################################
##                 SETUP               ##
#########################################

def set_global_seeds(seed: int = 42) -> None:
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
        path: Path,
    ) -> float:
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
        tr/100,
        vr/100
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
    depth = trial.suggest_int("depth", 2, 3)
    base_dim = trial.suggest_categorical("base_dim", [128, 256, 512])
    hidden_dims = [max(32, int(base_dim / (2**i))) for i in range(depth)]
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    patience = trial.suggest_int("patience", 4, 9)
    embedding_dim = trial.suggest_categorical("embedding_dim", [8, 12, 16])
    noise_std = trial.suggest_float("noise_std", 0.0, 0.20)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr_scheduler = trial.suggest_categorical(
        "lr_scheduler",
        ["none", "plateau", "cosine", "onecycle"]
    )
    cont_weight = trial.suggest_float("cont_weight", 0.0, 2.0)
    cat_weight = trial.suggest_float("cat_weight", 0.0, 2.0)
    loss_weights = {"cont_weight": cont_weight, "cat_weight": cat_weight}
    activation_en = trial.suggest_categorical("activation_en", ["relu", "leaky_relu", "tanh", "sigmoid", "silu"])
    activation_de = trial.suggest_categorical("activation_de", ["relu", "leaky_relu", "tanh", "sigmoid", "silu"])

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
        num_epochs=50,
        patience=patience,
        gradient_clip=1.0,
        use_lr_scheduler=True,
        embedding_dim=embedding_dim,
        continuous_noise_std=noise_std,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        activation_en=activation_en,
        activation_de=activation_de,
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
