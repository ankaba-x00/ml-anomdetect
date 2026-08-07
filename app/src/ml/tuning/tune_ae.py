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
        params: dict
    ) -> float:
    set_global_seeds(42)
    
    trial_path = path / "trial_history"
    trial_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # Load feature matrix
    # ------------------------------------
    print(f"\n[INFO] Setting up new trial")
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
    depth = trial.suggest_int(
        "depth", 
        params["depth"]["start"], 
        params["depth"]["end"]
    )
    base_dim = trial.suggest_categorical(
        "base_dim", 
        params["base_dim"]
    )
    hidden_dims = [max(32, int(base_dim / (2**i))) for i in range(depth)]
    latent_dim = trial.suggest_categorical(
        "latent_dim", 
        params["latent_dim"]
    )
    activation_en = trial.suggest_categorical(
        "activation_en", 
        params["activation_en"]
    )
    activation_de = trial.suggest_categorical(
        "activation_de", 
        params["activation_de"]
    )
    dropout = trial.suggest_float(
        "dropout", 
        params["dropout"]["start"], 
        params["dropout"]["end"]
    )
    optimizer = trial.suggest_categorical(
        "optimizer", 
        params["optimizer"]
    )
    lr_scheduler = trial.suggest_categorical(
        "lr_scheduler", 
        params["lr_scheduler"]
    )
    lr = trial.suggest_float(
        "lr", 
        float(params["lr"]["start"]), 
        float(params["lr"]["end"]), 
        log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 
        float(params["weight_decay"]["start"]), 
        float(params["weight_decay"]["end"]), 
        log=True
    )
    adam_beta1 = trial.suggest_float(
        "adam_beta1", 
        float(params["adam_beta1"]["start"]), 
        float(params["adam_beta1"]["end"]), 
    )
    adam_beta2 = trial.suggest_float(
        "adam_beta2", 
        float(params["adam_beta2"]["start"]), 
        float(params["adam_beta2"]["end"]), 
    )
    sgd_momentum = trial.suggest_float(
        "sgd_momentum", 
        float(params["sgd_momentum"]["start"]), 
        float(params["sgd_momentum"]["end"]), 
    )
    gradient_clip = trial.suggest_categorical(
        "gradient_clip",
        params["gradient_clip"]
    )
    batch_size = trial.suggest_categorical(
        "batch_size", 
        params["batch_size"]
    )
    patience = trial.suggest_int(
        "patience", 
        params["patience"]["start"], 
        params["patience"]["end"]
    )
    noise_gauss_std = trial.suggest_float(
        "noise_gauss_std", 
        float(params["noise_gauss_std"]["start"]), 
        float(params["noise_gauss_std"]["end"])
    )
    noise_mask_prob = trial.suggest_float(
        "noise_mask_prob", 
        float(params["noise_mask_prob"]["start"]), 
        float(params["noise_mask_prob"]["end"])
    )
    cont_w = trial.suggest_float(
        "cont_w", 
        float(params["cont_weight"]["start"]), 
        float(params["cont_weight"]["end"])
    )
    cat_w = trial.suggest_float(
        "cat_w", 
        float(params["cat_weight"]["start"]), 
        float(params["cat_weight"]["end"])
    )
    loss_weights = {"cont_w": cont_w, "cat_w": cat_w}

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
        use_embedding=False,
        hidden_dims=tuple(hidden_dims),
        latent_dim=latent_dim,
        activation_en=activation_en,
        activation_de=activation_de,
        dropout=dropout,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        sgd_momentum=sgd_momentum,
        lr_scheduler=lr_scheduler,
        gradient_clip=1.0,
        batch_size=batch_size,
        allow_noise_injection=True,
        noise_gauss_std=noise_gauss_std,
        noise_mask_prob=noise_mask_prob,
        num_epochs=60,
        warmup_epochs=10,
        patience=patience,
        anomaly_threshold=None,
        temperature=1.0,  
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    trial.set_user_attr("cont_w", cont_w)
    trial.set_user_attr("cat_w", cat_w)
    trial.set_user_attr("best_epoch", int(best_epoch) if 'best_epoch' in locals() else -1)

    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return final_val_loss
