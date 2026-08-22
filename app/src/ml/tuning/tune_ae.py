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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if ae_type == "vae":
        beta = trial.suggest_float(
            "beta", 
            float(params["beta"]["start"]),
            float(params["beta"]["end"]),
            log=True
        )
        
        base_cfg = base_cfg | dict(
            temperature=1.0,
            use_beta_annealing=True,
            beta_schedule="linear",
            beta=beta,
            debug_kl_stats=True
        )
        del base_cfg["warmup_epochs"]

    config_map = {
        "ae": AEConfig,
        "vae": VAEConfig
    }
    cfg = config_map[ae_type](**base_cfg)

    metric1_name = "cont_loss" if ae_type == "ae" else "recon_loss"
    metric2_name = "cat_loss" if ae_type == "ae" else "kl_loss"

    # ------------------------------------
    # Train model
    # ------------------------------------
    model, history = train_autoencoder(
        Xc_train_scald, Xk_train,
        Xc_val_scald, Xk_val, 
        cfg,
        loss_weights=loss_weights
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
    trial.set_user_attr("cont_w", cont_w)
    trial.set_user_attr("cat_w", cat_w)
    if ae_type == "vae":
        trial.set_user_attr("beta", beta)
    trial.set_user_attr("best_epoch", int(best_epoch) if 'best_epoch' in locals() else -1)

    # ------------------------------------
    # Store trial
    # ------------------------------------
    trial_history_path = trial_path / f"{country}_trial_{trial.number:04d}_history.json"
    with open(trial_history_path, "w") as f:
        json.dump(history, f, indent=2)


    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return final_val_loss
