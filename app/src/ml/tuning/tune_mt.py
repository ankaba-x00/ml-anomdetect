import json, torch, optuna
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

from app.src.data.feature_engineering import load_supervised_feature_matrix
from app.src.data.split import timeseries_seq_split
from app.src.ml.models.mte import MTEConfig
from app.src.ml.training.train_mt import train_multitask_model


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
    trial: optuna.Trial,
    country: str,
    tr: int,
    vr: int,
    out_path: Path,
    params: dict,
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
    y3 = y_l3.values.astype(np.float32)
    y7 = y_l7.values.astype(np.float32)
    ya = y_attack.values.astype(np.int64)

    # -----------------------------
    # Split dataset
    # -----------------------------
    (Xc_tr, Xk_tr), (Xc_val, Xk_val), _ = timeseries_seq_split(
        Xc_np, 
        Xk_np, 
        tr/100, 
        vr/100
    )
    y3_tr, y3_val, _ = timeseries_seq_split(y3, None, tr/100, vr/100)
    y7_tr, y7_val, _ = timeseries_seq_split(y7, None, tr/100, vr/100)
    ya_tr, ya_val, _ = timeseries_seq_split(ya, None, tr/100, vr/100)

    # -----------------------------
    # Fit scaler on cont features and tranform data
    # -----------------------------
    scaler = RobustScaler()
    Xc_tr = scaler.fit_transform(Xc_tr).astype(np.float32)
    Xc_val = scaler.transform(Xc_val).astype(np.float32)
    
    # -----------------------------
    # Hyperparameter search space
    # -----------------------------
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
    head_hidden_dim = trial.suggest_categorical(
        "head_hidden_dim", 
        params["head_hidden_dim"]
    )
    dropout = trial.suggest_float(
        "dropout", 
        params["dropout"]["start"], 
        params["dropout"]["end"]
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
    batch_size = trial.suggest_categorical(
        "batch_size", 
        params["batch_size"]
    )
    patience = trial.suggest_int(
        "patience", 
        params["patience"]["start"], 
        params["patience"]["end"]
    )
    activation_en = trial.suggest_categorical(
        "activation_en", 
        params["activation_en"]
    )
    activation_de_reg = trial.suggest_categorical(
        "activation_de_reg", 
        params["activation_de_reg"]
    )
    activation_de_cls = trial.suggest_categorical(
        "activation_de_cls", 
        params["activation_de_cls"]
    )
    lambda_l3 = trial.suggest_float(
        "lambda_l3", 
        params["lambda_l3"]["start"], 
        params["lambda_l3"]["end"]
    )
    lambda_l7 = trial.suggest_float(
        "lambda_l7", 
        params["lambda_l7"]["start"], 
        params["lambda_l7"]["end"]
    )
    lambda_attack = trial.suggest_float(
        "lambda_attack", 
        params["lambda_attack"]["start"], 
        params["lambda_attack"]["end"], 
        log=True
    )
    loss_weights = {
        "l3": lambda_l3,
        "l7": lambda_l7,
        "attack": lambda_attack,
    }
    use_focal_loss = params["use_focal_loss"][0]
    if use_focal_loss:
        focal_gamma = trial.suggest_float(
            "focal_gamma", 
            params["focal_gamma"]["start"],
            params["focal_gamma"]["end"]
        )

    # -----------------------------
    # Config object
    # -----------------------------
    cfg = MTEConfig(
        num_cont=num_cont,
        cat_dims=cat_dims,
        n_attack_types=8,
        hidden_dims=tuple(hidden_dims),
        latent_dim=latent_dim,
        quantiles=(0.5, 0.9, 0.99),
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=50,
        warmup_epochs=5,
        patience=patience,
        activation_en=activation_en,
        activation_de_reg=activation_de_reg,
        activation_de_cls=activation_de_cls,
        lambda_l3=lambda_l3,
        lambda_l7=lambda_l7,
        lambda_attack=lambda_attack,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
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
        if epoch < cfg.warmup_epochs:
            continue
        if val_loss is None or not np.isfinite(val_loss):
            raise optuna.TrialPruned()

        trial.report(val_loss, step=epoch)

        if epoch >= cfg.warmup_epochs and trial.should_prune():
            raise optuna.TrialPruned()
    
    valid_losses = [
        (i, v) for i, v in enumerate(history["val_loss"])
        if i >= cfg.warmup_epochs and v is not None
    ]
    if not valid_losses and trial.should_prune():
        raise optuna.TrialPruned()
    
    best_epoch, best_val_loss = min(valid_losses, key=lambda x: x[1])
    final_val_loss = float(best_val_loss)
    
    trial.set_user_attr(
        "loss_weights",
        {
            "l3": lambda_l3,
            "l7": lambda_l7,
            "attack": lambda_attack,
        }
    )
    trial.set_user_attr(
        "attack_class_weights",
        model.attack_class_weights.cpu().tolist() if model.attack_class_weights is not None else None
    )
    frequ = np.bincount(ya_tr)
    trial.set_user_attr(
        "attack_class_frequencies", 
        frequ.tolist()
    )
    trial.set_user_attr("best_epoch", best_epoch)

    return final_val_loss
