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
    _, history = train_multitask_model(
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
