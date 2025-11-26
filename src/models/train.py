from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.autoencoder import AEConfig, TabularAutoencoder


#########################################
##           TRAINING HELPERS          ##
#########################################

def _make_dataloader(
    X_f: np.ndarray, 
    X_i: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Build DataLoader with
       X_f : FloatType (float32) features
       X_i : IntegerType (int64) features 
    """
    Xc_f = torch.from_numpy(X_f.astype(np.float32))
    Xk_i = torch.from_numpy(X_i.astype(np.int64))

    Tds = TensorDataset(Xc_f, Xk_i)

    return DataLoader(
        Tds, 
        batch_size=batch_size, 
        shuffle=shuffle,
    )

def train_autoencoder(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    val_cont: Optional[np.ndarray],
    val_cat: Optional[np.ndarray],
    config: AEConfig,
) -> tuple[TabularAutoencoder, dict[str, Any]]:
    """
    Train autoencoder with continuous + categorical input features on split dataset with early stopping and val metrics OR full dataset with no early stopping.
   
    Returns
    -------
    model : TabularAutoencoder
    history : dict
    """

    device = torch.device(config.device)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = _make_dataloader(
        train_cont, 
        train_cat, 
        config.batch_size, 
        True, 
    ) 
    val_loader = (
        _make_dataloader(
            val_cont, 
            val_cat, 
            config.batch_size, 
            False)
        if val_cont is not None
        else None        
    ) 

    # -------------------------
    # Build model
    # -------------------------
    model = TabularAutoencoder(
        num_cont=config.num_cont,
        cat_dims=config.cat_dims,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
        continuous_noise_std=config.continuous_noise_std,
        residual_strength=config.residual_strength,
    ).to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError("Unknown optimizer")

    # -------------------------
    # LR scheduler
    # -------------------------
    scheduler = None
    if config.use_lr_scheduler:
        if config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        elif config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.num_epochs
            )
        elif config.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.lr,
                total_steps=config.num_epochs * len(train_loader)
            )

    # -------------------------
    # Loss
    # -------------------------
    criterion = torch.nn.MSELoss(reduction="none") 

    history = {
        "train_loss": [],
        "val_loss": [] if val_loader else None,
        "learning_rates": [],
        "best_epoch": None,
    }

    best_metric = float("inf")
    best_state = None
    no_improve = 0

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(config.num_epochs):
        model.train()
        running_train = 0.0
        n_train = 0

        for batch_Xc, batch_Xk in train_loader:
            batch_Xc = batch_Xc.to(device)
            batch_Xk = batch_Xk.to(device)

            optimizer.zero_grad()

            recon = model(batch_Xc, batch_Xk)
            # criterion returns (batch, feature_dim) and loss=mean over all elements
            per_feature = criterion(recon, batch_Xc)
            loss = per_feature.mean()
            loss.backward()

            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()

            if scheduler is not None and config.lr_scheduler == "onecycle":
                scheduler.step()

            running_train += loss.item() * batch_Xc.size(0)
            n_train += batch_Xc.size(0)

        train_loss = running_train / max(n_train, 1)
        history["train_loss"].append(train_loss)

        # -----------------------------
        # Validation
        # -----------------------------
        if val_loader:
            model.eval()
            running_val = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_Xc, batch_Xk in val_loader:
                    assert batch_Xk.shape[1] == len(config.cat_dims), "Batch categorical shape mismatch!"
                    
                    batch_Xc = batch_Xc.to(device)
                    batch_Xk = batch_Xk.to(device)
                    
                    recon = model(batch_Xc, batch_Xk)
                    per_feature = criterion(recon, batch_Xc)
                    v_loss = per_feature.mean()

                    running_val += v_loss.item() * batch_Xc.size(0)
                    n_val += batch_Xc.size(0)

            val_loss = running_val / max(n_val, 1)
            history["val_loss"].append(val_loss)

            if scheduler is not None and config.lr_scheduler != "onecycle":
                if config.lr_scheduler == "plateau":
                    scheduler.step(val_loss)
                else:  # cosine or onecycle
                    scheduler.step()

            # -----------------------------
            # Early stopping
            # -----------------------------
            if val_loss < best_metric - 1e-9:
                best_metric = val_loss
                best_state = model.state_dict()
                history["best_epoch"] = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # -----------------------------
        # Full training / No Validation
        # -----------------------------
        else:
            if scheduler is not None and config.lr_scheduler != "onecycle":
                if config.lr_scheduler == "plateau":
                    scheduler.step(train_loss)
                else:  # cosine or onecycle
                    scheduler.step()

            # -----------------------------
            # Track best loss
            # -----------------------------
            if train_loss < best_metric - 1e-9:
                best_metric = train_loss
                best_state = model.state_dict()
                history["best_epoch"] = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)

        print(
            f"[Epoch {epoch+1}/{config.num_epochs}] "
            f"train_loss={train_loss:.6f} "
            + (f" val_loss={val_loss:.6f} " if val_loader else "")
            + f" lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # print extra info every 10 epochs
        if epoch % 10 == 0:
            # show gradient norms for debugging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Grad norm: {total_norm:.4f}")

    # restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return model, history


#########################################
##         SAVE / LOAD HELPERS         ##
#########################################

def save_autoencoder(
    model: TabularAutoencoder,
    config: AEConfig,
    path: Path,
) -> None:
    """Save model weights + config to a single .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, path)
    print(f"[OK] Saved autoencoder to {path}")


def load_autoencoder(path: Path) -> tuple[TabularAutoencoder, AEConfig]:
    """Load model + config from a .pt file."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu")

    cfg = AEConfig(**payload["config"])
    model = TabularAutoencoder(
        num_cont=cfg.num_cont,
        cat_dims=cfg.cat_dims,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        embedding_dim=cfg.embedding_dim,
        continuous_noise_std=cfg.continuous_noise_std,
        residual_strength=cfg.residual_strength,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model, cfg
