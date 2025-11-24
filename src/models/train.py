from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.autoencoder import AEConfig, TabularAutoencoder


#########################################
##           TRAINING HELPERS          ##
#########################################

def _make_dataloaders(
    train_cont: np.ndarray, 
    train_cat: np.ndarray,
    val_cont: np.ndarray, 
    val_cat: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for:
       X_cont : continuous scaled features (float32)
       X_cat  : categorical integer features (int64)
    """
    Xc_train = torch.from_numpy(train_cont.astype(np.float32))
    Xk_train = torch.from_numpy(train_cat.astype(np.int64))

    Xc_val = torch.from_numpy(val_cont.astype(np.float32))
    Xk_val = torch.from_numpy(val_cat.astype(np.int64))
   
    train_ds = TensorDataset(Xc_train, Xk_train)
    val_ds   = TensorDataset(Xc_val, Xk_val)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
    )

    return train_loader, val_loader


def train_autoencoder(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    val_cont: np.ndarray,
    val_cat: np.ndarray,
    config: AEConfig,
) -> tuple[TabularAutoencoder, dict[str, Any]]:
    """
    Train autoencoder with continuous + categorical inputs.

    X_cont : (T, D_cont) float32 scaled features
    X_cat  : (T, 3) int64 categorical indices

    Returns
    -------
    model : TabularAutoencoder
    history : dict
        Contains train/val loss curves and best_epoch.
    """

    device = torch.device(config.device)

    # assert X_cat.shape[1] == len(config.cat_dims), f"Mismatch: X_cat has {X_cat.shape[1]} cols, but cat_dims defines {len(config.cat_dims)} categories"

    # DataLoaders
    train_loader, val_loader = _make_dataloaders(
        train_cont, 
        train_cat,
        val_cont, 
        val_cat,
        batch_size=config.batch_size, 
    )

    # Model
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

    criterion = torch.nn.MSELoss(reduction="none") #weighted loss that emphasizes rare feature patterns TODO: best here?

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "best_epoch": None,
    }

    for epoch in range(config.num_epochs):
        # -----------------------------
        # Training
        # -----------------------------
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

            # gradient clipping (optional)
            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()

            if scheduler is not None and config.lr_scheduler == "onecycle":
                scheduler.step()

            running_train += loss.item() * batch_Xc.size(0)
            n_train += batch_Xc.size(0)

        train_loss = running_train / max(n_train, 1)

        # -----------------------------
        # Validation
        # -----------------------------
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
                loss = per_feature.mean()

                running_val += loss.item() * batch_Xc.size(0)
                n_val += batch_Xc.size(0)

        val_loss = running_val / max(n_val, 1)

        if scheduler is not None and config.lr_scheduler != "onecycle":
            if config.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:  # cosine or onecycle
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(current_lr)

        print(
            f"[Epoch {epoch+1:03d}/{config.num_epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"lr={current_lr:.2e} "
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

        # -----------------------------
        # Early stopping
        # -----------------------------
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = model.state_dict()
            history["best_epoch"] = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

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
