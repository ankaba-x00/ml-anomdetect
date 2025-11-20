from dataclasses import asdict
from pathlib import Path
from typing import Any
from src.models.autoencoder import AEConfig, TabularAutoencoder
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


#########################################
##           TRAINING HELPERS          ##
#########################################

def _make_dataloaders(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    batch_size: int,
    val_split: float,
    time_series_split: bool,
) -> tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for:
       X_cont : continuous scaled features (float32)
       X_cat  : categorical integer features (int64)
    """
    Xc = torch.from_numpy(X_cont.astype(np.float32))
    Xk = torch.from_numpy(X_cat.astype(np.int64))
    n_total = len(Xc)

    if time_series_split: 
        # chronological split: FIRST segment train, LAST segment validation: TODO: random sampling better?
        n_train = int(n_total * (1 - val_split))
        train_ds = TensorDataset(Xc[:n_train], Xk[:n_train])
        val_ds = TensorDataset(Xc[n_train:], Xk[n_train:])
    else:
        # random split
        full_ds = TensorDataset(Xc, Xk)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=not time_series_split, # no shuffle for timeseries
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )

    return train_loader, val_loader


def train_autoencoder(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
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

    assert X_cat.shape[1] == len(config.cat_dims), f"Mismatch: X_cat has {X_cat.shape[1]} cols, but cat_dims defines {len(config.cat_dims)} categories"

    # DataLoaders
    train_loader, val_loader = _make_dataloaders(
        X_cont=X_cont,
        X_cat=X_cat,
        batch_size=config.batch_size, 
        val_split=config.val_split,
        time_series_split=config.time_series_split,
    )

    # Model
    model = TabularAutoencoder(
        num_cont=config.num_cont,
        cat_dims=config.cat_dims,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', # monitor validation loss going down
            factor=0.5, # reduce LR by half when plateau
            patience=3, # wait 3 epochs without improvement
            min_lr=1e-6 # minimum learning rate
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
            
            # criterion returns (batch, feature_dim)
            per_feature_loss = criterion(recon, batch_Xc)
            per_sample_loss = per_feature_loss.mean(dim=1)
            loss = per_sample_loss.mean()
            
            loss.backward()

            # gradient clipping (optional)
            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()

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

                per_feature_loss = criterion(recon, batch_Xc)
                per_sample_loss = per_feature_loss.mean(dim=1)
                loss = per_sample_loss.mean()

                running_val += loss.item() * batch_Xc.size(0)
                n_val += batch_Xc.size(0)

        val_loss = running_val / max(n_val, 1)

        if scheduler is not None:
            scheduler.step(val_loss)
        
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
        if val_loss < best_val_loss - 1e-8:
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
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model, cfg
