from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from app.src.models.autoencoder import AEConfig, TabularAutoencoder


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
        pin_memory=True if torch.cuda.is_available() else False
    )

def train_autoencoder(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    val_cont: Optional[np.ndarray],
    val_cat: Optional[np.ndarray],
    config: AEConfig,
    loss_weights: Optional[dict] = None,
) -> tuple[TabularAutoencoder, dict[str, Any]]:
    """
    Train autoencoder with continuous + categorical input features on split dataset with early stopping and val metrics OR full dataset with no early stopping.
   
    Returns
    -------
    model : TabularAutoencoder
    history : dict
    """

    if loss_weights is None:
        loss_weights = {"cont_weight": 1.0, "cat_weight": 0.0}

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
        if val_cont is not None and val_cat is not None
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
        activation=config.activation
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
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # -------------------------
    # LR scheduler
    # -------------------------
    scheduler = None
    if config.use_lr_scheduler:
        if config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=True
            )
        elif config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.num_epochs,
                eta_min=1e-6
            )
        elif config.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.lr,
                total_steps=config.num_epochs * len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif config.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=20,
                gamma=0.5
            )

    # -------------------------
    # History tracking
    # -------------------------
    history = {
        "train_loss": [],
        "train_cont_loss": [],
        "train_cat_loss": [],
        "val_loss": [] if val_loader else None,
        "val_cont_loss": [] if val_loader else None,
        "val_cat_loss": [] if val_loader else None,
        "learning_rates": [],
        "best_epoch": 0,
        "config": asdict(config),
        "loss_weights": loss_weights,
    }

    best_metric = float("inf")
    best_state = None
    no_improve = 0

    cat_names = list(config.cat_dims.keys())
    
    print(f"Training autoencoder with:")
    print(f"  Continuous features: {config.num_cont}")
    print(f"  Categorical features: {len(config.cat_dims)}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Hidden dimensions: {config.hidden_dims}")
    print(f"  Device: {device}")
    print(f"  Loss weights - Continuous: {loss_weights['cont_weight']}, ")
    print(f"                 Categorical: {loss_weights['cat_weight']}")

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(config.num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_cont = 0.0
        epoch_train_cat = 0.0
        n_train_batches = 0

        for batch_Xc, batch_Xk in train_loader:
            batch_Xc = batch_Xc.to(device, non_blocking=True)
            batch_Xk = batch_Xk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            cont_recon, cat_recons = model(batch_Xc, batch_Xk, temperature=config.temperature)
            
            # Continuous loss (MSE)
            cont_loss = ((cont_recon - batch_Xc) ** 2).mean()

            # Categorical loss (Cross-entropy)
            cat_loss = 0.0
            for i, name in enumerate(cat_names):
                cat_loss += F.cross_entropy(
                    cat_recons[name],
                    batch_Xk[:, i].long()
                )
            cat_loss = cat_loss / len(cat_names)  # Average over categorical features

            # Weighted total loss
            total_loss = (
                loss_weights["cont_weight"] * cont_loss +
                loss_weights["cat_weight"] * cat_loss
            )
            
            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if config.gradient_clip and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()

            if scheduler is not None and config.lr_scheduler == "onecycle":
                scheduler.step()
            
            # Accumulate metrics
            batch_size = batch_Xc.size(0)
            epoch_train_loss += total_loss.item() * batch_size
            epoch_train_cont += cont_loss.item() * batch_size
            epoch_train_cat += cat_loss.item() * batch_size
            n_train_batches += batch_size

        # Calculate epoch averages
        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        avg_train_cont = epoch_train_cont / max(n_train_batches, 1)
        avg_train_cat = epoch_train_cat / max(n_train_batches, 1)
        
        history["train_loss"].append(avg_train_loss)
        history["train_cont_loss"].append(avg_train_cont)
        history["train_cat_loss"].append(avg_train_cat)

        # -----------------------------
        # Validation
        # -----------------------------
        if val_loader:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_cont = 0.0
            epoch_val_cat = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch_Xc, batch_Xk in val_loader:
                    batch_Xc = batch_Xc.to(device, non_blocking=True)
                    batch_Xk = batch_Xk.to(device, non_blocking=True)

                    # Forward pass
                    cont_recon, cat_recons = model(batch_Xc, batch_Xk, temperature=config.temperature)
                    
                    # Continuous loss
                    cont_loss = ((cont_recon - batch_Xc) ** 2).mean()
                    
                    # Categorical loss
                    cat_loss = 0.0
                    for i, name in enumerate(cat_names):
                        cat_loss += F.cross_entropy(
                            cat_recons[name],
                            batch_Xk[:, i].long()
                        )
                    cat_loss = cat_loss / len(cat_names)
                    
                    # Weighted total loss
                    total_loss = (
                        loss_weights["cont_weight"] * cont_loss +
                        loss_weights["cat_weight"] * cat_loss
                    )

                    # Accumulate
                    batch_size = batch_Xc.size(0)
                    epoch_val_loss += total_loss.item() * batch_size
                    epoch_val_cont += cont_loss.item() * batch_size
                    epoch_val_cat += cat_loss.item() * batch_size
                    n_val_batches += batch_size

            # Calculate validation averages
            avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
            avg_val_cont = epoch_val_cont / max(n_val_batches, 1)
            avg_val_cat = epoch_val_cat / max(n_val_batches, 1)
            
            history["val_loss"].append(avg_val_loss)
            history["val_cont_loss"].append(avg_val_cont)
            history["val_cat_loss"].append(avg_val_cat)

            if scheduler is not None and config.lr_scheduler != "onecycle":
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_val_loss)
                else:  # cosine or onecycle
                    scheduler.step()

            # -----------------------------
            # Early stopping
            # -----------------------------
            if avg_val_loss < best_metric - 1e-9:
                best_metric = avg_val_loss
                best_state = model.state_dict().copy()
                history["best_epoch"] = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"Epoch {epoch + 1:3d}/{config.num_epochs}: "
                  f"Train Loss: {avg_train_loss:.6f} "
                  f"(C: {avg_train_cont:.6f}, K: {avg_train_cat:.6f}) | "
                  f"Val Loss: {avg_val_loss:.6f} "
                  f"(C: {avg_val_cont:.6f}, K: {avg_val_cat:.6f}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
        # -----------------------------
        # Full training / No Validation
        # -----------------------------
        else:
            if scheduler is not None and config.lr_scheduler != "onecycle":
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_train_loss)
                else:  # cosine or onecycle
                    scheduler.step()

            # -----------------------------
            # Track best loss
            # -----------------------------
            if avg_train_loss < best_metric - 1e-9:
                best_metric = avg_train_loss
                best_state = model.state_dict().copy()
                history["best_epoch"] = epoch + 1

            print(f"Epoch {epoch + 1:3d}/{config.num_epochs}: "
                    f"Loss: {avg_train_loss:.6f} "
                    f"(C: {avg_train_cont:.6f}, K: {avg_train_cat:.6f}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)

        # print extra gradient norms every 10 epochs
        if (epoch + 1) % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Gradient norm: {total_norm:.4f}")

    # -----------------------------
    # Restore best model
    # -----------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {history['best_epoch']}")
    
    model.eval()
    return model, history


#########################################
##         SAVE / LOAD HELPERS         ##
#########################################

def save_autoencoder(
    model: TabularAutoencoder,
    config: AEConfig,
    path: Path,
    additional_info: Optional[dict] = None
) -> None:
    """Save model weights + config to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    cat_dims_from_model = model.cat_dims

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "cat_dims": cat_dims_from_model,
        "num_cont": model.num_cont,
        "model_class": "TabularAutoencoder",
        "additional_info": additional_info or {},
    }
    torch.save(payload, path)
    print(f"[OK] Saved autoencoder to {path}")


def load_autoencoder(
        path: Path,
        device: Optional[str] = None
    ) -> tuple[TabularAutoencoder, AEConfig]:
    """Load model + config from a .pt file."""
    if torch.cuda.is_available():
        payload = torch.load(path, map_location="cuda")
    else:
        payload = torch.load(path, map_location="cpu")

    cfg = AEConfig(**payload["config"])

    if device is not None:
        cfg.device = device

    model = TabularAutoencoder(
        num_cont=cfg.num_cont,
        cat_dims=cfg.cat_dims,
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        embedding_dim=cfg.embedding_dim,
        continuous_noise_std=cfg.continuous_noise_std,
        residual_strength=cfg.residual_strength,
        activation=cfg.activation
    )
    model.load_state_dict(payload["state_dict"])
    target_device = torch.device(cfg.device)
    model = model.to(target_device)
    model.eval()
    
    print(f"[INFO] Loaded autoencoder from {path}")
    print(f"[INFO] Model moved to device: {target_device}")
    
    return model, cfg