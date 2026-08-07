from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.src.ml.models.ae import AEConfig, TabularAE
from app.src.ml.models.vae import VAEConfig, TabularVAE


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
    val_cont: np.ndarray | None,
    val_cat: np.ndarray | None,
    config: AEConfig | VAEConfig,
    loss_weights: dict[str, float] | None = None,
) -> tuple[TabularAE | TabularVAE, dict[str, Any]]:
    """
    Train autoencoder with cont + cat input features on split dataset with early stopping and val metrics OR full dataset with no early stopping.
   
    Returns
    -------
    model : TabularAE or TabularVAE
    history : dict
    """

    if loss_weights is None:
        loss_weights = {"cont_w": float(1/config.num_cont), "cat_w": float(1/len(cat_dims.keys()))}
    
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
    if isinstance(config, AEConfig):
        model = TabularAE(config=config).to(device)
        cont_loss_name, cat_loss_name = "cont_loss", "cat_loss"
        cont_short, cat_short = "Cont", "Cat"
    elif isinstance(config, VAEConfig):
        model = TabularVAE(
            num_cont=config.num_cont,
            cat_dims=config.cat_dims,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            embedding_dim=config.embedding_dim,
            continuous_noise_std=config.continuous_noise_std,
            activation_en=config.activation_en,
            activation_de=config.activation_de
        ).to(device)
        cont_loss_name, cat_loss_name = "recon_loss", "kl_loss"
        cont_short, cat_short = "Recon", "KL"
    else: 
        raise ValueError(f"[ERROR] Unsupported model config type; exprected AEConfig or VAEConfig.")

    # -------------------------
    # Model-specific option
    # -------------------------
    use_beta_annealing = isinstance(config, VAEConfig) and hasattr(model, "set_beta_annealing")

    if config.warmup_epochs > 0:
        print(
            f"[INFO] Cat-only warmup enabled for "
            f"{config.warmup_epochs} epochs"
        )

    # -------------------------------
    # Benchmark toggles
    # -------------------------------
    if hasattr(model, "debug_kl_stats"):
        model.debug_kl_stats = False 

    # -------------------------
    # Optimizer
    # -------------------------
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=1e-8
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=1e-8
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # -------------------------
    # LR scheduler
    # -------------------------
    if config.lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=config.num_epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True
        )
    elif config.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1,
            patience=10,
            min_lr=1e-6,
        )
    elif config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs,
            eta_min=1e-6
        )
    elif config.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # -------------------------
    # History tracking
    # -------------------------
    history = {
        "train_loss": [],
        f"train_{cont_loss_name}": [],
        f"train_{cat_loss_name}": [],
        "val_loss": [] if val_loader else None,
        f"val_{cont_loss_name}": [] if val_loader else None,
        f"val_{cat_loss_name}": [] if val_loader else None,
        "warmup": [],
        "learning_rates": [],
        "best_epoch": 0,
        "config": asdict(config),
        "loss_weights": loss_weights,
    }

    best_metric = float("inf")
    best_state = None
    no_improve = 0
    
    print(f"[INFO] Training autoencoder with:")
    print(f"\tCont features        {config.num_cont}")
    print(f"\tCat features         {len(config.cat_dims)}")
    print(f"\tTransformation Cont  {'RobustScaler'}")
    print(f"\t               Cat   {'Embedding layers' if config.use_embedding else 'One-Hot encoding'}")
    print(f"\tHidden dimensions    {config.hidden_dims}")
    print(f"\tLatent dimension     {config.latent_dim}")
    print(f"\tActivation encoder   {config.activation_en}")
    print(f"\tActivation decoder   {config.activation_de}")
    print(f"\tOptimizer            {config.optimizer}")
    print(f"\tLR scheduler         {config.lr_scheduler}")
    print(f"\tNoise injection      {'enabled' if config.allow_noise_injection else 'disabled'}")
    print(f"\tNum of epochs        {config.num_epochs}")
    print(f"\tNum of warmup epochs {config.warmup_epochs}")
    print(f"\tDevice               {device}")
    print(f"\tLoss weights Cont    {loss_weights['cont_w']:.5f}")
    print(f"\t             Cat     {loss_weights['cat_w']:.5f}")

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(config.num_epochs):
        model.train()    

        # -------------------------
        # Cat-only warmup setup
        # -------------------------
        in_warmup = epoch < config.warmup_epochs

        if in_warmup:
            # freeze cont heads
            for p in model.D.cont_recon_head.parameters():
                p.requires_grad = False
            # adjust lr to avoid overfitting
            for g in optimizer.param_groups:
                g["lr"] = config.lr * float((epoch+1) / config.warmup_epochs)
        else:
            # unfreeze cont heads
            for p in model.D.cont_recon_head.parameters():
                p.requires_grad = True
            # adjust lr to avoid overfitting
            for g in optimizer.param_groups:
                g["lr"] = config.lr

        # -------------------------
        # Setup training
        # -------------------------
        epoch_train_loss = 0.0
        epoch_train_cont = 0.0
        epoch_train_cat = 0.0
        n_train_batches = 0

        if use_beta_annealing:
            model.set_beta_annealing(
                epoch=epoch,
                total_epochs=config.num_epochs,
                beta_max=config.beta,
                schedule="linear" # or "cyclic"
            )
        
        for batch_Xc, batch_Xk in train_loader:
            batch_Xc = batch_Xc.to(device, non_blocking=True)
            batch_Xk = batch_Xk.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if isinstance(config, AEConfig):
                cont_recon, cat_logits = model(batch_Xc, batch_Xk)
                cont_loss, cat_loss, total_loss = model.scoring(
                    batch_Xc, 
                    batch_Xk, 
                    cont_recon,
                    cat_logits,
                    loss_weights,
                    in_warmup
                )

            elif isinstance(config, VAEConfig):
                total_loss, recon_loss, kl_loss = model.elbo_loss(
                    batch_Xc,
                    batch_Xk,
                    beta=config.beta,
                    cont_w=loss_weights["cont_w"],
                    cat_w=loss_weights["cat_w"],
                    temperature=config.temperature,
                    reduction="mean",
                )
                cont_loss, cat_loss = recon_loss, kl_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if config.gradient_clip is not None and config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.gradient_clip
                )
            
            optimizer.step()

            if scheduler is not None and config.lr_scheduler == "onecycle":
                scheduler.step()
            
            # Accumulate metrics
            batch_size = batch_Xc.size(0)
            epoch_train_loss += total_loss.item() * batch_size
            epoch_train_cont += cont_loss.sum().item()
            epoch_train_cat += cat_loss.sum().item()
            n_train_batches += batch_size

        # Calculate epoch averages
        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        avg_train_cont = epoch_train_cont / max(n_train_batches, 1)
        avg_train_cat = epoch_train_cat / max(n_train_batches, 1)
        
        history["train_loss"].append(avg_train_loss)
        history[f"train_{cont_loss_name}"].append(avg_train_cont)
        history[f"train_{cat_loss_name}"].append(avg_train_cat)
        history["warmup"].append(in_warmup),

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

                    if isinstance(config, AEConfig):
                        cont_recon, cat_logits = model(batch_Xc, batch_Xk)
                        cont_loss, cat_loss, total_loss = model.scoring(
                            batch_Xc, 
                            batch_Xk,
                            cont_recon,
                            cat_logits,
                            loss_weights,
                        )
                    elif isinstance(config, VAEConfig):
                        total_loss, recon_loss, kl_loss = model.elbo_loss(
                            batch_Xc,
                            batch_Xk,
                            beta=None,
                            cont_w=loss_weights["cont_w"],
                            cat_w=loss_weights["cat_w"],
                            temperature=config.temperature,
                            reduction="mean",
                        )
                        cont_loss, cat_loss = recon_loss, kl_loss

                    # Accumulate
                    batch_size = batch_Xc.size(0)
                    epoch_val_loss += total_loss.item() * batch_size
                    epoch_val_cont += cont_loss.sum().item()
                    epoch_val_cat += cat_loss.sum().item()
                    n_val_batches += batch_size

            # Calculate validation averages
            avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
            avg_val_cont = epoch_val_cont / max(n_val_batches, 1)
            avg_val_cat = epoch_val_cat / max(n_val_batches, 1)
            
            history["val_loss"].append(avg_val_loss)
            history[f"val_{cont_loss_name}"].append(avg_val_cont)
            history[f"val_{cat_loss_name}"].append(avg_val_cat)

            if scheduler is not None:
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_train_loss)
                elif config.lr_scheduler in ["cosine", "step"]:
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

            mode = "WARMUP" if in_warmup else "FULL"
            print(f"Epoch {epoch + 1:2d}/{config.num_epochs}: "
                  f"Mode {mode:<6} | "
                  f"Train loss: {avg_train_loss:.6f} "
                  f"({cont_short}: {avg_train_cont:.6f}, {cat_short}: {avg_train_cat:.6f}) | "
                  f"Val loss: {avg_val_loss:.6f} "
                  f"({cont_short}: {avg_val_cont:.6f}, {cat_short}: {avg_val_cat:.6f}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
        # -----------------------------
        # Full training / No Validation
        # -----------------------------
        else:
            # -----------------------------
            # Track best loss
            # -----------------------------
            if avg_train_loss < best_metric - 1e-9:
                best_metric = avg_train_loss
                best_state = model.state_dict().copy()
                history["best_epoch"] = epoch + 1

            mode = "WARMUP" if in_warmup else "FULL"
            print(f"Epoch {epoch + 1:2d}/{config.num_epochs}: "
                    f"Mode {mode:<6} | "
                    f"Loss: {avg_train_loss:.6f} "
                    f"({cont_short}: {avg_train_cont:.6f}, {cat_short}: {avg_train_cat:.6f}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)

        # print extra gradient norms every 10 epochs
        if (epoch + 1) % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.detach().data.pow(2).sum().item()
            total_norm = total_norm ** 0.5
            print(f"  Gradient norm: {total_norm:.4f}")

    # -----------------------------
    # Restore best model
    # -----------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {history['best_epoch']}")
    
    return model.eval(), history


#########################################
##         SAVE / LOAD HELPERS         ##
#########################################

def save_autoencoder(
    model: TabularAE | TabularVAE,
    config: AEConfig | VAEConfig,
    cat_dims: dict[str, int],
    num_cont: int,
    path: Path,
    additional_info: dict[str, Any] | None = None
) -> None:
    """Save model weights + config to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "cat_dims": cat_dims,
        "num_cont": num_cont,
        "model_class": model.__class__.__name__,
        "additional_info": additional_info or {},
    }
    torch.save(payload, path)

    print(f"[OK] Saved autoencoder to {path}")


def load_autoencoder(
    path: Path,
    device: str = "cpu"
) -> tuple[TabularAE | TabularVAE, AEConfig | VAEConfig, int, dict[str, int]]:
    """Load model + config from a .pt file."""
    payload = torch.load(path, map_location=device, weights_only=True)

    num_cont = payload["num_cont"]
    cat_dims = payload["cat_dims"]

    ae_class = payload["model_class"]
    if ae_class == "TabularAE":
        cfg = AEConfig(**payload["config"])
        model = TabularAE(config=cfg)
    elif ae_class == "TabularVAE":
        cfg = VAEConfig(**payload["config"])
        model = TabularVAE(
            num_cont=num_cont,
            cat_dims=cat_dims,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            embedding_dim=cfg.embedding_dim,
            continuous_noise_std=cfg.continuous_noise_std,
            activation_en=cfg.activation_en,
            activation_de=cfg.activation_de
        )
    else:
        raise ValueError(f"Unknown model_class: {ae_class}")
    
    model.load_state_dict(payload["state_dict"])
    target_device = torch.device(cfg.device)
    model = model.to(target_device)
    
    print(f"[INFO] Loaded autoencoder from {path}")
    print(f"[INFO] Model moved to device: {target_device}")
    
    return model.eval(), cfg, num_cont, cat_dims