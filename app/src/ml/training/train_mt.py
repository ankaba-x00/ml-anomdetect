from dataclasses import asdict
import numpy as np
import torch
from typing import Any

from app.src.ml.models.configs import MTAEConfig
from app.src.ml.models.helpers import supervised_dataloader
from app.src.ml.models.mtae import MTTabularAE


def train_mt_autoencoder(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    train_l3: np.ndarray,
    train_l7: np.ndarray,
    train_attack: np.ndarray,
    val_cont: np.ndarray | None,
    val_cat: np.ndarray | None,
    val_l3: np.ndarray | None,
    val_l7: np.ndarray | None,
    val_attack: np.ndarray | None,
    config: MTAEConfig,
    loss_weights: dict[str, float],
) -> tuple[MTTabularAE, dict[str, Any]]:
    """
    Train multi-task autoencoder with cont and cat features on split dataset with early stopping OR full dataset with no early stopping.
    
    Returns
    -------
    model : MTTabularAE
    history : dict
    """

    device = torch.device(config.device)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = supervised_dataloader(
        train_cont, 
        train_cat,
        train_l3, 
        train_l7, 
        train_attack,
        config.batch_size,
        shuffle=True
    )

    val_loader = None
    if val_cont is not None:
        val_loader = supervised_dataloader(
            val_cont, 
            val_cat,
            val_l3, 
            val_l7, 
            val_attack,
            config.batch_size,
            shuffle=False
        )

    # -------------------------
    # Build model and set configs
    # -------------------------
    model = MTTabularAE(config).to(device)

    if config.warmup_epochs > 0:
        print(
            f"[INFO] Recon-only warmup enabled for "
            f"{config.warmup_epochs} epochs"
        )
    
    train_at_counts = np.bincount(
        train_attack, 
        minlength=config.n_attack_types
    ).tolist()
    if val_cont is not None:
        val_at_counts = np.bincount(
            val_attack, 
            minlength=config.n_attack_types
        ).tolist()
    else:
        val_at_counts = None
    attack_type_weights = model.compute_attack_type_weights(
        train_attack,
        config.n_attack_types,
    )
    
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
    # LR Scheduler
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
    # Set history and param tracking
    # -------------------------
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_l3": [],
        "train_l7": [],
        "train_at": [],
        "train_cont": [],
        "train_cat": [],
        "val_recon": [],
        "val_loss": [] if val_loader else None,
        "val_l3": [] if val_loader else None,
        "val_l7": [] if val_loader else None,
        "val_at": [] if val_loader else None,
        "val_cont": [],
        "val_cat": [],
        "learning_rates": [],
        "best_epoch": 0,
        "config": asdict(config),
        "loss_weights": loss_weights,
        "attack_type_weights": [],
        "train_at_counts": train_at_counts
    }
    if val_cont is not None:
        history["val_at_counts"] = val_at_counts

    best_metric = float("inf")
    best_state = None
    no_improve = 0
    in_warmup = False
    in_stepup = False

    # -------------------------
    # Print config summary
    # -------------------------
    print(f"[INFO] Training autoencoder with:")
    print(f"\tCont features        {config.num_cont}")
    print(f"\tCat features         {len(config.cat_dims)}")
    print(f"\tTransformation Cont  {'RobustScaler'}")
    print(f"\t               Cat   {'Embedding layers' if config.use_embedding else 'One-Hot encoding'}")
    print(f"\tBatch size           {config.batch_size}")
    print(f"\tHidden dimensions    {config.hidden_dims}")
    print(f"\tLatent dimension     {config.latent_dim}")
    print(f"\tActivation encoder   {config.activation_en}")
    print(f"\tActivation decoder   {config.activation_de}")
    print(f"\tOptimizer            {config.optimizer}")
    print(f"\tLR scheduler         {config.lr_scheduler}")
    print(f"\tNoise injection      {'enabled' if config.allow_noise_injection else 'disabled'}")
    print(f"\tNum of warmup epochs {config.warmup_epochs}")
    print(f"\tAlpha max            {config.alpha}")
    print(f"\tLoss weights Cont    {loss_weights['cont_w']:.5f}")
    print(f"\t             Cat     {loss_weights['cat_w']:.5f}")
    print(f"\t             L3      {loss_weights['l3_w']:.5f}")
    print(f"\t             L7      {loss_weights['l7_w']:.5f}")
    print(f"\t             At      {loss_weights['at_w']:.5f}")

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(config.num_epochs):
        model.train()

        # -------------------------
        # Warmup setup
        # -------------------------
        in_warmup = epoch < config.warmup_epochs
        if in_warmup:
            # freeze regression heads
            for p in model.D.l3_head.parameters():
                p.requires_grad = False
            for p in model.D.l7_head.parameters():
                p.requires_grad = False
            for p in model.D.at_head.parameters():
                p.requires_grad = False
            # adjust lr to avoid overfitting
            for g in optimizer.param_groups:
                    g["lr"] = config.lr * float((epoch+1) / config.warmup_epochs)
        else:
            # unfreeze after warmup
            for p in model.D.l3_head.parameters():
                p.requires_grad = True
            for p in model.D.l7_head.parameters():
                p.requires_grad = True
            for p in model.D.at_head.parameters():
                p.requires_grad = True
            # adjust lr
            for g in optimizer.param_groups:
                g["lr"] = config.lr
        in_stepup = config.warmup_epochs <= epoch < config.stepup_epochs + config.warmup_epochs 
        model.set_alpha(epoch, in_warmup, in_stepup)
        
        # -------------------------
        # Setup training
        # -------------------------
        tl, tr, tc, tk, tl3, tl7, tatt, tn = 0, 0, 0, 0, 0, 0, 0, 0

        for bXc, bXk, by3, by7, bya in train_loader:
            bXc, bXk = bXc.to(device), bXk.to(device)
            by3, by7, bya = by3.to(device), by7.to(device), bya.to(device)

            optimizer.zero_grad(set_to_none=True)

            cont_recon, cat_logits, l3_pred, l7_pred, at_logits = model(bXc, bXk)
            cont_loss, cat_loss, recon_score, l3_loss, l7_loss, at_loss, total_loss = model.scoring(
                    bXc, 
                    bXk, 
                    by3,
                    by7,
                    bya,
                    cont_recon,
                    cat_logits,
                    l3_pred,
                    l7_pred,
                    at_logits,
                    loss_weights,
                    attack_type_weights,
                    in_warmup
                )

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
            bs = bXc.size(0)
            tl += total_loss.item() * bs
            tr += recon_score.sum().item()
            tl3 += l3_loss.sum().item()
            tl7 += l7_loss.sum().item()
            tatt += at_loss.sum().item()
            tc += cont_loss.sum().item()
            tk += cat_loss.sum().item()
            tn += bs
        
        # Calculate epoch averages
        avg_tl = tl / max(1, tn)
        avg_train_recon = tr / max(1, tn)
        avg_tl3 = tl3 / max(1, tn)
        avg_tl7 = tl7 / max(1, tn)
        avg_tatt = tatt / max(1, tn)
        avg_tc = tc / max(1, tn)
        avg_tk = tk / max(1, tn)
        
        history["train_loss"].append(avg_tl)
        history["train_recon"].append(avg_train_recon)
        history["train_l3"].append(avg_tl3)
        history["train_l7"].append(avg_tl7)
        history["train_at"].append(avg_tatt)
        history["train_cont"].append(avg_tc)
        history["train_cat"].append(avg_tk)
        history["attack_type_weights"] = (
            attack_type_weights.cpu().tolist()
            if attack_type_weights is not None
            else []
        )

        # -------------------------
        # Validation
        # -------------------------
        if val_loader:
            model.eval()
            vl, vr, vc, vk, vl3, vl7, vatt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            vn = 0

            with torch.no_grad():
                for bXc, bXk, by3, by7, bya in val_loader:
                    bXc, bXk = bXc.to(device), bXk.to(device)
                    by3, by7, bya = by3.to(device), by7.to(device), bya.to(device)

                    cont_recon, cat_logits, l3_pred, l7_pred, at_logits = model(bXc, bXk)
                    cont_loss, cat_loss, recon_score, l3_loss, l7_loss, at_loss, total_loss = model.scoring(
                            bXc, 
                            bXk, 
                            by3,
                            by7,
                            bya,
                            cont_recon,
                            cat_logits,
                            l3_pred,
                            l7_pred,
                            at_logits,
                            loss_weights,
                            attack_type_weights,
                            in_warmup
                        )

                    bs = bXc.size(0)
                    vl += total_loss.item() * bs
                    vr += recon_score.sum().item()
                    vl3 += l3_loss.sum().item()
                    vl7 += l7_loss.sum().item()
                    vatt += at_loss.sum().item()
                    vc += cont_loss.sum().item()
                    vk += cat_loss.sum().item()
                    vn += bs
                
            # Calculate validation averages
            avg_vl = vl / max(1, vn)
            avg_vr = vr / max(1, vn)
            avg_vl3 = vl3 / max(1, vn)
            avg_vl7 = vl7 / max(1, vn)
            avg_vatt = vatt / max(1, vn)
            avg_vc = vc / max(1, vn)
            avg_vk = vk / max(1, vn)
                
            history["val_loss"].append(avg_vl)
            history["val_recon"].append(avg_vr)
            history["val_l3"].append(avg_vl3)
            history["val_l7"].append(avg_vl7)
            history["val_at"].append(avg_vatt)
            history["val_cont"].append(avg_vc)
            history["val_cat"].append(avg_vk)

            if scheduler is not None:
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_tl)
                elif config.lr_scheduler in ["cosine", "step"]:
                    scheduler.step()
            
            # -----------------------------
            # Early stopping
            # -----------------------------
            if not in_warmup and not in_stepup:
                if avg_vl < best_metric - 1e-9:
                    best_metric = avg_vl
                    best_state = model.state_dict().copy()
                    history["best_epoch"] = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= config.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # -----------------------------
            # Print summary
            # -----------------------------
            # mode = "WARMUP" if in_warmup else "FULL"
            if in_warmup:
                mode = "WARMUP"
            elif in_stepup:
                mode = "STEPUP"
            else:
                mode = "FULL"
            print(
                f"Epoch {epoch+1:2d}/{config.num_epochs} - "
                f"Mode {mode:<3} | "
                f"TRAIN: {avg_tl:.4f} "
                f"(rc: {avg_train_recon:.4f}, l3: {avg_tl3:.4f}, l7: {avg_tl7:.4f}, la: {avg_tatt:.4f}) | "
                f"VAL {avg_vl:.4f} "
                f"(rc: {avg_vr:.4f}, l3: {avg_vl3:.4f}, l7: {avg_vl7:.4f}, at: {avg_vatt:.4f}) | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}, ɑ {model.current_alpha:.1f}"
            )
        
        # -----------------------------
        # Full training / No Validation
        # -----------------------------
        else:
            # -----------------------------
            # Track best loss
            # -----------------------------
            if avg_tl < best_metric - 1e-9:
                best_metric = avg_tl
                best_state = model.state_dict().copy()
                history["best_epoch"] = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # -----------------------------
            # Print summary
            # -----------------------------
            mode = "WARMUP" if in_warmup else "FULL"
            print(
                f"Epoch {epoch+1:2d}/{config.num_epochs} - "
                f"Mode {mode:<3} | "
                f"TRAIN: {avg_tl:.4f} | "
                f"(rc: {avg_train_recon:.4f}, l3: {avg_tl3:.4f}, l7: {avg_tl7:.4f}, at: {avg_tatt:.4f}) | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}, ɑ {model.current_alpha:.1f}"
            )

        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # -----------------------------
        # Print gradient norm
        # -----------------------------
        if (epoch + 1) % 10 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.detach().pow(2).sum().item()
            total_norm = total_norm ** 0.5
            print(f"  Gradient norm: {total_norm:.4f}")

    # -----------------------------
    # Restore best model
    # -----------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {history['best_epoch']}")

    return model.eval(), history