from dataclasses import asdict
import numpy as np
import torch
from typing import Any

from app.src.ml.models.ae import TabularAE
from app.src.ml.models.configs import AEConfig, VAEConfig
from app.src.ml.models.helpers import unsupervised_dataloader
from app.src.ml.models.vae import TabularVAE


def train_autoencoder(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    val_cont: np.ndarray | None,
    val_cat: np.ndarray | None,
    config: AEConfig | VAEConfig,
    loss_weights: dict[str, float],
) -> tuple[TabularAE | TabularVAE, dict[str, Any]]:
    """
    Train autoencoder with cont and cat features on split dataset with early stopping OR full dataset with no early stopping.
   
    Returns
    -------
    model : TabularAE or TabularVAE
    history : dict
    """

    device = torch.device(config.device)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = unsupervised_dataloader(
        train_cont, 
        train_cat, 
        config.batch_size, 
        shuffle=True
    ) 

    val_loader = None
    if val_cont is not None:
        val_loader = unsupervised_dataloader(
            val_cont, 
            val_cat, 
            config.batch_size, 
            shuffle=False
        )  

    # -------------------------
    # Build model and set configs
    # -------------------------
    if isinstance(config, AEConfig):
        model = TabularAE(config=config).to(device)
        m1_name, m2_name = "Cont", "Cat"
        m1_key, m2_key = "cont_loss", "cat_loss"
    elif isinstance(config, VAEConfig):
        model = TabularVAE(config=config).to(device)
        m1_name, m2_name = "Recon", "KL"
        m1_key, m2_key = "recon_loss", "kl_loss"
    else: 
        raise ValueError(f"[ERROR] Unsupported model config type; exprected AEConfig or VAEConfig.")

    if isinstance(config, AEConfig) and config.warmup_epochs > 0:
        print(
            f"[INFO] Cat-only warmup enabled for "
            f"{config.warmup_epochs} epochs"
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
    # Set history and param tracking
    # -------------------------
    history = {
        "train_loss": [],
        f"train_{m1_key}": [],
        f"train_{m2_key}": [],
        f"train_recon_cont": [],
        f"train_recon_cat": [],
        "val_loss": [] if val_loader else None,
        f"val_{m1_key}": [] if val_loader else None,
        f"val_{m2_key}": [] if val_loader else None,
        f"val_recon_cont": [],
        f"val_recon_cat": [],
        "warmup": [],
        "learning_rates": [],
        "best_epoch": 0,
        "config": asdict(config),
        "loss_weights": loss_weights,
    }

    best_metric = float("inf")
    best_state = None
    no_improve = 0
    in_warmup = False
    annealing = False
    
    # -------------------------
    # Print config summary
    # -------------------------
    print(f"[INFO] Training autoencoder with:")
    print(f"\tCont features        {config.num_cont}")
    print(f"\tCat features         {len(config.cat_dims)}")
    print(f"\tTransformation Cont  {'RobustScaler'}")
    print(f"\t               Cat   {'Embedding' if config.use_embedding else 'One-Hot encoding'}")
    print(f"\tBatch size           {config.batch_size}")
    print(f"\tHidden dimensions    {config.hidden_dims}")
    print(f"\tLatent dimension     {config.latent_dim}")
    print(f"\tActivation encoder   {config.activation_en}")
    print(f"\tActivation decoder   {config.activation_de}")
    print(f"\tOptimizer            {config.optimizer}")
    print(f"\tLR scheduler         {config.lr_scheduler}")
    print(f"\tNoise injection      {'enabled' if config.allow_noise_injection else 'disabled'}")
    if isinstance(config, VAEConfig):
        print(f"\tBeta annealing       {f'{config.beta_schedule} | {config.beta:.5f}' if config.use_beta_annealing else 'disabled'}")
        print(f"\tKL clipping          {f'enabled | {config.kl_clamp:.5f}' if config.use_kl_clipping else 'disabled'}")
    elif isinstance(config, AEConfig):
        print(f"\tNum of warmup epochs {config.warmup_epochs}")
    print(f"\tLoss weights Cont    {loss_weights['cont_w']:.5f}")
    print(f"\t             Cat     {loss_weights['cat_w']:.5f}")

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(config.num_epochs):
        model.train()    

        # -------------------------
        # Warmup setup
        # -------------------------
        if isinstance(config, AEConfig):
            in_warmup = epoch < config.warmup_epochs
            if in_warmup:
                # freeze cont heads
                for p in model.D.cont_recon_head.parameters():
                    p.requires_grad = False
                # adjust lr to avoid overfitting
                for g in optimizer.param_groups:
                    g["lr"] = config.lr * float((epoch+1) / config.warmup_epochs)   
            else:
                # unfreeze after warmup
                for p in model.D.cont_recon_head.parameters():
                    p.requires_grad = True
                # adjust lr
                for g in optimizer.param_groups:
                    g["lr"] = config.lr
        else:
            if config.use_beta_annealing:
                annealing = True
                model.set_beta_annealing(epoch=epoch)
            if model.current_beta >= config.beta:
                annealing = False

        # -------------------------
        # Setup training
        # -------------------------
        tl, tm1, tm2, tn = 0, 0, 0, 0
        if isinstance(model, TabularVAE):
            tc, tk = 0, 0
        
        for bXc, bXk in train_loader:
            bXc = bXc.to(device)
            bXk = bXk.to(device)

            optimizer.zero_grad(set_to_none=True)

            if isinstance(config, AEConfig):
                cont_recon, cat_logits = model(bXc, bXk)
                cont_loss, cat_loss, total_loss = model.scoring(
                    bXc, 
                    bXk, 
                    cont_recon,
                    cat_logits,
                    loss_weights,
                    in_warmup
                )
                m1_loss, m2_loss = cont_loss, cat_loss
            elif isinstance(config, VAEConfig):
                cont_recon, cat_logits, mu, logvar = model(bXc, bXk)
                cont_loss, cat_loss, recon_loss, kl_loss, total_loss = model.scoring(
                    bXc, 
                    bXk,
                    cont_recon,
                    cat_logits,
                    mu,
                    logvar,
                    loss_weights,
                )
                m1_loss, m2_loss = recon_loss, kl_loss

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
            tm1 += m1_loss.sum().item()
            tm2 += m2_loss.sum().item()
            if isinstance(model, TabularVAE):
                tc += cont_loss.sum().item()
                tk += cat_loss.sum().item()
            tn += bs

        # Calculate epoch averages
        avg_tl = tl / max(tn, 1)
        avg_tm1 = tm1 / max(tn, 1)
        avg_tm2 = tm2 / max(tn, 1)
        if isinstance(model, TabularVAE):
            avg_tc = tc / max(tn, 1)
            avg_tk = tk / max(tn, 1)
        
        history["train_loss"].append(avg_tl)
        history[f"train_{m1_key}"].append(avg_tm1)
        history[f"train_{m2_key}"].append(avg_tm2)
        if isinstance(model, TabularVAE):
            history["train_recon_cont"].append(avg_tc)
            history["train_recon_cat"].append(avg_tk)
        history["warmup"].append(in_warmup),

        # -----------------------------
        # Validation
        # -----------------------------
        if val_loader:
            model.eval()
            vl, vm1, vm2, vn = 0, 0, 0, 0
            if isinstance(model, TabularVAE):
                vc, vk = 0, 0

            with torch.no_grad():
                for bXc, bXk in val_loader:
                    bXc = bXc.to(device, non_blocking=True)
                    bXk = bXk.to(device, non_blocking=True)

                    if isinstance(config, AEConfig):
                        cont_recon, cat_logits = model(bXc, bXk)
                        cont_loss, cat_loss, total_loss = model.scoring(
                            bXc, 
                            bXk,
                            cont_recon,
                            cat_logits,
                            loss_weights,
                        )
                        m1_loss, m2_loss = cont_loss, cat_loss 
                    elif isinstance(config, VAEConfig):
                        cont_recon, cat_logits, mu, logvar = model(bXc, bXk)
                        cont_loss, cat_loss, recon_loss, kl_loss, total_loss = model.scoring(
                            bXc, 
                            bXk,
                            cont_recon,
                            cat_logits,
                            mu,
                            logvar,
                            loss_weights,
                        )
                        m1_loss, m2_loss  = recon_loss, kl_loss

                    # Accumulate
                    bs = bXc.size(0)
                    vl += total_loss.item() * bs
                    vm1 += m1_loss.sum().item()
                    vm2 += m2_loss.sum().item()
                    if isinstance(model, TabularVAE):
                        vc += cont_loss.sum().item()
                        vk += cat_loss.sum().item()
                    vn += bs

            # Calculate validation averages
            avg_vl = vl / max(vn, 1)
            avg_vm1 = vm1 / max(vn, 1)
            avg_vm2 = vm2 / max(vn, 1)
            if isinstance(model, TabularVAE):
                avg_vc = vc / max(vn, 1)
                avg_vk = vk / max(vn, 1)
            
            history["val_loss"].append(avg_vl)
            history[f"val_{m1_key}"].append(avg_vm1)
            history[f"val_{m2_key}"].append(avg_vm2)
            if isinstance(model, TabularVAE):
                history["val_recon_cont"].append(avg_vc)
                history["val_recon_cat"].append(avg_vk)

            if scheduler is not None:
                if config.lr_scheduler == "plateau":
                    scheduler.step(avg_tl)
                elif config.lr_scheduler in ["cosine", "step"]:
                    scheduler.step()
            
            # -----------------------------
            # Early stopping
            # -----------------------------
            if not in_warmup or annealing:
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
            if isinstance(model, TabularAE):
                mode = "WARMUP" if in_warmup else "FULL"
                print(f"Epoch {epoch+1:2d}/{config.num_epochs} - "
                    f"Mode {mode:<3} | "
                    f"TRAIN: {avg_tl:.4f} "
                    f"({m1_name}: {avg_tm1:.4f}, {m2_name}: {avg_tm2:.4f}) |  "
                    f"VAL: {avg_vl:.4f} "
                    f"({m1_name}: {avg_vm1:.4f}, {m2_name}: {avg_vm2:.4f}) |  "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}")
            else:
                eff_kl_train = config.beta*avg_tm2
                eff_kl_val = config.beta*avg_vm2
                print(f"E {epoch+1:1d}/{config.num_epochs} - "
                    f"TRAIN: {avg_tl:.4f} "
                    f"({m1_name}: {avg_tm1:.4f}, β·{m2_name}: {eff_kl_train:.4f}={eff_kl_train*100/avg_tm1:.2f}%, "
                    f"cont: {avg_tc:.3f}, cat: {avg_tk:.3f}) | "
                    f"VAL: {avg_vl:.4f} "
                    f"({m1_name}: {avg_vm1:.4f}, β·{m2_name}: {eff_kl_val:.4f}={eff_kl_val*100/avg_vm1:.2f}%, "
                    f"cont: {avg_tc:.3f}, cat: {avg_tk:.3f}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}, "
                    f"β: {model.current_beta:.4f}")
            
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
            if isinstance(model, TabularAE):
                mode = "WARMUP" if in_warmup else "FULL"
                print(f"Epoch {epoch+1:2d}/{config.num_epochs} - "
                    f"Mode {mode:<3} | "
                    f"TRAIN: {avg_tl:.4f} "
                    f"({m1_name}: {avg_tm1:.4f}, {m2_name}: {avg_tm2:.4f}) |  "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}")
            else:
                eff_kl_train = config.beta*avg_tm2
                print(f"Epoch {epoch+1:1d}/{config.num_epochs} - "
                    f"TRAIN: {avg_tl:.4f} "
                    f"({m1_name}: {avg_tm1:.4f}, β·{m2_name}: {eff_kl_train:.4f}={eff_kl_train*100/avg_tm1:.2f}%, "
                    f"cont: {avg_tc:.3f}, cat: {avg_tk:.3f}) | "
                    f"LR: {optimizer.param_groups[0]['lr']:.5f}, "
                    f"β: {model.current_beta:.4f}")

        history["learning_rates"].append(optimizer.param_groups[0]['lr'])

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
    
    if isinstance(model, TabularVAE):
        del history["warmup"]
    else:
        for key in ["train_recon_cont", "train_recon_cat", "val_recon_cont", "val_recon_cat"]:
            del history[key],

    return model.eval(), history
