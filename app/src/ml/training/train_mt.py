from dataclasses import asdict
from pathlib import Path
from typing import Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from app.src.ml.models.mte import MTEConfig, TrafficAttackPredictor


#########################################
##           TRAINING HELPERS          ##
#########################################

def _make_supervised_dataloader(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    y_l3: np.ndarray,
    y_l7: np.ndarray,
    y_attack: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    Xc = torch.from_numpy(X_cont.astype(np.float32))
    Xk = torch.from_numpy(X_cat.astype(np.int64))
    y3 = torch.from_numpy(y_l3.astype(np.float32))
    y7 = torch.from_numpy(y_l7.astype(np.float32))
    ya = torch.from_numpy(y_attack.astype(np.int64))

    ds = TensorDataset(Xc, Xk, y3, y7, ya)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


# TODO: max_ratio needs benchmarking!
def compute_attack_class_weights(
    y: np.ndarray,
    n_classes: int,
    max_ratio: float = 20.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for CrossEntropyLoss.
    weight_c = total_samples / (n_classes * count_c)
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, eps)

    weights = counts.sum() / (n_classes * counts)
    # clamps extreme values
    weights = np.clip(weights, 1.0 / max_ratio, max_ratio)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    target_logp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    focal_factor = (1.0 - target_p) ** gamma
    loss = -focal_factor * target_logp

    if weight is not None:
        class_weight = weight.gather(0, targets)
        loss = loss * class_weight

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def train_multitask_model(
    train_cont: np.ndarray,
    train_cat: np.ndarray,
    train_l3: np.ndarray,
    train_l7: np.ndarray,
    train_attack: np.ndarray,
    val_cont: Optional[np.ndarray],
    val_cat: Optional[np.ndarray],
    val_l3: Optional[np.ndarray],
    val_l7: Optional[np.ndarray],
    val_attack: Optional[np.ndarray],
    config: MTEConfig,
    loss_weights: dict[str, float],
) -> tuple[TrafficAttackPredictor, dict[str, Any]]:
    """
    Train multi-task traffic predictor on split dataset with early stopping.
    
    Returns
    -------
    model : TrafficAttackPredictor
    history : dict
    """

    device = torch.device(config.device)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = _make_supervised_dataloader(
        train_cont, 
        train_cat,
        train_l3, 
        train_l7, 
        train_attack,
        config.batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_cont is not None:
        val_loader = _make_supervised_dataloader(
            val_cont, 
            val_cat,
            val_l3, 
            val_l7, 
            val_attack,
            config.batch_size,
            shuffle=False,
        )

    # -------------------------
    # Class weights
    # -------------------------
    attack_class_weights = compute_attack_class_weights(
        train_attack,
        config.n_attack_types,
    )

    if config.warmup_epochs > 0:
        print(
            f"[INFO] Attack-only warmup enabled for "
            f"{config.warmup_epochs} epochs"
        )

    # -------------------------
    # Build model
    # -------------------------
    model = TrafficAttackPredictor(config, attack_class_weights).to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # -------------------------
    # LR Scheduler
    # -------------------------
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    base_lr = config.lr
    warmup_lr = config.lr * 0.3

    # -------------------------
    # History
    # -------------------------
    history = {
        "train_loss": [],
        "train_l3": [],
        "train_l7": [],
        "train_attack": [],
        "val_loss": [] if val_loader else None,
        "val_l3": [] if val_loader else None,
        "val_l7": [] if val_loader else None,
        "val_attack": [] if val_loader else None,
        "learning_rates": [],
        "best_epoch": 0,
        "config": asdict(config),
        "loss_weights": loss_weights,
        "attack_class_weights": []
    }

    best_metric = float("inf")
    best_state = None
    no_improve = 0
    reset_after_warmup = False

    lambda_l3 = loss_weights.get("l3", 1.0)
    lambda_l7 = loss_weights.get("l7", 1.0)
    lambda_att = loss_weights.get("attack", 3.0)

    print("Training multi-task model with:")
    print(f"  L3 weight: {lambda_l3}")
    print(f"  L7 weight: {lambda_l7}")
    print(f"  Attack weight: {lambda_att}")
    print(f"  Device: {device}")

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(config.num_epochs):
        model.train()

        # -------------------------
        # Attack-only warmup logic
        # -------------------------
        in_warmup = epoch < config.warmup_epochs

        if in_warmup:
            # freeze regression heads
            for p in model.l3_head.parameters():
                p.requires_grad = False
            for p in model.l7_head.parameters():
                p.requires_grad = False
            # adjust lr to avoid overfitting
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr
        else:
            # unfreeze after warmup
            for p in model.l3_head.parameters():
                p.requires_grad = True
            for p in model.l7_head.parameters():
                p.requires_grad = True
            # use base lr
            for g in optimizer.param_groups:
                g["lr"] = base_lr

        tl, tl3, tl7, tatt = 0.0, 0.0, 0.0, 0.0
        n = 0

        for Xc, Xk, y3, y7, ya in train_loader:
            Xc, Xk = Xc.to(device), Xk.to(device)
            y3, y7, ya = y3.to(device), y7.to(device), ya.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(Xc, Xk)

            loss_l3 = F.mse_loss(out["l3"], y3)
            loss_l7 = F.mse_loss(out["l7"], y7)
            if config.use_focal_loss and not in_warmup:
                loss_att = focal_loss(
                out["attack_logits"],
                ya,
                config.focal_gamma,
                model.attack_class_weights,
            )
            else:
                loss_att = F.cross_entropy(
                    out["attack_logits"], 
                    ya,
                    model.attack_class_weights
                )

            if in_warmup:
                total_loss = lambda_att * loss_att
            else:
                total_loss = (
                    lambda_l3 * loss_l3 +
                    lambda_l7 * loss_l7 +
                    lambda_att * loss_att
                )

            total_loss.backward()

            if config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.gradient_clip
                )

            optimizer.step()

            bs = Xc.size(0)
            tl += total_loss.item() * bs
            tl3 += loss_l3.item() * bs
            tl7 += loss_l7.item() * bs
            tatt += loss_att.item() * bs
            n += bs

        history["train_loss"].append(tl / n)
        history["train_l3"].append(tl3 / n)
        history["train_l7"].append(tl7 / n)
        history["train_attack"].append(tatt / n)
        history["attack_class_weights"] = (
            attack_class_weights.cpu().tolist()
            if attack_class_weights is not None
            else None
        )

        # -------------------------
        # Validation
        # -------------------------
        if val_loader:
            model.eval()
            vl, vl3, vl7, vatt = 0.0, 0.0, 0.0, 0.0
            vn = 0

            with torch.no_grad():
                for Xc, Xk, y3, y7, ya in val_loader:
                    Xc, Xk = Xc.to(device), Xk.to(device)
                    y3, y7, ya = y3.to(device), y7.to(device), ya.to(device)

                    out = model(Xc, Xk)

                    loss_l3 = F.mse_loss(out["l3"], y3)
                    loss_l7 = F.mse_loss(out["l7"], y7)
                    if config.use_focal_loss  and not in_warmup:
                        loss_att = focal_loss(
                        out["attack_logits"],
                        ya,
                        config.focal_gamma,
                        model.attack_class_weights,
                    )
                    else:
                        loss_att = F.cross_entropy(
                            out["attack_logits"], 
                            ya,
                            model.attack_class_weights
                        )

                    total_loss = (
                        lambda_l3 * loss_l3 +
                        lambda_l7 * loss_l7 +
                        lambda_att * loss_att
                    )

                    bs = Xc.size(0)
                    vl += total_loss.item() * bs
                    vl3 += loss_l3.item() * bs
                    vl7 += loss_l7.item() * bs
                    vatt += loss_att.item() * bs
                    vn += bs

            avg_val = vl / vn
            if in_warmup:
                history["val_loss"].append(None)
                history["val_l3"].append(None)
                history["val_l7"].append(None)
                history["val_attack"].append(None)
                continue
            else:
                history["val_loss"].append(avg_val)
                history["val_l3"].append(vl3 / vn)
                history["val_l7"].append(vl7 / vn)
                history["val_attack"].append(vatt / vn)

            if not in_warmup:
                scheduler.step(avg_val)

            if epoch == config.warmup_epochs and not reset_after_warmup:
                best_metric = float("inf")
                best_state = None
                no_improve = 0
                reset_after_warmup = True
                print(f"[INFO] Warmup finished at epoch {epoch}. Resetting early stopping baseline.")

            # -----------------------------
            # Early stopping
            # -----------------------------
            if epoch >= config.warmup_epochs:
                if avg_val < best_metric - 1e-9:
                    best_metric = avg_val
                    best_state = model.state_dict()
                    history["best_epoch"] = epoch + 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= config.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            mode = "ATTACK-WARMUP" if in_warmup else "MULTI-TASK"
            print(
                f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                f"Mode {mode:<12} | "
                f"Train {history['train_loss'][-1]:.4f} | "
                f"Val {avg_val:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

    # -----------------------------
    # Restore best model
    # -----------------------------
    if best_state:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {history['best_epoch']}")

    return model.eval(), history


#########################################
##         SAVE / LOAD HELPERS         ##
#########################################

def save_multitask_model(
    model: TrafficAttackPredictor,
    config: MTEConfig,
    cat_dims: dict,
    num_cont: int,
    path: Path,
    additional_info: Optional[dict] = None
) -> None:
    """Save model weights + config to a single .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "cat_dims": cat_dims,
        "num_cont": num_cont,
        "additional_info": additional_info or {},
    }
    torch.save(payload, path)

    print(f"[OK] Saved multi-task predictor to {path}")


def load_multitask_model(
    path: Path,
    device: Optional[str] = "cpu"
) -> tuple[TrafficAttackPredictor, MTEConfig, int, dict]:
    """Load model + config from a .pt file."""
    payload = torch.load(path, map_location=device)

    num_cont = payload["num_cont"]
    cat_dims = payload["cat_dims"]

    attack_class_weights = None
    if "additional_info" in payload:
        weights = payload["additional_info"].get("attack_class_weights")
        if weights is not None:
            attack_class_weights = torch.tensor(weights, dtype=torch.float32)
    cfg = MTEConfig(**payload["config"])
    model = TrafficAttackPredictor(cfg, attack_class_weights)

    model.load_state_dict(payload["state_dict"])
    target_device = torch.device(cfg.device)
    model = model.to(target_device)
    
    print(f"[INFO] Loaded multi-task predictor from {path}")
    print(f"[INFO] Model moved to device: {target_device}")
    
    return model.eval(), cfg, num_cont, cat_dims