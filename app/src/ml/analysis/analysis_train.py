from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .common import apply_custom_theme


def plot_training_curves(
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fnames: list[str] = ["loss_curve.png", "lr_schedule.png"],
    show: bool = False,
    MT: bool = False
) -> None:
    """Lineplots showing a) loss curve (train vs val) and b) learning rate schedule."""
    apply_custom_theme()

    train_loss = np.array(history["train_loss"], dtype=float)
    val_loss = np.array(history["val_loss"], dtype=float)
    lrs = np.array(history["learning_rates"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = history.get("best_epoch", None)

    # Ensure lr schedule length matches number of epochs
    if len(lrs) < len(epochs):
        pad = np.full(len(epochs) - len(lrs), lrs[-1] if len(lrs) > 0 else 0.0)
        lrs = np.concatenate([lrs, pad])

    # normalization for shape comparison
    train_norm = train_loss / train_loss[0]
    val_norm   = val_loss / val_loss[0]

    # -------------------------------
    # 1. Loss curves
    # -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1) Raw losses (log-scale)
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train", linewidth=2)
    ax.plot(epochs, val_loss, label="Val", linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")

    ax.set_title(f"{country} — Loss Curve (Raw Loss, Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss" if MT else "Loss")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    # 2) Normalized losses (linear scale)
    ax2 = axes[1]
    ax2.plot(epochs, train_norm, label="Train (norm)", linewidth=2)
    ax2.plot(epochs, val_norm, label="Val (norm)", linewidth=2)
    if best_epoch:
        ax2.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")

    ax2.set_title(f"{country} — Learning Curve (Normalized to check for overfitting)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Normalized Total Loss" if MT else "Normalized Loss")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(folder / fnames[0], dpi=160)
    print(f"[OK] Saved to {fnames[0]}")
    if show: plt.show()
    plt.close(fig)

    # -------------------------------
    # 2. Learning rate curve
    # -------------------------------
    fig2, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(epochs, lrs, linewidth=2)
    ax3.set_title(f"{country} — Learning Rate Schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.grid(True)

    plt.tight_layout()
    fig2.savefig(folder / fnames[1], dpi=160)
    print(f"[OK] Saved to {fnames[1]}")
    if show: plt.show()
    plt.close(fig2)


def plot_detailed_loss_curves(
    ae_type: str,
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fname: str = "detailed_loss_curves.png",
    show: bool = False,
) -> None:
    """Plot separate loss curves for continuous and categorical components."""
    apply_custom_theme()

    cont_loss_name = "cont_loss" if ae_type == "ae" else "recon_loss"
    cat_loss_name = "cat_loss" if ae_type == "ae" else "kl_loss"

    if f"train_{cont_loss_name}" not in history or f"train_{cat_loss_name}" not in history:
        print(f"[INFO] Detailed loss components not available for {country}")
        return
    
    train_cont = np.array(history[f"train_{cont_loss_name}"], dtype=float)
    train_cat = np.array(history[f"train_{cat_loss_name}"], dtype=float)
    val_cont = np.array(history.get(f"val_{cont_loss_name}", []), dtype=float)
    val_cat = np.array(history.get(f"val_{cat_loss_name}", []), dtype=float)
    epochs = np.arange(1, len(train_cont) + 1)

    # normalization for shape comparison
    train_norm = train_cont / train_cont[0]
    val_norm   = val_cont / val_cont[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(epochs, train_norm, label="Train Continuous", linewidth=2, color='blue')
    if len(val_cont) > 0:
        axes[0, 0].plot(epochs, val_norm, label="Val Continuous", linewidth=2, color='cyan')
    axes[0, 0].set_title("Continuous MSE Loss (Normalized to check for overfitting)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, train_cat, label="Train Categorical", linewidth=2, color='red')
    if len(val_cat) > 0:
        axes[0, 1].plot(epochs, val_cat, label="Val Categorical", linewidth=2, color='orange')
    axes[0, 1].set_title("Categorical Cross-Entropy Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Cross-Entropy")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # loss ratio (CE/MSE)
    if len(train_cat) > 0 and len(train_cont) > 0:
        loss_ratio = train_cat / (train_cont + 1e-8)
        axes[1, 0].plot(epochs, loss_ratio, linewidth=2, color='purple')
        axes[1, 0].set_title("Loss Ratio (CE / MSE)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Ratio")
        axes[1, 0].grid(True)
    
    # loss weights
    if "loss_weights" in history:
        weights = history["loss_weights"]
        axes[1, 1].bar(["Continuous", "Categorical"], 
                      [weights.get("cont_weight", 1.0), weights.get("cat_weight", 0.0)],
                      color=['blue', 'red'])
        axes[1, 1].set_title("Loss Weights")
        axes[1, 1].set_ylabel("Weight")
        axes[1, 1].grid(True, axis='y')
    
    plt.suptitle(f"{country} — Detailed Loss Analysis", fontsize=20)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_detailed_mt_loss_curves(
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_detailed_mt_loss_curves.png",
    show: bool = False,
) -> None:
    """Plot separate loss curves for L3 and l7 regression losses and attack classification loss."""
    apply_custom_theme()

    required = ["train_l3", "train_l7", "train_attack"]
    if not all(k in history for k in required):
        print(f"[INFO] Detailed MT loss components not available for {country}")
        return
    
    train_l3 = np.array(history["train_l3"], dtype=float)
    train_l7 = np.array(history["train_l7"], dtype=float)
    train_la = np.array(history["train_attack"], dtype=float)
    
    val_l3 = np.array(history.get("val_l3", []), dtype=float)
    val_l7 = np.array(history.get("val_l7", []), dtype=float)
    val_la = np.array(history.get("val_attack", []), dtype=float)
    
    epochs = np.arange(1, len(train_l3) + 1)

    # normalization for shape comparison
    def safe_norm(x):
        ref = np.median(x[:3]) if len(x) >= 3 else x[0]
        return x / ref if ref > 0 else x

    train_l3_n = safe_norm(train_l3)
    train_l7_n = safe_norm(train_l7)

    val_l3_n = safe_norm(val_l3)
    val_l7_n = safe_norm(val_l7)


    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    axes[0, 0].plot(epochs, train_l3_n, label="Train L3", lw=2, color="blue")
    axes[0, 0].plot(epochs, train_l7_n, label="Train L7", lw=2, color="lightblue")

    if len(val_l3_n):
        axes[0, 0].plot(epochs, val_l3_n, "--", label="Val L3", lw=2, color="green")
    if len(val_l7_n):
        axes[0, 0].plot(epochs, val_l7_n, "--", label="Val L7", lw=2, color="palegreen")

    axes[0, 0].set_title("Regression Losses (Normalized)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Normalized Loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, train_la, label="Train Attack", lw=2, color="red")
    if len(val_la):
        axes[0, 1].plot(epochs, val_la, "--", label="Val Attack", lw=2, color="orange")

    axes[0, 1].set_title("Attack Classification Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # loss ratio
    eps = 1e-8
    axes[1, 0].plot(
        epochs,
        train_la / (train_l3 + eps),
        label="Attack / L3",
        lw=2,
        color="purple"
    )
    axes[1, 0].plot(
        epochs,
        train_la / (train_l7 + eps),
        label="Attack / L7",
        lw=2,
        color="darkkhaki"
    )

    axes[1, 0].set_title("Loss Ratios")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # loss weights
    if "loss_weights" in history:
        w = history["loss_weights"]
        axes[1, 1].bar(
            ["L3", "L7", "Attack"],
            [
                w.get("l3_weight", 1.0),
                w.get("l7_weight", 1.0),
                w.get("attack_weight", 1.0),
            ],
        )
        axes[1, 1].set_title("Loss Weights")
        axes[1, 1].set_ylabel("Weight")
        axes[1, 1].grid(True, axis="y")
    else:
        axes[1, 1].axis("off")
    
    plt.suptitle(f"{country} — Detailed MT Loss Analysis", fontsize=20)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)
