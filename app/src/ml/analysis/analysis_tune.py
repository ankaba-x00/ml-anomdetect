import json, optuna
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
)

from .common import apply_custom_theme


def save_optuna_plots(
    study: optuna.Study, 
    folder: Path, 
    html_out: bool = True, 
    png_out: bool = False
) -> None:
    """
    Save Optuna-provided visualizations as png if kaleido is installed and/or 
    html file which requires no add package and has nicer formatting.
    """

    figs = {
        "optimization_history": plot_optimization_history(study),
        "param_importance": plot_param_importances(study),
        "parallel_coordinates": plot_parallel_coordinate(study),
        "slice": plot_slice(study),
        "contour": plot_contour(study),
    }
    for fname, fig in figs.items():
        try:
            if png_out: 
                png_fname = folder / f"{fname}.png"
                fig.write_image(str(png_fname), scale=2)
            if html_out:
                html_fname = folder / f"{fname}.html"
                fig.write_html(str(html_fname))
        except Exception:
            pass


def plot_correlation_heatmap(
    df: pd.DataFrame, 
    folder: Path = Path.cwd(), 
    fname: str = "plot_correlation_heatmap.png",
    show: bool = False
) -> None:
    """Heatmaps to show correlation between hyperparameters and val loss."""
    apply_custom_theme()

    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Hyperparameter Correlations")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


def plot_loss_curves_all_trials(
    study: optuna.Study, 
    country: str, 
    history_dir: Path, 
    folder: Path = Path.cwd(),
    fname: str = "plot_loss_curves_all_trials.png", 
    show: bool = False
) -> None:
    """Lineplot train/val loss curves for each finished trial. Skipped if not saved during tuning."""
    apply_custom_theme()

    plt.figure(figsize=(20, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(study.trials)))
    plotted = False

    for idx, trial in enumerate(study.trials):
        if trial.state.name != "COMPLETE":
            continue
        hist_file = history_dir / f"{country}_trial_{trial.number:04d}_history.json"
        if not hist_file.exists():
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)
        train_loss = hist.get("train_loss")
        val_loss = hist.get("val_loss")
        if train_loss is None or val_loss is None:
            continue
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, val_loss, label=f"trial {trial.number}", alpha=0.6, color=colors[idx])
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(f"{country} — Validation Loss per Trial")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


def plot_best_trial_learning_curve(
    best_history: dict, 
    folder: Path = Path.cwd(),
    fname: str = "plot_best_trial_learning_curve.png",
    show: bool = False
) -> None:
    """Lineplot train/val learning curves of best trial."""
    apply_custom_theme()

    train_loss = np.array(best_history["train_loss"], dtype=float)
    val_loss   = np.array(best_history["val_loss"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = best_history.get("best_epoch")

    # normalization for shape comparison
    train_norm = train_loss / train_loss[0]
    val_norm   = val_loss / val_loss[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    
    # 1) Raw losses (log-scale)
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)

    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Learning Curve (Raw Loss, Log Scale)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    # 2) Normalized losses (linear scale)
    ax2 = axes[1]
    ax2.plot(epochs, train_norm, label="Train (norm)", linewidth=2)
    ax2.plot(epochs, val_norm, label="Val (norm)", linewidth=2)

    if best_epoch is not None:
        ax2.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Normalized Loss")
    ax2.set_title("Learning Curve (Normalized to check for overfitting)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


def plot_3d_scatter(
    df: pd.DataFrame, 
    folder: Path = Path.cwd(), 
    fname: str = "plot_3d_scatter.png",
    show: bool = False
) -> None:
    """Scatter plot 3D (dropout, lr, val_loss) of hyperparameter landscape with annotated marking of best trial in red."""
    apply_custom_theme()

    required_cols = {"dropout", "lr", "value"}
    if not required_cols.issubset(df.columns):
        print("[WARN] Missing columns for 3D plot, skipping...")
        return
    best_idx = df["value"].idxmin()
    best_row = df.loc[best_idx]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=df["value"].min(), vmax=df["value"].max())
    # Log-transform LR for visualization
    lr_log = np.log10(df["lr"])
    sc = ax.scatter(
        df["dropout"], 
        lr_log, 
        df["value"], 
        c=df["value"], 
        cmap=cmap, 
        norm=norm, 
        s=60, 
        alpha=0.85, 
        edgecolor="k"
    )
    # annotation best trial
    ax.scatter(
        best_row["dropout"], 
        np.log10(best_row["lr"]), 
        best_row["value"], 
        color="red", 
        s=200, 
        marker="X", 
        edgecolor="black", 
        label=f"Best Trial (val_loss={best_row['value']:.4f})"
    )
    cbar = fig.colorbar(
        sc, 
        ax=ax, 
        location="left", 
        fraction=0.015, 
        pad=0.05
    )
    cbar.set_label("Val Loss", fontsize=12)
    ax.set_xlabel("Dropout", labelpad=12)
    ax.set_ylabel("log10(LR)", labelpad=12)
    ax.set_zlabel("Val Loss", labelpad=12)
    ax.set_title("3D Hyperparameter Landscape")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_loss_component_analysis(
    ae_type: str, 
    study: optuna.Study,
    country: str, 
    history_dir: Path,
    folder: Path = Path.cwd(),
    fname: str = "loss_component_analysis.png",
    show: bool = False
) -> None:
    """Scatter plots showing how continuous vs categorical losses contribute to total loss."""
    apply_custom_theme()
    
    cont_loss_name = "cont_loss" if ae_type == "ae" else "recon_loss"
    cat_loss_name = "cat_loss" if ae_type == "ae" else "kl_loss"
    
    cont_losses = []
    cat_losses = []
    total_losses = []
    trial_numbers = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        hist_file = history_dir / f"{country}_trial_{trial.number:04d}_history.json"
        if not hist_file.exists():
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)
        if f"val_{cont_loss_name}" in hist and f"val_{cat_loss_name}" in hist:
            cont_losses.append(hist[f"val_{cont_loss_name}"][-1])
            cat_losses.append(hist[f"val_{cat_loss_name}"][-1])
            total_losses.append(trial.value)
            trial_numbers.append(trial.number)
    if len(cont_losses) < 3:
        print(f"[INFO] Not enough loss component data for {country}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].scatter(cont_losses, total_losses, alpha=0.7)
    axes[0, 0].set_xlabel("Continuous Loss (MSE)")
    axes[0, 0].set_ylabel("Total Validation Loss")
    axes[0, 0].set_title("Continuous Loss Contribution")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(cat_losses, total_losses, alpha=0.7)
    axes[0, 1].set_xlabel("Categorical Loss (CE)")
    axes[0, 1].set_ylabel("Total Validation Loss")
    axes[0, 1].set_title("Categorical Loss Contribution")
    axes[0, 1].grid(True, alpha=0.3)
    
    ratios = [c/(m+1e-8) for m, c in zip(cont_losses, cat_losses)]
    axes[1, 0].scatter(ratios, total_losses, alpha=0.7)
    axes[1, 0].axvline(1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Loss Ratio (CE/MSE)")
    axes[1, 0].set_ylabel("Total Validation Loss")
    axes[1, 0].set_title("Loss Ratio vs Performance")
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    df_components = pd.DataFrame({
        'trial': trial_numbers,
        'cont_loss': cont_losses,
        'cat_loss': cat_losses,
        'total_loss': total_losses,
        'ratio': ratios
    })
    best_idx = df_components['total_loss'].idxmin()
    
    axes[1, 1].plot(['cont_loss', 'cat_loss', 'total_loss'], 
                   df_components.loc[best_idx, ['cont_loss', 'cat_loss', 'total_loss']], 
                   'ro-', label='Best Trial', linewidth=3)
    
    for idx, row in df_components.iterrows():
        if idx != best_idx:
            axes[1, 1].plot(['cont_loss', 'cat_loss', 'total_loss'], 
                           row[['cont_loss', 'cat_loss', 'total_loss']], 
                           'b-', alpha=0.2)
    
    axes[1, 1].set_title("Loss Component Comparison")
    axes[1, 1].set_ylabel("Loss Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"{country} — Loss Component Analysis")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_multi_loss_overview(
    best_losses: dict, 
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_loss_overview.png",
    show: bool = False
) -> None:
    """Barplot comparing best val losses across countries."""
    apply_custom_theme()

    if not best_losses:
        print("[INFO] No losses found to plot")
        return

    countries = list(best_losses.keys())
    losses = [best_losses[c] for c in countries]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].bar(countries, losses)
    axes[0].set_title("Best Validation Loss per Country (Linear)")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_xlabel("Country")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, axis="y", alpha=0.4)
    
    axes[1].bar(countries, losses)
    axes[1].set_title("Best Validation Loss per Country (Log Scale)")
    axes[1].set_ylabel("Validation Loss (log)")
    axes[1].set_xlabel("Country")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", alpha=0.4)
    
    plt.suptitle(f"Loss Weight Comparison Across Countries (n={len(countries)})")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_multi_weights_overview(
    best_weights: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_weights_overview.png",
    show: bool = False,
) -> None:
    """Bar and scatter plots comparing loss weights and their ratio across countries."""
    apply_custom_theme()
    
    if not best_weights:
        print("[INFO] No loss weights found to plot")
        return
    
    df = pd.DataFrame.from_dict(best_weights, orient='index')
    df.index.name = "country"
    df = df.reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  

    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width/2, df["cont_weight"], width, label="Continuous", color='blue')
    axes[0].bar(x + width/2, df["cat_weight"], width, label="Categorical", color='red')
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Loss Weights by Country")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["country"], rotation=45)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].scatter(df["cont_weight"], df["cat_weight"], s=100, alpha=0.7)
    for i, row in df.iterrows():
        axes[1].annotate(row["country"], (row["cont_weight"], row["cat_weight"]), 
                        fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    axes[1].axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Equal weights')
    axes[1].set_xlabel("Continuous Weight")
    axes[1].set_ylabel("Categorical Weight")
    axes[1].set_title("Weight Scatter Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(df["country"], df["ratio"], color='purple')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1')
    axes[2].set_xlabel("Country")
    axes[2].set_ylabel("Ratio (cat/cont)")
    axes[2].set_title("Weight Ratio by Country")
    #axes[2].set_xticklabels(df["country"], rotation=45)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend()
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f"Loss Weight Comparison Across Countries (n={len(df)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160, bbox_inches='tight')
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_multi_weight_loss_correlation(
    weights_data: dict,
    losses_data: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_weight_loss_correlation.png",
    show: bool = False,
) -> None:
    """Scatter plots showing relationship between loss weights and validation performance."""
    apply_custom_theme()

    merged = {}
    for country in set(weights_data.keys()) & set(losses_data.keys()):
        merged[country] = {
            "cont_weight": weights_data[country]["cont_weight"],
            "cat_weight": weights_data[country]["cat_weight"],
            "ratio": weights_data[country]["ratio"],
            "loss": losses_data[country],
        }

    df = pd.DataFrame.from_dict(merged, orient='index')
    df.index.name = "country"
    df = df.reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].scatter(df["cont_weight"], df["loss"], s=100, alpha=0.7)
    for _, row in df.iterrows():
        axes[0, 0].annotate(row["country"], (row["cont_weight"], row["loss"]), 
                          fontsize=9, alpha=0.8)
    axes[0, 0].set_xlabel("Continuous Weight")
    axes[0, 0].set_ylabel("Validation Loss")
    axes[0, 0].set_title("Continuous Weight vs Performance")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(df["cat_weight"], df["loss"], s=100, alpha=0.7)
    for _, row in df.iterrows():
        axes[0, 1].annotate(row["country"], (row["cat_weight"], row["loss"]), 
                          fontsize=9, alpha=0.8)
    axes[0, 1].set_xlabel("Categorical Weight")
    axes[0, 1].set_ylabel("Validation Loss")
    axes[0, 1].set_title("Categorical Weight vs Performance")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(df["ratio"], df["loss"], s=100, alpha=0.7)
    for _, row in df.iterrows():
        axes[1, 0].annotate(row["country"], (row["ratio"], row["loss"]), 
                          fontsize=9, alpha=0.8)
    axes[1, 0].axvline(1, color='gray', linestyle='--', alpha=0.5, label='Equal weights')
    axes[1, 0].set_xlabel("Weight Ratio (cat/cont)")
    axes[1, 0].set_ylabel("Validation Loss")
    axes[1, 0].set_title("Weight Ratio vs Performance")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    scatter = axes[1, 1].scatter(df["cont_weight"], df["cat_weight"], 
                                 c=df["loss"], s=100, alpha=0.7, cmap='viridis')
    for _, row in df.iterrows():
        axes[1, 1].annotate(row["country"], (row["cont_weight"], row["cat_weight"]), 
                          fontsize=9, alpha=0.8)
    axes[1, 1].axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Equal weights')
    axes[1, 1].set_xlabel("Continuous Weight")
    axes[1, 1].set_ylabel("Categorical Weight")
    axes[1, 1].set_title("Weight Space (color = loss)")
    plt.colorbar(scatter, ax=axes[1, 1], label='Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Loss Weight vs Performance Correlation Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_mt_loss_component_analysis(
    study: optuna.Study,
    country: str, 
    history_dir: Path,
    folder: Path = Path.cwd(),
    fname: str = "plot_mt_loss_component_analysis.png",
    show: bool = False
) -> None:
    """Scatter plots showing how MT loss components contribute to total loss."""
    apply_custom_theme()
    
    reg_losses = []
    attack_losses = []
    total_losses = []
    trial_numbers = []

    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        hist_file = history_dir / f"{country}_trial_{trial.number:04d}_history.json"
        if not hist_file.exists():
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)
        required = ["val_l3", "val_l7", "val_attack"]
        if not all(k in hist for k in required):
            continue
       
        l3 = hist["val_l3"][-1]
        l7 = hist["val_l7"][-1]
        attack = hist["val_attack"][-1]

        reg_losses.append(l3 + l7)
        attack_losses.append(attack)
        total_losses.append(trial.value)
        trial_numbers.append(trial.number)

    if len(reg_losses) < 3:
        print(f"[INFO] Not enough loss component data for {country}")
        return
    
    ratios = [a / (r + 1e-8) for r, a in zip(reg_losses, attack_losses)]
    best_idx = int(np.argmin(total_losses))

    df = pd.DataFrame({
        "trial": trial_numbers,
        "reg_loss": reg_losses,
        "attack_loss": attack_losses,
        "total_loss": total_losses,
        "ratio": ratios,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Regression vs total
    axes[0, 0].scatter(reg_losses, total_losses, alpha=0.7)
    axes[0, 0].set_xlabel("Regression Loss (L3 + L7)")
    axes[0, 0].set_ylabel("Total Validation Loss")
    axes[0, 0].set_title("Regression Loss Contribution")
    axes[0, 0].grid(True, alpha=0.3)

    # Attack vs total
    axes[0, 1].scatter(attack_losses, total_losses, alpha=0.7)
    axes[0, 1].set_xlabel("Attack Classification Loss")
    axes[0, 1].set_ylabel("Total Validation Loss")
    axes[0, 1].set_title("Attack Loss Contribution")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ratio plot
    axes[1, 0].scatter(ratios, total_losses, alpha=0.7)
    axes[1, 0].scatter(
        ratios[best_idx],
        total_losses[best_idx],
        s=200,
        marker="X",
        color="red",
        label="Best Trial"
    )
    axes[1, 0].legend()
    axes[1, 0].axvline(1, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Loss Ratio (Attack / Regression)")
    axes[1, 0].set_ylabel("Total Validation Loss")
    axes[1, 0].set_title("Loss Ratio vs Performance")
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component comparison
    axes[1, 1].plot(
        ["reg_loss", "attack_loss", "total_loss"],
        df.loc[best_idx, ["reg_loss", "attack_loss", "total_loss"]],
        "ro-",
        linewidth=3,
        label="Best Trial",
    )

    for idx, row in df.iterrows():
        if idx != best_idx:
            axes[1, 1].plot(
                ["reg_loss", "attack_loss", "total_loss"],
                row[["reg_loss", "attack_loss", "total_loss"]],
                "b-",
                alpha=0.2,
            )

    axes[1, 1].set_title("Loss Component Comparison")
    axes[1, 1].set_ylabel("Loss Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"{country} — MT Loss Component Analysis")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_attack_class_weights(
    country: str,
    attack_class_weights: np.ndarray,
    class_names: list[str],
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_class_weights.png",
    show: bool = False,
):
    """Bar plot of attack class weights for given country."""
    apply_custom_theme()

    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(attack_class_weights))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(class_names, attack_class_weights)
    ax.set_title(f"{country} — Attack Class Weights")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Attack Class")
    ax.tick_params(axis="x", rotation=45, labelsize=11)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_attack_class_balance(
    country: str,
    attack_class_weights: np.ndarray,
    attack_class_counts: np.ndarray,
    class_names: list[str],
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_class_balance.png",
    show: bool = False,
):
    """Bar plots of class frequencies and weights for given country."""
    apply_custom_theme()

    classes = class_names #[f"class_{i}" for i in range(len(attack_class_weights))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Frequencies
    axes[0].bar(classes, attack_class_counts)
    axes[0].set_title("Attack Class Frequencies (Train)")
    axes[0].set_ylabel("Samples")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=45, labelsize=11)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Weights
    axes[1].bar(classes, attack_class_weights)
    axes[1].set_title("Attack Class Weights")
    axes[1].set_ylabel("Weight")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=45, labelsize=11)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"{country} — Attack Class Balance")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_multi_mt_weights_overview(
    best_weights: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_mt_weights_overview.png",
    show: bool = False,
) -> None:
    """Bar and scatter plots comparing loss weights (regression vs classification) and their ratio across countries."""
    apply_custom_theme()
    
    if not best_weights:
        print("[INFO] No loss weights found to plot")
        return
    
    df = pd.DataFrame.from_dict(best_weights, orient='index')
    df.index.name = "country"
    df = df.reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  

    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width/2, df["reg"], width, label="regression (l3 + l7)", color='blue')
    axes[0].bar(x + width/2, df["class"], width, label="classification", color='red')
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Loss Weights by Country")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["country"], rotation=45)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].scatter(
        df["reg"], 
        df["class"], 
        s=100, 
        alpha=0.7
    )
    for _, row in df.iterrows():
        axes[1].annotate(
            row["country"], 
            (row["reg"], row["class"]), 
            fontsize=9, 
            alpha=0.8, 
            xytext=(5, 5), 
            textcoords='offset points'
        )
    axes[1].axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='Equal weights')
    axes[1].set_xlabel("regression (l3 + l7)")
    axes[1].set_ylabel("classification")
    axes[1].set_title("Weight Balance (Regression vs Attack)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(df["country"], df["ratio"], color='purple')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1')
    axes[2].set_xlabel("Country")
    axes[2].set_ylabel("Classification/Regression Ratio")
    axes[2].set_title("Weight Ratio by Country")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].legend()
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f"MT Loss Weight Comparison Across Countries (n={len(df)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160, bbox_inches='tight')
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_multi_mt_weight_loss_correlation(
    weights_data: dict,
    losses_data: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_mt_weight_loss_correlation.png",
    show: bool = False,
) -> None:
    """Scatter plots showing relationship between MT loss weights and validation performance."""
    apply_custom_theme()

    merged = {}
    for country in set(weights_data.keys()) & set(losses_data.keys()):
        merged[country] = {
            "reg_weight": weights_data[country]["reg"],
            "class_weight": weights_data[country]["class"],
            "ratio": weights_data[country]["ratio"],
            "loss": losses_data[country],
        }
    
    if not merged:
        print("[INFO] No overlapping weight/loss data to plot")
        return

    df = pd.DataFrame.from_dict(merged, orient='index')
    df.index.name = "country"
    df = df.reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # regression weight vs loss
    axes[0, 0].scatter(df["reg_weight"], df["loss"], s=100, alpha=0.75)
    for _, r in df.iterrows():
        axes[0, 0].annotate(r["country"], (r["reg_weight"], r["loss"]), fontsize=9)
    axes[0, 0].set_xlabel("Regression Weight (L3 + L7)")
    axes[0, 0].set_ylabel("Validation Loss")
    axes[0, 0].set_title("Regression Weight vs Performance")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.3)
    
    # classification weight vs loss
    axes[0, 1].scatter(df["class_weight"], df["loss"], s=100, alpha=0.75, color="darkred")
    for _, r in df.iterrows():
        axes[0, 1].annotate(r["country"], (r["class_weight"], r["loss"]), fontsize=9)
    axes[0, 1].set_xlabel("Classification Weight")
    axes[0, 1].set_ylabel("Validation Loss")
    axes[0, 1].set_title("Classificaion Weight vs Performance")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.3)

    # ratio vs loss
    axes[1, 0].scatter(df["ratio"], df["loss"], s=100, alpha=0.75, color="purple")
    for _, r in df.iterrows():
        axes[1, 0].annotate(r["country"], (r["ratio"], r["loss"]), fontsize=9)
    axes[1, 0].axvline(1.0, linestyle="--", color="gray", alpha=0.5, label="Balanced")
    axes[1, 0].set_xlabel("Classification / Regression Weight Ratio")
    axes[1, 0].set_ylabel("Validation Loss")
    axes[1, 0].set_title("Loss Weight Ratio vs Performance")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # weight space colored by loss
    scatter = axes[1, 1].scatter(
        df["reg_weight"],
        df["class_weight"],
        c=df["loss"],
        cmap="viridis",
        s=120,
        alpha=0.8,
    )
    for _, r in df.iterrows():
        axes[1, 1].annotate(
            r["country"],
            (r["reg_weight"], r["class_weight"]),
            fontsize=9,
        )

    axes[1, 1].axline((0, 0), slope=1, linestyle="--", color="gray", alpha=0.5)
    axes[1, 1].set_xlabel("Regression Weight (L3 + L7)")
    axes[1, 1].set_ylabel("Classification Weight")
    axes[1, 1].set_title("Weight Space (color = validation loss)")
    plt.colorbar(scatter, ax=axes[1, 1], label="Validation Loss")
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("MT Loss Weight vs Performance Correlation Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_multi_country_attack_weights(
    weights_by_country: dict[str, list[float]],
    class_names: list[str],
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_country_attack_weights.png",
    show: bool = False,
):
    """Heatmap of attack class weights across countries."""
    apply_custom_theme()

    df = pd.DataFrame.from_dict(
        weights_by_country,
        orient="index"
    )
    if df.shape[1] != len(class_names):
        raise ValueError("[Error] len class_weights do not match len class_names")
    df.columns = class_names #[f"{i}" for i in range(df.shape[1])]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(df))))
    sns.heatmap(
        np.log10(df + 1e-6),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax
    )
    ax.set_title("Attack Class Weights (log10 scale)")
    ax.set_xlabel("Attack Class")
    ax.set_ylabel("Country")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)
