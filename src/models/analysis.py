import json
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
)

#########################################
##                CONFIG               ##
#########################################

custom_rc = {
    "figure.titlesize": 22,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "Arial",
    "legend.title_fontsize": 14,
    "legend.fontsize": 12,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}

def apply_custom_theme() -> None:
    """Apply consistent Matplotlib styling."""
    mpl.rcParams.update(custom_rc)


#########################################
##           TRAINING PLOTS            ##
#########################################

def plot_training_curves(
    country: str,
    history: dict,
    out_dir: Path,
    show: bool = False,
):
    """Lineplots showing a) loss curve (train vs val) and b) learning rate schedule."""
    apply_custom_theme()

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    lrs = np.array(history["learning_rates"])
    best_epoch = history.get("best_epoch", None)

    # -------------------------------
    # 1. Loss curves
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch = {best_epoch}")
    ax.set_title(f"{country} — Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_xticks(list(range(1, len(epochs) + 1, 3)))
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fpath = out_dir / f"{country}_loss_curve.png"
    fig.savefig(fpath, dpi=160)
    if show: plt.show()
    plt.close(fig)

    # -------------------------------
    # 2. Learning rate curve
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, label="Learning Rate", linewidth=2)
    ax.set_title(f"{country} — LR Schedule")
    ax.set_xlabel("Epoch")
    ax.set_xticks(list(range(1, len(epochs) + 1, 3)))
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True)
    plt.tight_layout()
    fpath = out_dir / f"{country}_lr_schedule.png"
    fig.savefig(fpath, dpi=160)
    if show: plt.show()
    plt.close(fig)


#########################################
##          VALIDATION PLOTS           ##
#########################################

def plot_error_histogram(
    country: str,
    df: pd.DataFrame,
    out_dir: Path,
    show: bool = False,
):
    """Histogram of log errors with percentile lines."""
    apply_custom_theme()

    errors = df["error"].values
    log_err = np.log10(errors + 1e-8)
    p95 = np.percentile(errors, 95)
    p99 = np.percentile(errors, 99)
    p995 = np.percentile(errors, 99.5)
    med = np.median(errors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(log_err, bins=60, color="steelblue", alpha=0.7)
    # marker lines (in log space for consistency)
    for p, label in [(med, "median"), (p95, "p95"), (p99, "p99"), (p995, "p995")]:
        ax.axvline(np.log10(p + 1e-8), linestyle="--", label=label)
    ax.set_title(f"{country} — Validation Error Distribution (log10 scale)")
    ax.set_xlabel("log10(error)")
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fpath = out_dir / f"{country}_error_hist.png"
    fig.savefig(fpath, dpi=160)
    if show: plt.show()
    plt.close(fig)

def plot_error_timeseries(
    country: str,
    df: pd.DataFrame,
    threshold: Optional[float],
    out_dir: Path,
    show: bool = False,
):
    """Lineplot error over time with optional threshold line."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["ts"], df["error"], linewidth=1, label="Error")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.2f}")
    ax.set_title(f"{country} — Validation Error Time Series")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Reconstruction Error")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fpath = out_dir / f"{country}_error_timeseries.png"
    fig.savefig(fpath, dpi=160)
    if show: plt.show()
    plt.close(fig)

def summarize_validation(
    country: str,
    df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Summary statistics for validation error distribution as json."""
    errors = df["error"].values
    summary = {
        "country": country,
        "count": int(len(errors)),
        "min": float(errors.min()),
        "max": float(errors.max()),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "p995": float(np.percentile(errors, 99.5)),
    }
    path = out_dir / f"{country}_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


#########################################
##             TUNING PLOTS            ##
#########################################

def save_optuna_plots(study, out_dir: Path):
    """Save Optuna-provided visualizations as png if kaleido is installed and/or html file which requires no add package and has nicer formatting."""

    figs = {
        "optimization_history": plot_optimization_history(study),
        "param_importance": plot_param_importances(study),
        "parallel_coordinates": plot_parallel_coordinate(study),
        "slice": plot_slice(study),
        "contour": plot_contour(study),
    }
    for fname, fig in figs.items():
        png_path = out_dir / f"{fname}.png"
        html_path = out_dir / f"{fname}.html"
        try:
            fig.write_html(str(html_path))
            fig.write_image(str(png_path), scale=2)
        except Exception:
            pass

def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path, show: bool):
    """Heatmaps to show correlation between hyperparameters and val loss."""
    apply_custom_theme()

    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Hyperparameter Correlations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show: plt.show()
    plt.close()

def plot_loss_curves_all_trials(study, country: str, history_dir: Path, out_path: Path, show: bool):
    """Lineplot train/val loss curves for each finished trial. Skipped if not saved during tuning."""
    apply_custom_theme()

    plt.figure(figsize=(20, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(study.trials)))
    plotted = False

    for i, t in enumerate(study.trials):
        if t.state.name != "COMPLETE":
            continue
        hist_file = history_dir / f"{country}_trial_{t.number}_history.json"
        if not hist_file.exists():
            continue
        with open(hist_file, "r") as f:
            hist = json.load(f)
        train_loss = hist.get("train_loss")
        val_loss = hist.get("val_loss")
        if train_loss is None or val_loss is None:
            continue
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, val_loss, label=f"trial {t.number}", alpha=0.6, color=colors[i])
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
    plt.savefig(out_path, dpi=160)
    if show: plt.show()
    plt.close()

def plot_best_trial_learning_curve(best_history: dict, out_path: Path, show: bool):
    """Lineplot train/val learning curves of best trial."""
    apply_custom_theme()

    train_loss = best_history["train_loss"]
    val_loss = best_history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = best_history.get("best_epoch")

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Best Trial — Learning Curve")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show: plt.show()
    plt.close()

def plot_3d_scatter(df: pd.DataFrame, out_path: Path, show: bool):
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
    plt.savefig(out_path, dpi=160)
    if show: plt.show()
    plt.close(fig)

def plot_multi_country_overview(best_losses: dict, out_path: Path, show: bool):
    """Barplot comparing best val losses across countries."""
    apply_custom_theme()

    countries = list(best_losses.keys())
    losses = [best_losses[c] for c in countries]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=countries, y=losses)
    plt.title("Best Validation Loss per Country")
    plt.ylabel("Validation Loss")
    plt.xlabel("Country")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    if show: plt.show()
    plt.close()