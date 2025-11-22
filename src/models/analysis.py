import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


#########################################
##                CONFIG               ##
#########################################

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

custom_rc = {
    "figure.titlesize": 22,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
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
##               LOAD DATA             ##
#########################################

def load_training_history(country: str, models_dir: Path) -> Dict:
    """Load training history for country and returns train_loss, val_loss, learning_rates, best_epoch."""
    path = Path(models_dir) / f"{country}_training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_validation_errors(country: str, validated_dir: Path) -> pd.DataFrame:
    """Load validation CSV and returns ts, error."""
    path = Path(validated_dir) / f"{country}_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["ts"])


#########################################
##           TRAINING PLOTS            ##
#########################################

def plot_training_curves(
    country: str,
    history: Dict,
    out_dir: Path,
    show: bool = False,
):
    """Lineplots showing a) loss curve (train vs val) and b) learning rate schedule."""
    apply_custom_theme()

    out_dir = _ensure_dir(Path(out_dir))

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
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")
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

    out_dir = _ensure_dir(Path(out_dir))
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

    out_dir = _ensure_dir(Path(out_dir))

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


#########################################
##               SUMMARY               ##
#########################################

def summarize_validation(
    country: str,
    df: pd.DataFrame,
    out_dir: Path,
) -> Dict:
    """Cummary statistics for validation error distribution as json."""
    out_dir = _ensure_dir(Path(out_dir))

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
##              RUN ANALYSIS           ##
#########################################

def analyze_country(
    country: str,
    trained_dir: Path,
    validated_dir: Path,
    show_plots: bool = False,
):
    """
    Runs full analysis pipeline for a single country.
    Generates:
        - loss curves
        - lr schedule
        - error histogram
        - error timeseries
        - summary
    """

    print(f"[INFO] Analyzing {country}")

    trained_plot_dir = Path(trained_dir) / "plots"
    validated_plot_dir = Path(validated_dir) / "plots"

    # ---- Load data ----
    history = load_training_history(country, trained_dir)
    val_df = load_validation_errors(country, validated_dir)

    # ---- Compute threshold from validation distribution (optional) ----
    threshold = np.percentile(val_df["error"], 99)

    # ---- Plots ----
    plot_training_curves(
        country,
        history,
        trained_plot_dir,
        show=show_plots,
    )

    plot_error_histogram(
        country,
        val_df,
        validated_plot_dir,
        show=show_plots,
    )

    plot_error_timeseries(
        country,
        val_df,
        threshold,
        validated_plot_dir,
        show=show_plots,
    )

    # ---- Summary ----
    summary = summarize_validation(
        country,
        val_df,
        validated_plot_dir,
    )

    print(f"[OK] Analysis for {country} completed!")
    return summary
