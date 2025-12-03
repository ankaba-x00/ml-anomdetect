import json
from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from app.src.models.autoencoder import TabularAutoencoder
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    sns.set_style("whitegrid")


#########################################
##           TRAINING PLOTS            ##
#########################################

def plot_training_curves(
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fnames: list[str] = ["loss_curve.png", "lr_schedule.png"],
    show: bool = False,
):
    """Lineplots showing a) loss curve (train vs val) and b) learning rate schedule."""
    apply_custom_theme()

    train_loss = np.array(history["train_loss"], dtype=float)
    val_loss   = np.array(history["val_loss"], dtype=float)
    lrs        = np.array(history["learning_rates"], dtype=float)
    epochs     = np.arange(1, len(train_loss) + 1)
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
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")

    ax.set_title(f"{country} — Loss Curve (Raw Loss, Log Scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
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
    ax2.set_ylabel("Normalized Loss")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(folder / fnames[0], dpi=160)
    print(f"[OK] Saved to {fnames[0]}")
    if show:
        plt.show()
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
    if show:
        plt.show()
    plt.close(fig2)

def plot_detailed_loss_curves(
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fname: str = "detailed_loss_curves.png",
    show: bool = False,
):
    """Plot separate loss curves for continuous and categorical components."""
    apply_custom_theme()

    if "train_cont_loss" not in history or "train_cat_loss" not in history:
        print(f"[INFO] Detailed loss components not available for {country}")
        return
    
    train_cont = np.array(history["train_cont_loss"], dtype=float)
    train_cat = np.array(history["train_cat_loss"], dtype=float)
    val_cont = np.array(history.get("val_cont_loss", []), dtype=float)
    val_cat = np.array(history.get("val_cat_loss", []), dtype=float)
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
                      [weights.get("cont_weight", 1.0), weights.get("cat_weight", 1.0)],
                      color=['blue', 'red'])
        axes[1, 1].set_title("Loss Weights")
        axes[1, 1].set_ylabel("Weight")
        axes[1, 1].grid(True, axis='y')
    
    plt.suptitle(f"{country} — Detailed Loss Analysis", fontsize=20)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved detailed loss curves to {fname}")
    if show:
        plt.show()
    plt.close(fig)

#########################################
##          VALIDATION PLOTS           ##
#########################################

def plot_error_histogram(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_error_histogram.png",
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
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_error_timeseries(
    country: str,
    df: pd.DataFrame,
    threshold: Optional[float],
    folder: Path = Path.cwd(),
    fname: str = "plot_error_timeseries.png",
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
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def summarize_validation(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "summarize_validation.png",
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
    with open(folder / fname, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


#########################################
##             TUNING PLOTS            ##
#########################################

def save_optuna_plots(study, folder: Path):
    """Save Optuna-provided visualizations as png if kaleido is installed and/or html file which requires no add package and has nicer formatting."""

    figs = {
        "optimization_history": plot_optimization_history(study),
        "param_importance": plot_param_importances(study),
        "parallel_coordinates": plot_parallel_coordinate(study),
        "slice": plot_slice(study),
        "contour": plot_contour(study),
    }
    for fname, fig in figs.items():
        png_fname = folder / f"{fname}.png"
        html_fname = folder / f"{fname}.html"
        try:
            fig.write_html(str(html_fname))
            fig.write_image(str(png_fname), scale=2)
        except Exception:
            pass

def plot_correlation_heatmap(
        df: pd.DataFrame, 
        folder: Path = Path.cwd(), 
        fname: str = "plot_correlation_heatmap.png",
        show: bool = False
    ):
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
        study, 
        country: str, 
        history_dir: Path, 
        folder: Path = Path.cwd(),
        fname: str = "plot_loss_curves_all_trials.png", 
        show: bool = False
    ):
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
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()

def plot_best_trial_learning_curve(
        best_history: dict, 
        folder: Path = Path.cwd(),
        fname: str = "plot_best_trial_learning_curve.png",
        show: bool = False
    ):
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

    if show:
        plt.show()

    plt.close()

def plot_3d_scatter(
        df: pd.DataFrame, 
        folder: Path = Path.cwd(), 
        fname: str = "plot_3d_scatter.png",
        show: bool = False
    ):
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

def plot_multi_country_overview(
        best_losses: dict, 
        folder: Path = Path.cwd(),
        fname: str = "plot_multi_country_overview.png",
        show: bool = False
    ):
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
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


#########################################
##             TESTING PLOTS           ##
#########################################

def plot_error_curve(
        country: str, 
        df_err: pd.DataFrame, 
        threshold: float, 
        method: str, 
        folder: Path = Path.cwd(),
        fname: str = "plot_error_curve.png", 
        show: bool = False
    ):
    """Lineplot reconstruction error over timestamps with color-coded error predictions, smoothed error curve, threshold and detected anomalies."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_err["ts"], df_err["error"], label="Error", alpha=0.6)
    # smoothed error
    df_err["smooth"] = df_err["error"].rolling(48, min_periods=1).mean()
    ax.plot(df_err["ts"], df_err["smooth"], label="Smoothed", linewidth=2)
    # threshold
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({method})")
    # anomalies
    anomalies = df_err[df_err["is_anomaly"] == 1]
    ax.scatter(anomalies["ts"], anomalies["error"], color="red", s=12, label="Detected")
    ax.set_title(f"{country} – Test Error Curve ({method})")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_intervals(
        country: str, 
        df_err: pd.DataFrame, 
        df_int: pd.DataFrame, 
        method: str, 
        folder: Path = Path.cwd(),
        fname: str = "plot_intervals.png", 
        show: bool = False
    ):
    """Lineplot with detected anomalies over timestamps."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 2))
    ax.plot(df_err["ts"], np.zeros_like(df_err["ts"]), alpha=0)  # invisible anchor
    for _, row in df_int.iterrows():
        ax.axvspan(row["start_ts"], row["end_ts"], color="red", alpha=0.3)
    ax.set_title(f"{country} – Anomaly Intervals ({method})")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_error_hist(
        country: str, 
        df_err: pd.DataFrame, 
        threshold: float, 
        method: str, 
        folder: Path = Path.cwd(),
        fname: str = "plot_error_hist.png", 
        show: bool = False
    ):
    """Histogram showing error counts and threshold."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_err["error"], bins=60, ax=ax)
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylabel("Log(counts)")
    ax.set_title(f"{country} – Error Histogram ({method})")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_raw_with_errors(
        signal_name: str,
        ts: Union[np.ndarray, pd.Index, pd.Series], 
        raw_signal: np.ndarray, 
        errors: np.ndarray, 
        mask: np.ndarray, 
        folder: Path = Path.cwd(),
        fname: str = "plot_error_hist.png", 
        show: bool = False
    ):
    """Lineplot showing raw target signal with smoothed error scaled on same range and detected anomalies."""
    apply_custom_theme()

    # normalize errors to same scale as raw signal
    err_norm = errors / np.max(errors) * (raw_signal.max() - raw_signal.min()) * 0.4
    err_norm = err_norm + raw_signal.min()  # shift upward

    plt.figure(figsize=(16, 6))
    plt.plot(ts, raw_signal, label=f"Raw {signal_name} signal", color='black', linewidth=1.4)
    plt.plot(ts, err_norm, label="Scaled error", color='orange', alpha=0.7)
    # annotate anomalies
    plt.scatter(ts[mask], raw_signal[mask], color='red', label='Detected snomalies', s=25)
    plt.title("Raw Signal with Scaled Reconstruction Error Overlay")
    plt.legend()
    plt.grid(True)
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


#########################################
##             LATENT SPACE            ##
#########################################

def plot_latent_space(
    country: str,
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    model: TabularAutoencoder,
    device: str,
    max_samples: int = 1000,
    folder: Path = Path.cwd(),
    fname: str = "plot_latent_space.png",
    show: bool = False,
):
    """Scatter plot showing latent space using PCA/t-SNE."""
    apply_custom_theme()
    
    if len(X_cont) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_cont), max_samples, replace=False)
        X_cont = X_cont[indices]
        X_cat = X_cat[indices]
        print(f"[INFO] Using random subset of {max_samples} samples for latent space")

    Xc_tensor = torch.from_numpy(X_cont).to(device)
    Xk_tensor = torch.from_numpy(X_cat).to(device)
    
    model.eval()
    with torch.no_grad():
        z = model.encode(Xc_tensor, Xk_tensor).cpu().numpy()
    
    if len(z) < 10:
        print(f"[INFO] Not enough samples for latent space visualization: {len(z)}")
        return
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    
    if len(z) >= 50:
        tsne = TSNE(n_components=2, perplexity=min(30, len(z)-1), random_state=42)
        z_tsne = tsne.fit_transform(z)
    
    fig, axes = plt.subplots(1, 3 if len(z) >= 50 else 2, figsize=(18, 6))
    
    axes[0].scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6, s=20)
    axes[0].set_title(f"{country} — Latent Space (PCA)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(z, showfliers=False)
    axes[1].set_title(f"{country} — Latent Dimension Distributions")
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, axis='y', alpha=0.3)
    
    if len(z) >= 50:
        axes[2].scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.6, s=20)
        axes[2].set_title(f"{country} — Latent Space (t-SNE)")
        axes[2].set_xlabel("t-SNE 1")
        axes[2].set_ylabel("t-SNE 2")
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"{country} — Latent Space Analysis", fontsize=16)
    plt.tight_layout()
    
    pca_path = folder / f"{fname[:-4]}_pca_coords.csv"
    pd.DataFrame(z_pca, columns=['PC1', 'PC2']).to_csv(pca_path, index=False)
    
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved latent space visualization to {fname}")
    if show: plt.show()
    plt.close(fig)