import json, optuna
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from app.src.data.attack_labelling import ATTACK_LABELS, ID_TO_ATTACK

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


def plot_detailed_mt_loss_curves(
    country: str,
    history: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_detailed_mt_loss_curves.png",
    show: bool = False,
):
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
    print(f"[OK] Saved detailed loss curves to {fname}")
    if show: plt.show()
    plt.close(fig)


#########################################
##          VALIDATION PLOTS           ##
#########################################

def summarize_mt_validation(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "summarize_mt_validation.png",
) -> dict:
    """Summary statistics for MT validation output as json."""
    summary: dict = {"country": country}

    # -------------------------------
    # 1. Anomaly / total loss statistics
    # -------------------------------
    if "loss_total" in df:
        total = df["loss_total"].to_numpy(dtype=float)

        summary["anomaly"] = {
            "count": int(len(total)),
            "mean": float(np.mean(total)),
            "median": float(np.median(total)),
            "std": float(np.std(total)),
            "min": float(np.min(total)),
            "max": float(np.max(total)),
            "p95": float(np.percentile(total, 95)),
            "p99": float(np.percentile(total, 99)),
            "p995": float(np.percentile(total, 99.5)),
        }

        if "threshold" in df:
            summary["anomaly"]["threshold"] = float(df["threshold"].iloc[0])

        if "is_flagged" in df:
            summary["anomaly"]["flagged_count"] = int(df["is_flagged"].sum())
            summary["anomaly"]["flagged_ratio"] = float(df["is_flagged"].mean())
    
    # -------------------------------
    # 2. L3 regression metrics
    # -------------------------------
    if "l3_true" in df and "l3_pred" in df:
        y_true = df["l3_true"].to_numpy()
        y_pred = df["l3_pred"].to_numpy()

        summary["l3"] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
        }

    # -------------------------------
    # 3. L7 regression metrics
    # -------------------------------
    if "l7_true" in df and "l7_pred" in df:
        y_true = df["l7_true"].to_numpy()
        y_pred = df["l7_pred"].to_numpy()

        summary["l7"] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mean_true": float(np.mean(y_true)),
            "mean_pred": float(np.mean(y_pred)),
        }

    # -------------------------------
    # 4. Attack classification metrics
    # -------------------------------
    if "attack_pred" in df:
        attack_block = {
            "mean_confidence": float(df["attack_conf"].mean())
            if "attack_conf" in df else None
        }

        # ground truth available; full metrics
        if "attack_true" in df:
            y_true = df["attack_true"].to_numpy()
            y_pred = df["attack_pred"].to_numpy()

            attack_block.update({
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            })

        # no ground truth; inference-only stats
        else:
            attack_block.update({
                "positive_rate": float(df["attack_pred"].mean()),
            })

        summary["attack"] = attack_block
    
    with open(folder / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved MT val summary to {fname}")
    return summary

def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    max_points: int = 5000,
    folder: Path = Path.cwd(),
    fname: str = "plot_regression_scatter.png",
    show: bool = False,
):
    """Scatter plot of true vs predicted regression target."""
    apply_custom_theme()

    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)

    ax.set_title(f"{label}: True vs Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_attack_confusion_matrix(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_confusion_matrix.png",
    show: bool = False,
):
    """Confusion matrix for attack classification head."""
    required = ["attack_pred", "attack_true"]
    if not all(k in df for k in required):
        print(f"[INFO] Confusion matrix components not available for {country}")
        return
    apply_custom_theme()

    y_true = df["attack_true"]
    y_pred = df["attack_pred"]
    labels = list(range(len(ATTACK_LABELS)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    disp_raw = ConfusionMatrixDisplay(
        confusion_matrix=cm_raw,
        display_labels=ID_TO_ATTACK.keys(),
    )
    disp_raw.plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axes[0].set_title("Counts")

    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=ID_TO_ATTACK.keys(),
    )
    disp_norm.plot(
        ax=axes[1],
        cmap="Blues",
        colorbar=True,
        values_format=".2f",
    )
    axes[1].set_title("Normalized")

    legend_text = "\n".join(
        [f"{i}: {name}" for i, name in enumerate(ATTACK_LABELS)]
    )
    fig.text(
        0.905, 0.5,
        legend_text,
        va="center",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )
    plt.suptitle(f"{country} — Attack Classification Confusion Matrix")
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_attack_confidence_hist(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_confidence_hist.png",
    show: bool = False,
):
    """Histogram of max softmax confidence for attack predictions."""
    required = ["attack_conf", "attack_true"]
    if not all(k in df for k in required):
        print(f"[INFO] Confidence histogram components not available for {country}")
        return
    apply_custom_theme()
    
    attack_prob_max = df["attack_conf"]
    y_true = df["attack_true"]
    benign_mask = y_true == 0
    attack_mask = y_true != 0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        attack_prob_max[benign_mask],
        bins=40,
        alpha=0.5,
        label="Benign",
        density=True,
    )
    ax.hist(
        attack_prob_max[attack_mask],
        bins=40,
        alpha=0.5,
        label="Attack",
        density=True,
    )

    ax.set_title("Attack Prediction Confidence")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_mt_anomaly_timeseries(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_mt_anomaly_timeseries.png",
    show: bool = False,
    show_components: bool = True,
    show_attack_conf: bool = True,
):
    """
    Plot MT anomaly score over time with:
      - threshold
      - flagged anomaly regions
      - optional regression component overlays
      - optional attack confidence overlay (secondary axis)
    """
    apply_custom_theme()

    required = ["ts", "loss_total", "threshold", "is_flagged"]
    if not all(k in df for k in required):
        print(f"[INFO] Missing columns for anomaly timeseries in {country}")
        return

    ts = df["ts"]
    score = df["loss_total"]
    threshold = df["threshold"].iloc[0]
    flagged = df["is_flagged"].astype(bool)

    fig, ax = plt.subplots(figsize=(15, 4))

    # -----------------------------
    # Main anomaly score
    # -----------------------------
    ax.plot(
        ts,
        score,
        lw=1.4,
        color="steelblue",
        label="Anomaly score",
        zorder=3,
    )

    # Threshold
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Threshold = {threshold:.3f}",
        zorder=4,
    )

    # Flagged regions
    if flagged.any():
        ax.fill_between(
            ts,
            score,
            threshold,
            where=flagged,
            color="red",
            alpha=0.25,
            label="Flagged anomaly",
            interpolate=True,
            zorder=2,
        )

    # -----------------------------
    # Component overlays (same axis)
    # -----------------------------
    if show_components:
        if "loss_l3" in df:
            ax.plot(
                ts,
                df["loss_l3"],
                lw=0.8,
                linestyle=":",
                color="green",
                alpha=0.6,
                label="L3 loss",
            )
        if "loss_l7" in df:
            ax.plot(
                ts,
                df["loss_l7"],
                lw=0.8,
                linestyle=":",
                color="orange",
                alpha=0.6,
                label="L7 loss",
            )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Anomaly score / regression loss", fontsize=14)
    ax.set_title(f"{country} — Multi-Task Anomaly Score Over Time", fontsize=16)
    ax.grid(True)

    # -----------------------------
    # Attack confidence (secondary axis)
    # -----------------------------
    if show_attack_conf and "attack_conf" in df:
        ax2 = ax.twinx()
        ax2.plot(
            ts,
            df["attack_conf"],
            lw=1.0,
            color="purple",
            alpha=0.35,
            label="Attack confidence",
        )
        ax2.tick_params(axis="y", labelsize=10)
        ax2.set_ylabel("Attack confidence", fontsize=14)
        ax2.set_ylim(0.0, 1.05)

        # merge legends cleanly
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=10,
            loc="upper right",
            bbox_to_anchor=(1.35, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    else:
        ax.legend(
            fontsize=10,
            loc="upper left",
            bbox_to_anchor=(1.35, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(folder / fname, dpi=160, bbox_inches="tight")
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


#########################################
##             TUNING PLOTS            ##
#########################################


def plot_mt_loss_component_analysis(
    study: optuna.Study,
    country: str, 
    history_dir: Path,
    folder: Path = Path.cwd(),
    fname: str = "plot_mt_loss_component_analysis.png",
    show: bool = False
):
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


def plot_multi_mt_weights_overview(
    best_weights: dict,
    folder: Path = Path.cwd(),
    fname: str = "plot_multi_mt_weights_overview.png",
    show: bool = False,
):
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
):
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


#########################################
##             TESTING PLOTS           ##
#########################################

def plot_true_pred_anomalies(
    signal_name: str,
    ts: pd.Series, 
    y_true: pd.Series, 
    y_pred: pd.Series, 
    mask: np.ndarray, 
    anomaly_starts: pd.Series,
    anomaly_ends: pd.Series,
    folder: Path = Path.cwd(),
    fname: str = "plot_true_pred_anomalies.png", 
    show: bool = False
):
    """Lineplot showing raw signal with predicted signal and flagged anomalies."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        ts, 
        y_true, 
        label=f"{signal_name} true", 
        color='black', 
        linewidth=1.2, 
        zorder=2
    )
    ax.plot(
        ts, 
        y_pred, 
        label=f"{signal_name} pred", 
        color='orange', 
        linewidth=1.4, 
        zorder=3
    )
    if anomaly_starts is not None and anomaly_ends is not None:
        n = len(ts)
        for s, e in zip(anomaly_starts, anomaly_ends):
            if s < 0 or e > n:
                continue
            ax.axvspan(
                ts[s],
                ts[e - 1],
                color="red",
                alpha=0.6,
                zorder=1,
            )
    # annotate anomalies
    if mask is not None and mask.any():
        ax.scatter(
            ts[mask],
            y_true[mask],
            color="red",
            s=16,
            alpha=0.6,
            label="Anomaly",
            zorder=4,
        )
    plt.title(f"{signal_name} true vs pred with anomalies")
    ax.set_xlabel("Time")
    ax.set_ylabel(signal_name)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()

def plot_attack_timeline(
        df, 
        folder: Path = Path.cwd(), 
        fname: str = "plot_attack_timeline.png", 
        show: bool = False
    ):
    """Scatter plot of attack type over time"""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 2))
    ax.scatter(
        df["ts"],
        df["attack_pred"],
        c=df["attack_pred"],
        cmap="tab10",
        s=12,
        alpha=0.8
    )
    ax.set_title(f"Attack type timeline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Attack class")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    if show: plt.show()
    plt.close(fig)

def plot_loss_components_timeseries(
        df, 
        folder: Path = Path.cwd(), 
        fname: str = "plot_loss_components_timeseries.png", 
        show: bool = False
    ):
    """Lineplot with loss decomposition over time"""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["ts"], df["loss_l3"], label="L3 loss", alpha=0.7)
    ax.plot(df["ts"], df["loss_l7"], label="L7 loss", alpha=0.7)
    if "loss_attack" in df:
        ax.plot(df["ts"], df["loss_attack"], label="Attack loss", alpha=0.7)

    ax.set_yscale("log")
    ax.set_title(f"Loss components over time")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    if show: plt.show()
    plt.close(fig)

