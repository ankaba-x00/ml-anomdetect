import json
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
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
from .common import apply_custom_theme


def plot_error_histogram(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_error_histogram.png",
    show: bool = False,
) -> None:
    """Histogram of log errors with percentile lines."""
    apply_custom_theme()

    scores = df["scores"].values
    log_err = np.log10(scores + 1e-8)
    p95 = np.percentile(scores, 95)
    p99 = np.percentile(scores, 99)
    p995 = np.percentile(scores, 99.5)
    med = np.median(scores)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(log_err, bins=60, color="steelblue", alpha=0.7)
    # marker lines (in log space for consistency)
    for p, label in [(med, "median"), (p95, "p95"), (p99, "p99"), (p995, "p995")]:
        ax.axvline(np.log10(p + 1e-8), linestyle="--", label=label)
    ax.set_title(f"{country} — Validation Error Distribution (log10 scale)")
    ax.set_xlabel("log10(scores)")
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
) -> None:
    """Lineplot error over time with optional threshold line."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["ts"], df["scores"], linewidth=1, label="Scores")
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
) -> None:
    """Summary statistics for reconstruction error distribution as json."""
    scores = df["scores"].values

    summary = {
        "country": country,
        "count": int(len(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "p995": float(np.percentile(scores, 99.5)),
    }

    with open(folder / fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved to {fname}")


def summarize_mt_validation(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "summarize_mt_validation.png",
) -> None:
    """Summary statistics for MT validation output as json."""
    summary = {"country": country}

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
    print(f"[OK] Saved to {fname}")


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    max_points: int = 5000,
    folder: Path = Path.cwd(),
    fname: str = "plot_regression_scatter.png",
    show: bool = False,
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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