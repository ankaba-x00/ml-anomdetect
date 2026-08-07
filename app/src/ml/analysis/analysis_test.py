from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .common import apply_custom_theme


def plot_error_curve(
    country: str, 
    df_err: pd.DataFrame, 
    threshold: float, 
    method: str, 
    folder: Path = Path.cwd(),
    fname: str = "plot_error_curve.png", 
    show: bool = False
) -> None:
    """Lineplot reconstruction error over timestamps with color-coded error predictions, smoothed error curve, threshold and detected anomalies."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_err["ts"], df_err["scores"], label="Scores", alpha=0.6)
    # smoothed error
    df_err["smooth"] = df_err["scores"].rolling(48, min_periods=1).mean()
    ax.plot(df_err["ts"], df_err["smooth"], label="Smoothed", linewidth=2)
    # threshold
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({method})")
    # anomalies
    anomalies = df_err[df_err["is_anomaly"] == 1]
    ax.scatter(anomalies["ts"], anomalies["scores"], color="red", s=12, label="Detected")
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
) -> None:
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
    df: pd.DataFrame, 
    threshold: float, 
    method: str, 
    folder: Path = Path.cwd(),
    fname: str = "plot_score_hist.png", 
    show: bool = False,
    MT: bool = False
) -> None:
    """Histogram showing score counts and threshold."""
    apply_custom_theme()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["loss_total"] if MT else df["scores"], bins=60, ax=ax)
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylabel("Log(counts)")
    ax.set_title(f"{country} – Score Histogram ({method})")
    plt.tight_layout()
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)


def plot_raw_with_scores(
    signal_name: str,
    ts: Union[np.ndarray, pd.Index, pd.Series], 
    raw_signal: np.ndarray, 
    scores: np.ndarray, 
    mask: np.ndarray, 
    folder: Path = Path.cwd(),
    fname: str = "plot_raw_with_scores.png", 
    show: bool = False
) -> None:
    """Lineplot showing raw target signal with smoothed score scaled on same range and detected anomalies."""
    apply_custom_theme()

    # normalize scores to same scale as raw signal
    err_norm = scores / np.max(scores) * (raw_signal.max() - raw_signal.min()) * 0.4
    err_norm = err_norm + raw_signal.min()  # shift upward

    plt.figure(figsize=(16, 6))
    plt.plot(ts, raw_signal, label=f"Raw {signal_name} signal", color='black', linewidth=1.4)
    plt.plot(ts, err_norm, label="Scaled score", color='orange', alpha=0.7)
    # annotate anomalies
    plt.scatter(ts[mask], raw_signal[mask], color='red', label='Detected snomalies', s=25)
    plt.title("Raw Signal with Scaled Reconstruction Error Overlay")
    plt.legend()
    plt.grid(True)
    plt.savefig(folder / fname, dpi=150)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close()


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
) -> None:
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
) -> None:
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
) -> None:
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
