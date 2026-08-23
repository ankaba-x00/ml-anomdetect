import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .common import apply_custom_theme
from app.src.data.attack_labelling import ID_TO_ATTACK
from app.src.data.split import timeseries_seq_split
from app.src.data.feature_engineering import    build_supervised_feature_matrix


def plot_l3_l7_scatter_by_attack(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_timeseries.png",
    show: bool = False,
):
    apply_custom_theme()

    colors = ["white", "orange", "red", "magenta", "blue", "green", "cyan", "brown"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for k, v in ID_TO_ATTACK.items():
        df_red = df[df["attack_label"] == k]
        ax.scatter(
            df_red["l3_origin"],
            df_red["l7_traffic"],
            color=colors[k],
            label=v,
            alpha=0.4,
            s=16,
            edgecolors="none",
        )

    ax.set_title(f"{country} — L3 vs L7 by attack label", fontsize=16)
    ax.set_xlabel("L3 origin")
    ax.set_ylabel("L7 traffic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5, 1.5)
    ax.set_ylim(1e-6, 1.5)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def plot_timeseries_with_attack_labels(
    country: str,
    df: pd.DataFrame,
    folder: Path = Path.cwd(),
    fname: str = "plot_attack_timeseries.png",
    show: bool = False,
) -> None:
    """Lineplot error over time with optional threshold line."""
    apply_custom_theme()

    colors = ["white", "orange", "red", "magenta", "blue", "green", "cyan", "brown"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["l3_origin"], linewidth=.4, color="#787878")
    ax.plot([], [], color="#787878", linewidth=4, label="l3 origin")

    ax.plot(df.index, df["l7_traffic"], linewidth=.4, color="black")
    ax.plot([], [], color="black", linewidth=4, label="l7 traffic")

    for k, v in ID_TO_ATTACK.items():
        if v == "normal":
            continue
        df_red = df.loc[df["attack_label"] == k]
        ax.vlines(
            x=df_red.index,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            color=colors[k],
            alpha=.5,
            linewidth=.4,
            zorder=0,
        )
        ax.plot([], [], color=colors[k], alpha=.5, linewidth=4, label=v)
    ax.set_title(f"{country} — Traffic with attack labels")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic (log)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    fig.savefig(folder / fname, dpi=160)
    print(f"[OK] Saved to {fname}")
    if show: plt.show()
    plt.close(fig)

def print_labeldist(country: str, df: pd.DataFrame) -> None:
    """Prints label distribution for different train-val-test split ratios."""
    _, _, _, _, ya_s, _, _ = build_supervised_feature_matrix(country, df)
    ya = ya_s.values.astype(np.int64)

    ratios = [
        (75, 15), 
        (80, 15), 
        (80, 10), 
        (90, 5), 
        (100, 0)
    ]
    print(f"==============================")
    print(f"    {country} LABEL DISTRIBUTION")
    print(f"==============================")
    print("n_labels :", ya.shape[0])
    for tr, vr in ratios:
        print(f"\n{tr}% train | {vr}% val | {100 - tr - vr}% test")

        ya_tr, ya_val, ya_te = timeseries_seq_split(ya, None, tr / 100, vr / 100)
        splits = {
            "train": ya_tr,
            "val": ya_val,
            "test": ya_te,
        }
        counts = {k: np.bincount(v, minlength=8) for k, v in splits.items()}

        col_widths = [
            max(len(str(counts[k][i])) for k in counts)
            for i in (range(8))
        ]
        name_width = max(len(k) for k in splits)

        for split_name, v in counts.items():
            row = [f"{i}: {v[i]:<{col_widths[i]}}" for i in range(8)]
            print(f"→ {split_name:<{name_width}}  " + " | ".join(row))
