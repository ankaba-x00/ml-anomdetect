import os, warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib as mpl
from sklearn.decomposition import PCA
from core.time_utils import conv_iso_to_utc
warnings.filterwarnings("ignore", message=".*tight_layout.*")
pd.set_option('future.no_silent_downcasting', True)


#########################################
##                CONFIG               ##
#########################################

custom_rc = {
    "figure.figsize": (24, 6),
    "figure.titlesize": 25,
    "axes.titlesize": 23,
    "axes.titlepad": 35,
    "axes.labelsize": 22,
    "axes.labelpad": 10,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.family": "Arial",
    "legend.title_fontsize": 16,
    "legend.fontsize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}

def apply_custom_theme() -> None:
    """Apply consistent Seaborn + Matplotlib styling."""
    sns.set_theme(style="whitegrid", rc=custom_rc)
    sns.set_context("poster", rc=custom_rc)


#########################################
##              BOXPLOTS               ##
#########################################

def boxplot_valdist(
    df: pd.DataFrame,
    X: str = "countries", 
    Y: str = "values",
    top_num : int = 30, 
    col: str = "lightblue", 
    title: str = "",
    xlabel: str = "", 
    ylabel: str = "",
    folder: str = "./",
    fname: str = "boxplot_valdist.png"
):
    """Boxplot showing value distribution by country."""
    apply_custom_theme()
    
    country_medians = df.groupby(X)[Y].median().sort_values(ascending=False)
    topX = country_medians.head(top_num).index
    df_topX = df[df[X].isin(topX)]
    order = topX

    fig, ax = plt.subplots(figsize=(24, 10))
    sns.boxplot(
        data=df_topX,
        x=X,
        y=Y,
        order=order,
        fliersize=8,
        linewidth=1,
        whis=[0, 100],
        color=col,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(0, 30, 3))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(
        True, 
        axis="both", 
        which="major", 
        linestyle="--", 
        alpha=0.4, 
        zorder=0
    )
    # print(sns.plotting_context())
    # print(plt.rcParams)
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()

def boxplot_valdist_daytypes(
    df: pd.DataFrame,
    X: str = "countries", 
    Y: str = "values",
    top_num : int = 30,
    HUE: str = "daytype",
    palette: dict = {"Weekday": "yellow", "Weekend": "orange"},
    title: str = "", 
    xlabel: str = "", 
    ylabel: str = "",
    folder: str = "./",
    fname: str = "boxplot_valdist_daytypes.png"
):
    """Boxplot showing value distribution by country split into the following day types: 
        - weekdays: Mon-Fri
        - weekends: Sat-Sun
    """
    apply_custom_theme()

    country_medians = df.groupby(X)[Y].median().sort_values(ascending=False)
    topX = country_medians.head(top_num).index
    df_topX = df[df[X].isin(topX)]
    order = topX

    fig, ax = plt.subplots(figsize=(24, 10))
    sns.boxplot(
        data=df_topX,
        x=X,
        y=Y,
        order=order,
        hue=HUE,
        fliersize=8,
        linewidth=1,
        whis=[0, 100],
        palette=palette,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=90)
    ax.set_yticks(range(0, 31, 3))
    ax.grid(
        True, 
        axis="both", 
        which="major", 
        linestyle="--", 
        alpha=0.4, 
        zorder=0
    )
    ax.legend(
        title="Day Type", 
        loc="upper right", 
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()

def boxplot_valdist_daytimes(
    df: pd.DataFrame,
    X: str = "regions",
    Y: str = "values",
    top_num : int = 30,
    HUE: str = "daytime",
    palette: dict = {
        "Deep night": "yellow",
        "Morning": "red",
        "Business hours": "green",
        "Evening": "blue",
        "Early night": "purple"
    },
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    folder: str = "./",
    fname: str = "boxplot_valdist_daytimes.png"
):
    """Boxplot showing value distribution by country split into the following local daytimes:
        - Deep night: 00-06
        - Morning: 06-09
        - Business hours: 09-17
        - Evening: 17-22
        - Early night: 22-00
    """
    apply_custom_theme()

    country_medians = df.groupby(X)[Y].median().sort_values(ascending=False)
    topX = country_medians.head(top_num).index
    df_topX = df[df[X].isin(topX)]
    order = topX

    hue_order = ["Deep night", "Morning", "Business hours", "Evening", "Early night"]

    fig, ax = plt.subplots(figsize=(28, 10))
    sns.boxplot(
        data=df_topX,
        x=X,
        y=Y,
        order=order,
        hue=HUE,
        hue_order=hue_order,
        fliersize=5,
        linewidth=1,
        whis=[0, 100],
        palette=palette,
        ax=ax
    )
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", rotation=90)
    ax.grid(
        True, 
        axis="both", 
        which="major", 
        linestyle="--", 
        alpha=0.4, 
        zorder=0
    )
    ax.legend(
        title="Daytime (local time)", 
        loc="lower center",
        bbox_to_anchor=(0.5, 0.),
        ncol=5,
        frameon=True
    )
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.4, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()


#########################################
##              HEATMAPS               ##
#########################################

def heatmap_activity_fluctuations(
    df: pd.DataFrame,
    drop_cols: list = ["range", "std", "ratio"],
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "YlOrRd",
    top_n: int = None,
    title: str = "",
    xlabel: str = "Daytime",
    ylabel: str = "Countries",
    folder: str = "./",
    fname: str = "heatmap_activity_fluctuations.png"
):
    """
    Plot heatmap of median activity per country across daytimes.
    """
    apply_custom_theme()

    heatmap_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if top_n:
        heatmap_df = heatmap_df.head(top_n)
    
    plt.figure(figsize=(max(6, len(heatmap_df.columns) * 3 - 1), max(6, len(heatmap_df) * 0.4)))
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 20},
        cbar_kws={"shrink": 0.5, "aspect": 100}
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close()
    # plt.show()

def heatmap_anomalies(
        df: pd.DataFrame,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        col1: str = "#DC5B37",
        palette1: list = ["#C9F89E", "#46AD62", "#2F8F4B", "#1E603A"],
        col2: str = "#3EA25A",
        palette2: list = ["#F9A506", "#E86B3F", "#C53B2A", "#A9282E"],
        folder: str = "./",
        fname: str = "heatmap_anomalies.png"
    ):
    """Heatmap with anomalies by location and date. Color coding differentiates between type (Location vs. AS) and duration."""
    apply_custom_theme()

    for col in ["dates", "startDates", "endDates"]:
        df[col] = conv_iso_to_utc(df[col])

    df['duration'] = (df['endDates'] - df['startDates']).dt.total_seconds() / 3600.
    df['ongoing'] = df['endDates'].isna()
    df['type_code'] = df['types'].map({'AS': 1, 'LOCATION': 0})

    pivots = {
        "duration": df.pivot_table(index="location", columns="dates", values="duration", aggfunc="mean"),
        "type": df.pivot_table(index="location", columns="dates", values="type_code", aggfunc="mean"),
        "ongoing": df.pivot_table(index="location", columns="dates", values="ongoing", aggfunc="max"),
    }
    D, T, O = [p.fillna(0 if k != "ongoing" else False).reindex(sorted(p.columns), axis=1)
           for k, p in pivots.items()]
        
    norm = Normalize(vmin=0, vmax=np.nanpercentile(df["duration"], 95)) 

    teal_cmap = LinearSegmentedColormap.from_list(
        "green", palette1
    )
    orange_cmap = LinearSegmentedColormap.from_list(
        "orange", palette2
    )

    fig, ax = plt.subplots(figsize=(30, 20))
    for i, (loc, row) in enumerate(D.iterrows()):
        for j, (date, dur) in enumerate(row.items()):
            if np.isnan(dur):
                continue
            tcode, ongoing = T.iloc[i, j], O.iloc[i, j]
            cmap = orange_cmap if tcode == 1 else teal_cmap
            color = cmap(norm(dur))
            rect = mpl.patches.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="gray", lw=0.3)
            ax.add_patch(rect)

            if ongoing:
                ax.add_patch(
                    mpl.patches.Rectangle((j, i), 1, 1, facecolor="none", edgecolor="black",
                                        lw=0.6, hatch="///")
                )
    ax.set_xlim(0, len(D.columns))
    ax.set_ylim(0, len(D.index))
    ax.set_xticks(np.arange(len(D.columns)) + 0.5)
    ax.set_yticks(np.arange(len(D.index)) + 0.5)
    ax.set_xticklabels([d.strftime("%m/%d") for d in D.columns], rotation=90)
    ax.set_yticklabels(D.index)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=30, pad=20)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    
    cb_ax1 = fig.add_axes([0.15, 0.05, 0.35, 0.02])
    cb_ax2 = fig.add_axes([0.55, 0.05, 0.35, 0.02])
    for cmap, cax, label in [(teal_cmap, cb_ax1, "AS duration (hours)"),
                             (orange_cmap, cb_ax2, "LOCATION duration (hours)")]:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(label)
    
    legend_elements = [
        Patch(facecolor=col1, edgecolor="gray", label="AS (duration color)"),
        Patch(facecolor=col2, edgecolor="gray", label="LOCATION (duration color)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Ongoing (no endDate)"),
    ]
    fig.legend(
        handles=legend_elements,
        title='Legend',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()

def heatmap_log10_attack_profile(
        norm_matrix: pd.DataFrame,
        name: str = "l3_origin",
        xlabel: str = "Dates",
        ylabel: str = "Countries",
        cmap: str = "viridis",
        folder: str = "./",
        fname: str = "heatmap_log10_attack_profile.png"
    ):
    """"""
    apply_custom_theme()

    cols_converted = [conv_iso_to_utc(str(c)) for c in norm_matrix.columns]
    norm_matrix2 = norm_matrix.copy()
    norm_matrix2.columns = cols_converted

    formatted_labels = [
        d.strftime("%m/%d") if hasattr(d, "strftime") else str(d)
        for d in norm_matrix2.columns
    ]
    
    filt = norm_matrix.loc[norm_matrix2.sum(axis=1) > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.log10(filt + 1e-9)

    plt.figure(figsize=(20, max(6, 0.3 * len(filt))))
    ax = sns.heatmap(
        heat,
        cmap=cmap,
        xticklabels=formatted_labels,
        yticklabels=True,
        linewidths=0.2,
        linecolor="gray",
        cbar_kws={"label": "log10(normalized share)", "shrink": 0.5, "aspect": 100}
    )
    ax.set_xticklabels(formatted_labels, rotation=90)
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 5 != 0:
            label.set_visible(False)
    plt.title(f"{name[:2].upper()} Attack Profile: Normalized Daily Shares (log10)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close()
    # plt.show()


#########################################
##              BARPLOTS               ##
#########################################

def barplot_activity_fluctuations(
    df: pd.DataFrame,
    metric: str = "range",
    top_n: int = 30,
    head: bool = True,
    title: str = "",
    xlabel: str = "Countries",
    ylabel: str = "Median activity range (max - min)",
    col: str = "lightblue",
    folder: str = "./",
    fname: str = "barplot_activity_fluctuations.png"
):
    """
    Plot bar chart of top countries by daytime fluctuation metric (range/std/ratio).
    """
    apply_custom_theme()

    fluctuation_sorted = df.sort_values(metric, ascending=False)
    if head:
        top_fluctuating = fluctuation_sorted.head(top_n)
    else: 
        top_fluctuating = fluctuation_sorted.tail(top_n)

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=top_fluctuating.reset_index(),
        x=top_fluctuating.index.name,
        y=metric,
        color=col
    )
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close()
    # plt.show()

def barplot_top_attackers(
    df: pd.DataFrame,
    top_n: int = 30,
    name: str = "l3_origin",
    xlabel: str = "Total L3 Attack Volume",
    ylabel: str = "Countries",
    palette: str = "Blues_r",
    folder: str = "./",
    fname: str = "barplot_top_attackers.png"
):
    """Plot barplot with top attack countries worldwide."""
    apply_custom_theme()

    df_totals = (
        df.groupby("countries")["values"]
        .sum()
        .sort_values(ascending=False)
    )
    top_c = df_totals.head(top_n)
    
    plt.figure(figsize=(10, 0.35 * top_n + 3))
    sns.barplot(
        x=top_c.values,
        y=top_c.index,
        hue=top_c.index,
        dodge=False,
        orient="h",
        palette=palette
    )
    plt.title(f"Top {top_n} {name[:2].upper()} Attack Origin Countries")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close()
    # plt.show()


#########################################
##              LINEPLOTS              ##
#########################################

def lineplot_clustering_eval_curves(
        df: pd.DataFrame,
        col: str = "blue",
        folder: str = "./",
        fname: str = "lineplot_clustering_eval_curves.png"
    ):
    """Plot clustering eval curves to determine ideal k value."""
    apply_custom_theme()

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax = ax.ravel()
    
    xticks = list(range(2, 16, 2))

    # SSE (Elbow)
    ax[0].plot(df["k"], df["SSE"], marker="o", color=col)
    ax[0].set_title("SSE (Elbow Method)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("SSE")
    ax[0].set_xticks(xticks)

    # Silhouette
    ax[1].plot(df["k"], df["silhouette"], marker="o", color=col)
    ax[1].set_title("Silhouette Score")
    ax[1].set_xlabel("k")
    ax[1].set_xticks(xticks)

    # Calinski–Harabasz
    ax[2].plot(df["k"], df["calinski"], marker="o", color=col)
    ax[2].set_title("Calinski–Harabasz Index")
    ax[2].set_xlabel("k")
    ax[2].set_xticks(xticks)

    # Davies–Bouldin (lower is better)
    ax[3].plot(df["k"], df["davies"], marker="o", color=col)
    ax[3].set_title("Davies–Bouldin Index")
    ax[3].set_xlabel("k")
    ax[3].set_xticks(xticks)

    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()


#########################################
##             CLUSTERING              ##
#########################################

def pca_clusterplot(
        X: np.ndarray, 
        labels: np.ndarray, 
        countries: list[str],
        k: int,
        name: str = "l3_origin",
        folder: str = "./",
        fname: str = "pca_clusterplot.png"
    ):
    apply_custom_theme()

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X2[:, 0], y=X2[:, 1],
        hue=labels,
        palette="tab10",
        s=100
    )
    for i, c in enumerate(countries):
        plt.text(X2[i, 0], X2[i, 1], c, fontsize=10)
    ex1, ex2 = pca.explained_variance_ratio_[:2]
    plt.title(f"PCA {name[:2].upper()} attack profile (k={k}) w. variance ratio {ex1:.2%} / {ex2:.2%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close()
    # plt.show()


#########################################
##             RADAR CHARTS            ##
#########################################

def radarchart_attack_fingerprint(
        L3: pd.DataFrame,
        L7: pd.DataFrame,
        country: str = "US",
        dir: str = "origin",
        top_n: int = 30,
        col1= "#133a88",
        col2= "#971613",
        folder: str = "./",
        fname: str = "radarchart_attack_fingerprint.png"
    ):
    """
    Plot attack fingerprint for a given country based on L3 and L3 normalized share for the top_n attacking countries.
    """
    apply_custom_theme()

    # take top interaction countries (by L3+L7 avg)
    combined = L3 + L7
    top = combined.sort_values(ascending=False).head(top_n).index.tolist()
    vals_L3 = np.array([L3.get(c, 0) for c in top])
    vals_L7 = np.array([L7.get(c, 0) for c in top])
    # normalize each to 0–1
    L3n = vals_L3 / vals_L3.max() if vals_L3.max() > 0 else vals_L3
    L7n = vals_L7 / vals_L7.max() if vals_L7.max() > 0 else vals_L7

    N = len(top)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)  
    def close(v): return np.concatenate([v, [v[0]]])
    angles_c = close(angles)

    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.grid(True, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    for angle in angles:
        ax.plot([angle, angle], [0, 1], color="gray", linewidth=0.6, alpha=0.5)
    ax.plot(angles_c, close(L3n), color=col1, linewidth=2)
    ax.fill(angles_c, close(L3n), color=col1, alpha=0.2, label=f"L3 {dir}")
    ax.plot(angles_c, close(L7n), color=col2, linewidth=2)
    ax.fill(angles_c, close(L7n), color=col2, alpha=0.2, label=f"L7 {dir}")
    ax.set_xticks(angles)
    ax.set_xticklabels(top)
    ax.set_yticklabels([])
    plt.title(f"Attack {dir} fingerprint – {country}")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, fname), 
        bbox_inches="tight", 
        pad_inches=0.2, 
        dpi=300
    )
    plt.close(fig)
    # plt.show()
