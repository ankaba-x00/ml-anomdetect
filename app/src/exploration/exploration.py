#!/usr/bin/env python3
"""
Visually and numerically explores datasets as pulled from Cloudflare
- value distributions : boxplots ts, daytime, daytype resolved
- activity fluctuations : heatmaps daytime, daytype resolved
- activity fluctuations : barplots top/tail countries daytime, daytype resolved
- anomalies : heatmap countries x dates
- attack profiles : heatmaps log10 normalized daily shares per country L3, L7 resolved
- attack profiles : lineplots cluster eval and scatter plots PCA on specified final_k cluster L3, L7 resolved
- attack profiles : radar plots with attack fingerprint for specified countries origin, target resolved

Outputs:
    png files : results/exploration/<figure_type>.png

Usage: 
    python -m src.exploration.exploration
"""

from typing import Optional
import pandas as pd
from app.src.data import conv_pkltodf
from .core import (
    timezones, add_local_daytypes, add_local_daytimes, add_fluctuation_metrics, normalize_per_date, aggregate_directional, preprocess_matrix, evaluate_kmeans_over_k, fit_final_kmeans, boxplot_valdist, boxplot_valdist_daytypes, boxplot_valdist_daytimes, heatmap_activity_fluctuations, heatmap_anomalies, heatmap_log10_attack_profile, barplot_activity_fluctuations, barplot_top_attackers, lineplot_clustering_eval_curves, pca_clusterplot, radarchart_attack_fingerprint
)


#########################################
##        VALUE DISTRIBUTIONS          ##
#########################################

def gen_valdist(typ: str, in_folder: str, out_folder: str, show: bool):
    if typ == "traffic":
        name = "NetFlow traffic"
        shortname = "traffic"
        yname = "traffic (bytes/day)"
        color1 = "#5EA7E3"
        colpal1 = ["#7FB7E4", "#212B9E"]
        colpal2 = ["#A9CCE8", "#7FB7E4", "#6076D1", "#212B9E", "#060F75"]
    elif typ == "httpreq":
        name = "HTTP request"
        shortname = "requests"
        yname = "requests (intensity/day)"
        color1 = "#8DB76F"
        colpal1 = ["#A0C684", "#31800C"]
        colpal2 = ["#ADE287", "#84AF64", "#537737", "#31800C", "#244D05"]
    else:
        raise ValueError("[ERROR] Filename not recognized, aborting explorarion run!")
    
    df = conv_pkltodf(f"{typ}", in_folder)
    boxplot_valdist(
        df,
        col=color1,
        title=f"{name} volume distribution by country",
        xlabel="Most active countries (acc. to median)",
        ylabel=f"Absolute {yname}",
        folder=out_folder,
        fname=f"{typ}_dist.png",
        show=show
    )

    df_with_days = add_local_daytypes(df, timezones)
    boxplot_valdist_daytypes(
        df_with_days,
        title=f"{name} volume distribution by country: weekdays vs weekends",
        xlabel="Most active countries (acc. to median)",
        ylabel=f"Absolute {yname}",
        palette={"Weekday": colpal1[0], "Weekend": colpal1[1]},
        folder=out_folder,
        fname=f"{typ}_dist_daytypes.png",
        show=show
    )

    df = conv_pkltodf(f"{typ}_time", in_folder)
    df_countries = df.loc[df["regions"] != "worldwide"]
    df_with_times = add_local_daytimes(df_countries, timezones)
    boxplot_valdist_daytimes(
        df_with_times,
        title=f"{name} volume distribution by country: daytimes",
        xlabel="Most active countries (acc. to median)",
        ylabel=f"Relative {shortname} per hour to busiest hour of day",
        palette={"Deep night": colpal2[0], "Morning": colpal2[1], "Business hours": colpal2[2], "Evening": colpal2[3], "Early night": colpal2[4]},
        folder=out_folder,
        fname=f"{typ}_dist_daytimes.png",
        show=show
    )


#########################################
##        ACTIVITY FLUCTUATIONS        ##
#########################################

def gen_actfluct(typ: str, in_folder: str, out_folder: str, show: bool):
    if typ == "traffic":
        name = "traffic"
        color = "#5EA7E3"
        colpal = "Blues"
    elif typ == "httpreq":
        name = "request"
        color = "#8DB76F"
        colpal = "Greens"
    else:
        raise ValueError("[ERROR] Filename not recognized, aborting explorarion run!")

    for category in ["daytype", "daytime"]:
        if category == "daytype":
            df = conv_pkltodf(f"{typ}", in_folder)
            mod_df = add_local_daytypes(df, timezones)
            mod_df_medians = add_fluctuation_metrics(mod_df, group_col="countries", type_col="daytype")
            values = mod_df_medians.loc[:, "Weekday":"Weekend"]
        else:
            df = conv_pkltodf(f"{typ}_time", in_folder)
            df_countries = df.loc[df["regions"] != "worldwide"]
            mod_df = add_local_daytimes(df_countries, timezones)
            mod_df_medians = add_fluctuation_metrics(mod_df)
            values = mod_df_medians.loc[:, "Business hours":"Morning"]

        barplot_activity_fluctuations(
            mod_df_medians, 
            title=f"Countries with strongest {category} activity fluctuations for {name} intensity",
            col=color,
            folder=out_folder,
            fname=f"{typ}_{category}_top30_fluctuation.png",
            show=show
        )

        barplot_activity_fluctuations(
            mod_df_medians, 
            head=False, 
            title=f"Countries with lowest {category} activity fluctuations for {name} intensity",
            col=color,
            folder=out_folder,
            fname=f"{typ}_{category}_tail30_fluctuation.png",
            show=show
        )

        heatmap_activity_fluctuations(
            mod_df_medians,
            vmin=float(values.min(axis=1).min()),
            vmax=float(values.max(axis=1).max()),
            cmap=colpal,
            title=f"Median {name} activity fluctuations by {category}",
            folder=out_folder,
            fname=f"{typ}_{category}_fluctuations",
            show=show
        )


#########################################
##              ANOMALIES              ##
#########################################

def gen_anomheatmap(in_folder: str, out_folder: str, show: bool = False):
    df = conv_pkltodf("anomalies", in_folder)
    heatmap_anomalies(
        df,
        title="Detected Cloudflare anomalies",
        xlabel="Date",
        ylabel="Location",
        folder=out_folder,
        fname="anomalies_heatmap.png",
        show=show
)


#########################################
##           ATTACK PROFILES           ##
#########################################

def _load_data_worldwide(file: str, folder: str, col: str = "worldwide") -> pd.DataFrame:
    df = conv_pkltodf(file, folder)
    df = df[df["regions"] == col].copy()
    df["values"] = df["values"].astype(float)
    df["dates"] = pd.to_datetime(df["dates"])
    return df

def _ask_for_k():
    while True:
        user_in = input(">>> How many clusters do you choose? [int 2–15] ")
        try:
            k = int(user_in)
            if 2 <= k <= 15:
                return k
            else:
                print("[INFO] Enter a number between 2 and 15.")
        except ValueError:
            print("[ERROR] Invalid input — please enter an integer.")

def _run_kmeans_analysis(
        df: pd.DataFrame, 
        folder: str,
        name: str,
        col: str,
        k_min: int = 2, 
        k_max: int = 15, 
        final_k: Optional[int] = None,
        interactive: bool = False,
        show: bool = False
    ) -> int:
    countries, X = preprocess_matrix(df)
    eval_df = evaluate_kmeans_over_k(X, k_min=k_min, k_max=k_max)
    lineplot_clustering_eval_curves(
        eval_df,
        col=col, 
        folder=folder, 
        fname=f"{name}_cluster_eval_curves",
        show=show
    )
    
    if final_k is None:
        best_k = eval_df["silhouette"].idxmax()
        sugg_k = eval_df.loc[best_k, "k"]
        print(f"[INFO] Best k acc to Silhouette: {sugg_k}")
        if interactive:
            final_k = _ask_for_k()
        else:
            if name == "l3_origin":
                final_k = 4
            else:
                final_k = 5

    labels = fit_final_kmeans(X, final_k)
    pca_clusterplot(
        X, 
        labels, 
        countries,
        k=final_k, 
        name=name, 
        folder=folder, 
        fname=f"{name}_finalk_cluster_pca",
        show=show
    )

    return final_k

def gen_attackprofile(
        in_folder: str, 
        out_folder: str, 
        interactive: str = False,
        show: bool = False
    ):
    for key in ["l3_origin", "l7_origin"]:
        if key == "l3_origin":
            col = "#01205f"
            colpal = "Blues_r"
        else:
            col = "#610200"
            colpal = "Reds_r"
        df = _load_data_worldwide(key, in_folder)
        barplot_top_attackers(
            df, 
            top_n=30, 
            name=key, 
            palette=colpal,
            folder=out_folder, 
            fname=f"{key[:2]}_top_attackers",
            show=show
        )
        df_norm = normalize_per_date(df)
        
        final_k = _run_kmeans_analysis(df_norm, out_folder, key, col, interactive=interactive, show=show)

        heatmap_log10_attack_profile(
            df_norm, 
            name=key,
            folder=out_folder, 
            fname=f"{key[:2]}_log10_attack_profile",
            show=show
        )
       
    for c in ["US", "DE", "CN", "BR", "IN"]:
        for dir in ["origin", "target"]:
            l3 = aggregate_directional(_load_data_worldwide(f"l3_{dir}", in_folder, col=c))
            l7 = aggregate_directional(_load_data_worldwide(f"l7_{dir}", in_folder, col=c))
            radarchart_attack_fingerprint(
                l3,
                l7,
                country=c,
                dir=dir,
                col1= "#133a88",
                col2= "#971613",
                folder=out_folder,
                fname=f"fingerprint_{c}_{dir}.png",
                show=show
            )


#########################################
##                 RUN                 ##
#########################################

def run_exploration(in_folder: str, out_folder: str , show: bool = False):
    for typ in ["traffic", "httpreq"]:
        gen_valdist(typ, in_folder, out_folder, show)
        gen_actfluct(typ, in_folder, out_folder, show)
    gen_anomheatmap(in_folder, out_folder, show)
    gen_attackprofile(in_folder, out_folder, interactive=False, show=show)


if __name__ == "__main__":
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parents[2]
    PROJECT_ROOT = BASE_DIR.parent
    INDIR = BASE_DIR / "datasets" / "processed"
    OUTDIR = PROJECT_ROOT / "results" / "exploration" / "y2024-2025"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    SHOW = False

    print("[INFO] Starting data exploration...")
    result = run_exploration(INDIR, OUTDIR, SHOW)
    print("[DONE] Data exploration completed!")
