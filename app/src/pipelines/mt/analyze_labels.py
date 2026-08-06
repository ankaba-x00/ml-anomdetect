from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from app.src.data.feature_engineering import (
    COUNTRIES, 
    build_country_dataframe,
    build_supervised_feature_matrix
)
from app.src.data.attack_labelling import (
    ATTACK_LABELS, 
    ATTACK_TO_ID, 
    ID_TO_ATTACK
)
from app.src.data.split import timeseries_seq_split


#########################################
##               PARAMS                ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
FEATURE_DIR = PROJECT_ROOT / "datasets" / "featured"
OUT_DIR = PROJECT_ROOT.parent / "results" / "labelled"

LABEL_META = {
    0: ("normal", "60–80%"),
    1: ("udp_ampl", "5–15%"),
    2: ("tcp_syn", "3–10%"),
    3: ("icmp_flood", "<2%"),
    4: ("gre_flood", "<2%"),
    5: ("http_flood", "2–8%"),
    6: ("multi_vec", "1–3%"),
    7: ("stealth", "2–6%"),
}


#########################################
##                HELPERS              ##
#########################################

def _check_interim_exists(f_path: Path) -> bool:
    return f_path.is_file()


def _gen_interim_matrix(country: str, f_path: Path) -> None:
    df = build_country_dataframe(country, True, debug=True)
    df.to_parquet(f_path, compression="zstd")


#########################################
##                MAIN                 ##
#########################################

def num_analysis(country: str, df: pd.DataFrame) -> None:
    """Performs numerical analysis of attack label scheme."""
    print(f"==============================")
    print(f"    NUM ANALYSIS ({country})")
    print(f"==============================")

    counts = df["attack_label"].value_coun
    total = df.shape[0]

    print("\nATTACK TARGET HEURISTIC")
    print(f"    LABEL    | COUNT | DIST | TARGET")
    for i, (name, target) in LABEL_META.items():
        c = counts.get(i, 0)
        pct = math.ceil(100 * c / total)
        print(f"{i} {name:<10} | {c:<6} | {pct:>3}% | {target}")
    
    print(f"Normal traffic %: {100*counts[0]/total:.1f}%")
    print(f"Attack traffic %: {100*(1 - counts[0]/total):.1f}%")

    print("/nOVERLAP TO JUDGE WHETHER THERE ARE CONDITIONS FOR MULTI-VECTOR (>5% MV under-detected, <1% MV should be rare)")
    overlap = (
        (df[df["attack_label"] == 2] > 0.5) &
        (df[df["attack_label"] == 5] > 0.5)
    ).mean()
    print("TCP+HTTP overlap fraction:", overlap)


    print("\nSANITY CHECK STEALTH SHOULD APPEAR IN LONG RUNS")
    labels = df["attack_label"]
    stealth = labels == ATTACK_TO_ID["stealth_scan"]
    groups = (stealth != stealth.shift()).cumsum()
    lengths = stealth.groupby(groups).sum()
    print(lengths.describe())
    print(f"Stealth median duration (h): {lengths.median():.2f}")

    print("\nSANITY CHECK MULTI-VECTOR SHOULD APPEAR NEAR OVERLAPS")
    mv = labels == ATTACK_TO_ID["multi_vector"]
    neighbors = (
        labels.shift(1).isin([
            ATTACK_TO_ID["tcp_syn_flood"],
            ATTACK_TO_ID["http_flood"],
        ])
        | labels.shift(-1).isin([
            ATTACK_TO_ID["tcp_syn_flood"],
            ATTACK_TO_ID["http_flood"],
        ])
    )
    print("Isolated MV %:", (~neighbors[mv]).mean())

    print("\nREMOVING TEMPORAL LOGIC SHOULD NOT RADICALLY CHANGE PROPORTIONS")
    print(
        df["uncorr_attack_label"].value_counts(normalize=True) -
        df["attack_label"].value_counts(normalize=True)
    )

    print("\nTEMPORAL STABILITY BELOW 85-90% MEANS TOO NOISY")
    stability = (df["attack_label"] == df["attack_label"].shift()).mean()
    print(f"Label temporal stability: {stability:.2%}")


def num_analysis(country: str, df: pd.DataFrame) -> None:
    """
    Numerical diagnostics for attack label sanity.
    Answers:
      A) Are distributions plausible?
      B) Are labels temporally coherent?
      C) Do causal relationships exist?
    """

    print(f"\n{'='*60}")
    print(f"NUMERICAL LABEL DIAGNOSTICS — {country}")
    print(f"{'='*60}")

    labels = df["attack_label"]
    uncorr = df.get("uncorr_attack_label")
    n = len(labels)

    # --------------------------------------------------
    # A. Distribution sanity
    # --------------------------------------------------
    print("\n[A] DISTRIBUTION SANITY")

    counts = labels.value_counts().sort_index()
    freqs = counts / n

    entropy = -(freqs * np.log(freqs + 1e-9)).sum()

    print(f"{'ID':<3} {'LABEL':<12} {'%':>6}  TARGET")
    for i, (name, target) in LABEL_META.items():
        pct = 100 * freqs.get(i, 0)
        print(f"{i:<3} {name:<12} {pct:>5.2f}%  {target}")

    print(f"\nLabel entropy: {entropy:.3f}")
    print("Interpretation:")
    print("  < 1.0  → collapsed / trivial")
    print("  1.0–1.6 → reasonable structure")
    print("  > 1.8  → noisy / over-fragmented")

    # --------------------------------------------------
    # B. Temporal coherence
    # --------------------------------------------------
    print("\n[B] TEMPORAL COHERENCE")

    def run_lengths(mask: pd.Series) -> pd.Series:
        groups = (mask != mask.shift()).cumsum()
        return mask.groupby(groups).sum()

    flicker = (labels != labels.shift()).mean()
    print(f"Label flicker rate: {flicker:.2%} (target < 15–20%)")

    print("\nRun-length summary (median / p90):")
    for i, name in ATTACK_TO_ID.items():
        mask = labels == name
        if not mask.any():
            continue
        runs = run_lengths(mask)
        runs = runs[runs > 0]
        if len(runs) == 0:
            continue
        print(
            f"{name:<15} "
            f"median={runs.median():>4.0f}h  "
            f"p90={runs.quantile(0.9):>4.0f}h"
        )

    # --------------------------------------------------
    # C. Causal adjacency & correction impact
    # --------------------------------------------------
    print("\n[C] CAUSAL / TEMPORAL CONSISTENCY")

    mv = ATTACK_TO_ID["multi_vector"]
    tcp = ATTACK_TO_ID["tcp_syn_flood"]
    http = ATTACK_TO_ID["http_flood"]

    mv_mask = labels == mv
    mv_neighbors = (
        labels.shift(1).isin([tcp, http]) |
        labels.shift(-1).isin([tcp, http])
    )

    if mv_mask.any():
        isolated_mv = (~mv_neighbors[mv_mask]).mean()
        print(f"Isolated multi-vector rate: {isolated_mv:.2%}")
        print("  target: < 30% (preferably < 20%)")

    if uncorr is not None:
        print("\nUncorrected → temporal promotion matrix:")
        promo = pd.crosstab(
            uncorr.map(ID_TO_ATTACK),
            labels.map(ID_TO_ATTACK),
            normalize="index"
        )
        print(promo.round(2))

    print("\nInterpretation checklist:")
    print("✔ multi_vector adjacent to TCP/HTTP")
    print("✔ stealth long-duration, low flicker")
    print("✖ normal → attack frequent = thresholds too loose")



def vis_analysis(country: str, df: pd.DataFrame, show_plots: bool) -> None:
    """Generates visual analysis plots of attack label scheme."""
    print(f"==============================")
    print(f"    VIS ANALYSIS ({country})")
    print(f"==============================")
    
    out_path = OUT_DIR / country
    out_path.mkdir(parents=True, exist_ok=True)

    # df["attack_label"].plot(figsize=(15, 2), title="Attack Labels Over Time")
    
    # df[ATTACK_LABELS[1:]].rolling(3).mean().plot(
    #     subplots=True, figsize=(15, 8), alpha=0.7
    # )

    # pd.crosstab(
    #     df["uncorr_attack_label"],
    #     df["attack_label"],
    #     normalize="index"
    # )

    from app.src.ml.analysis.analysis_labels import plot_l3_l7_scatter_by_attack, plot_timeseries_with_attack_labels

    plot_l3_l7_scatter_by_attack(
        country,
        df,
        OUT_DIR / country,
        f"{country}_l3_l7_by_attack.png"
    )

    plot_timeseries_with_attack_labels(
        country,
        df,
        OUT_DIR / country,
        f"{country}_attack_timeseries.png"
    )

    print("\nSCORE HISTOGRAMS STEALTH SHOULD HAVE LOW CONFIDENCE, MULTVECTOR MID-CONFIDENCE")
    plt.hist(
        df.loc[df["attack_label"] == ATTACK_TO_ID["stealth_scan"]], 
        bins=50, 
        alpha=0.5, 
        label="stealth"
    )
    plt.hist(
        df.loc[df["attack_label"] == ATTACK_TO_ID["multi_vector"]], 
        bins=50, 
        alpha=0.5, 
        label="multi"
    )
    plt.legend()
    if show_plots: plt.show()
    plt.savefig(out_path / "hist_stealth_multivec.png")


def print_labeldist(country: str, df: pd.DataFrame) -> None:
    """Prints label distribution for different train-val-test split ratios."""
    _, _, _, _, ya_s, _, _ = build_supervised_feature_matrix(country, df)
    ya = ya_s.values.astype(np.int64)

    ratios = [
        #(70, 20), 
        #(70, 15), 
        #(75, 20), 
        (75, 15), 
        #(80, 15), 
        #(80, 10), 
        #(90, 5), 
        (100, 0)
    ]
    print(f"==============================")
    print(f"  LABEL DISTRIBUTION ({country})")
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
            print(f"\t{split_name:<{name_width}}  " + " | ".join(row))


#########################################
##                 RUN                 ##
#########################################

def analyze_country(
    country: str, 
    pre_run: bool,
    vis: bool,
    num: bool,
    dist: bool,
    show_plots: bool
) -> None:
    """Runs full numerical and visual analysis pipeline of attack labelling 
    scheme for a given country."""
    print(f"[INFO] Analyzing {country}...")

    f_path = FEATURE_DIR / f"_interim_{country}_superM.parquet"

    if not _check_interim_exists(f_path) or pre_run:
        _gen_interim_matrix(country, f_path)

    print(f"[INFO] Reading base DF for country={country}")
    df = pd.read_parquet(f_path)
    
    if vis: vis_analysis(country, df, show_plots)
    if num: num_analysis(country, df)
    if dist: print_labeldist(country, df)

    print(f"\n[OK] Analysis for {country} completed!")


def analyze_all(
    pre_run: bool,
    vis: bool,
    num: bool,
    dist: bool,
    show_plots: bool
) -> None:
    """Runs full numerical and visual analysis pipeline of attack labelling 
    scheme for all specified models."""
    print(f"\n[INFO] Analysis of all country attack labels starting...")

    for c in COUNTRIES:
        try:
            analyze_country(c, pre_run, vis, num, dist, show_plots)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")

    print(f"\n[DONE] Analysis of all country attack labels completed!")


def purge():
    n = 0
    for file in FEATURE_DIR.glob("_interim_*.parquet"):
        if file.is_file():
            file.unlink()
            n += 1
    print(f"[OK] Purged {n} interim parquet files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyzes attack labelling schema."
    )

    parser.add_argument(
        "-pre", "--pre-run",
        action="store_true",
        help="generate interim matrices to speed up analysis"
    )

    parser.add_argument(
        "-V", "--vis",
        action="store_true",
        help="run visual analysis"
    )

    parser.add_argument(
        "-N", "--num",
        action="store_true",
        help="run numerical analysis"
    )

    parser.add_argument(
        "-D", "--dist",
        action="store_true",
        help="print label distribution for different data split ratios"
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="show plots interactively when generated"
    )

    parser.add_argument(
        "--purge",
        action="store_true",
        help="removes all redundant intermediate helper files"
    )

    parser.add_argument(
        "target",
        help="<COUNTRY|all|none> e.g. 'US' to analyse US model, or 'all' to evaluate all country models"
    )

    args = parser.parse_args()

    target = args.target

    if args.purge:
        purge()

    if target.lower() == "all":
        analyze_all(
            args.pre_run, 
            args.vis,
            args.num,
            args.dist,
            args.show
        )
    else:
        analyze_country(
            target.upper(), 
            args.pre_run,
            args.vis,
            args.num,
            args.dist,
            args.show
        )
