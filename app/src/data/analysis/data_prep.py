import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from .time_utils import (
    conv_iso_to_utc, 
    conv_iso_to_local_with_daytype, 
    conv_iso_to_local_with_daytimes
)


def add_local_daytypes(
    df: pd.DataFrame, 
    tzmap: dict
) -> pd.DataFrame:
    """Adds columns for local time, weekday and daytype classification."""
    df = df.copy()

    df["dates"] = conv_iso_to_utc(df["dates"])

    converted = df.apply(
        lambda row: conv_iso_to_local_with_daytype(row["dates"], row["countries"], tzmap),
        axis=1
    ).apply(pd.Series)

    df_wday: pd.DataFrame = pd.concat([df, converted], axis=1)

    return df_wday

def add_local_daytimes(
    df: pd.DataFrame, 
    tzmap: dict
) -> pd.DataFrame:
    """
    Adds local time, local hour, and daytime classification per country's 
    timezone.
    """
    df = df.copy()
    df["timestamps"] = conv_iso_to_utc(df["timestamps"])

    converted = df.apply(
        lambda row: conv_iso_to_local_with_daytimes(row["timestamps"], row["regions"], tzmap),
        axis=1
    ).apply(pd.Series)


    df_wtimes: pd.DataFrame = pd.concat([df, converted], axis=1)

    return df_wtimes


def add_fluctuation_metrics(
    df: pd.DataFrame,
    group_col: str = "regions",
    value_col: str = "values",
    type_col: str = "daytime"
) -> pd.DataFrame:
    """
    Compute per-country median activity across daytimes and fluctuation 
    metrics.
    """
    country_daytime_medians = (
        df.groupby([group_col, type_col])[value_col]
        .median()
        .unstack()
    )
    country_daytime_medians["range"] = (
        country_daytime_medians.max(axis=1) - country_daytime_medians.min(axis=1)
    )
    country_daytime_medians["std"] = country_daytime_medians.std(axis=1)
    country_daytime_medians["ratio"] = (
        country_daytime_medians.max(axis=1) / country_daytime_medians.min(axis=1)
    )

    return country_daytime_medians


def normalize_per_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts df columns [countries, dates, values] to pivot table normalized 
    per date (all country shares sum up to 1 per day).
    """
    pivot = df.pivot(index="countries", columns="dates", values="values").fillna(0.0)
    norm = pivot.copy()
    for d in pivot.columns:
        colsum = pivot[d].sum()
        if colsum > 0:
            norm[d] = pivot[d] / colsum
        else:
            norm[d] = 0.0

    return norm

def aggregate_directional(df: pd.DataFrame) -> pd.Series:
    """Sums up value column."""
    return df.groupby("countries")["values"].sum()

def preprocess_matrix(
    df: pd.DataFrame, 
    min_activity: float = 1e-6
) -> tuple[list[str], np.ndarray]:
    """
    Standardizes input country × date matrix and removes countries with 
    extremely low activity.
    """
    activity = df.sum(axis=1)
    keep = activity[activity > min_activity].index
    mat2 = df.loc[keep].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(mat2.values)

    return mat2.index.tolist(), X

def evaluate_kmeans_over_k(
    X: np.ndarray | pd.DataFrame, 
    k_min: int = 2, 
    k_max: int = 15
) -> pd.DataFrame:
    """
    Computes:
        - SSE (inertia)
        - Silhouette Score
        - Calinski–Harabasz
        - Davies–Bouldin
    for k = k_min … k_max.
    """
    results : dict[str, list[int | float]] = {
        "k": [],
        "SSE": [],
        "silhouette": [],
        "calinski": [],
        "davies": []
    }

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        # centers = km.cluster_centers_
        labels = km.fit_predict(X)

        results["k"].append(k)
        results["SSE"].append(km.inertia_)
        results["silhouette"].append(
            float(silhouette_score(X, labels))
        )
        results["calinski"].append(
            calinski_harabasz_score(X, labels)
        )
        results["davies"].append(
            davies_bouldin_score(X, labels)
        )

    return pd.DataFrame(results)

def fit_final_kmeans(X: np.ndarray | pd.DataFrame, k: int) -> npt.NDArray:
    """Performs KMeans clustering."""
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels: npt.NDArray = km.fit_predict(X)
    return labels