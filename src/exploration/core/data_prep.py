import pandas as pd
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from .time_utils import (
    conv_iso_to_utc, conv_iso_to_local_with_daytype, conv_iso_to_local_with_daytimes
)


#########################################
##       VALUE DIST / ACT. FLUCT       ##
#########################################

def add_local_daytypes(df: pd.DataFrame, tzmap: dict) -> pd.DataFrame:
    """Adds columns for local time and weekday/weekend classification."""
    df = df.copy()
    df["dates"] = conv_iso_to_utc(df["dates"])
    converted = df.apply(
        lambda row: conv_iso_to_local_with_daytype(row["dates"], row["countries"], tzmap),
        axis=1
    )
    df["local_time"] = converted.apply(lambda val: val["local_time"])
    df["weekday"] = converted.apply(lambda val: val["weekday"])
    df["daytype"] = converted.apply(lambda val: val["daytype"])

    return df

def add_local_daytimes(df: pd.DataFrame, tzmap: dict) -> pd.DataFrame:
    """Adds local time, local hour, and daytime classification per country's timezone."""
    df = df.copy()
    df["timestamps"] = conv_iso_to_utc(df["timestamps"])

    converted = df.apply(
        lambda row: conv_iso_to_local_with_daytimes(row["timestamps"], row["regions"], tzmap),
        axis=1
    )

    df["local_time"] = converted.apply(lambda val: val["local_time"])
    df["local_hour"] = converted.apply(lambda val: val["local_hour"])
    df["daytime"] = converted.apply(lambda val: val["daytime"])

    return df

def add_fluctuation_metrics(
    df: pd.DataFrame,
    group_col: str = "regions",
    value_col: str = "values",
    type_col: str = "daytime"
) -> pd.DataFrame:
    """
    Compute per-country median activity across daytimes and fluctuation metrics.

    Returns:
        DataFrame with additional columns for each daytime:
        - range: max - min
        - std: standard deviation
        - ratio: max / min
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


#########################################
##             CLUSTERING              ##
#########################################

def normalize_per_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  df with columns: [countries, dates, values]
    Output: pivot table normalized per date (columns sum to 1)
    --> ALL COUNTRY SHARES SUM UP TO 1 PER DAY
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
    return df.groupby("countries")["values"].sum()

def preprocess_matrix(
        df: pd.DataFrame, 
        min_activity: float = 1e-6
    ) -> tuple[list[str], np.ndarray]:
    """
    Standardizes the input country × date matrix.
    Removes countries with extremely low activity.
    """    
    # Drop all-zero rows
    activity = df.sum(axis=1)
    keep = activity[activity > min_activity].index
    mat2 = df.loc[keep].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(mat2.values)

    return mat2.index.tolist(), X

def evaluate_kmeans_over_k(
    X: Union[np.ndarray, pd.DataFrame], 
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
    results = {
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
            silhouette_score(X, labels)
        )
        results["calinski"].append(
            calinski_harabasz_score(X, labels)
        )
        results["davies"].append(
            davies_bouldin_score(X, labels)
        )

    return pd.DataFrame(results)

def fit_final_kmeans(X: Union[np.ndarray, pd.DataFrame], k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    return labels

