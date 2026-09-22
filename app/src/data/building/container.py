from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True, frozen=True)
class ProcessedRegionTimeseries:
    """
    Immutable container storing preprocessed timeseries data for a single country.
    Used as input object for feature engineering phase.
    """

    l3o: pd.DataFrame
    l3t: pd.DataFrame
    l7: pd.DataFrame
    http: pd.DataFrame
    http_auto: pd.DataFrame
    http_human: pd.DataFrame
    netflow: pd.DataFrame
    bots: pd.DataFrame
    ai: pd.DataFrame
    l3_bitrate: pd.DataFrame
    l3_duration: pd.DataFrame
    protocol: pd.DataFrame


@dataclass(slots=True, frozen=True)
class FeatureMatrix:
    """
    Immutable container holding feature matrices and its metadata.
    """

    X_cont: pd.DataFrame
    X_cat: pd.DataFrame
    num_cont: int
    cat_dims: dict[str, int]
    y_l3: pd.Series | None = None
    y_l7: pd.Series | None = None
    y_at: pd.Series | None = None