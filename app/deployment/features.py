from . import RegionTimeseriesFetchResult
from app.src.data.building import FeatureMatrix
from app.src.data.building.feature_engineering import (
    build_country_dataframe, 
    build_feature_matrix, 
    build_supervised_feature_matrix
)


def build_features(
    country: str, 
    newdata: RegionTimeseriesFetchResult, 
    supervised: bool = False
) -> FeatureMatrix:
    """Fetches and processes raw dataset and returns feature matrix ready for inference."""
    
    if supervised:
        raw_df = build_country_dataframe(
            country, 
            newdata, 
            attack_label=True
        )
        fmatrix = build_supervised_feature_matrix(country, df=raw_df)
        print(f"[INFO] Supervised feature matrix for {country} ready")
    else:
        raw_df = build_country_dataframe(country, newdata)
        fmatrix = build_feature_matrix(country, df=raw_df)
        print(f"[INFO] Feature matrix for {country} ready")

    return fmatrix