import pandas as pd

from app.src.data.feature_engineering import (
    build_country_dataframe, 
    build_feature_matrix, 
    build_supervised_feature_matrix
)


def build_features(
    country: str, 
    newdata: dict, 
    supervised: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, int, dict[str, int]] | tuple[pd.DataFrame, pd.DataFrame, int, dict[str, int]]:
    """Fetches and processes raw dataset and returns feature matrix ready for inference."""
    
    if supervised:
        raw_df = build_country_dataframe(
            country=country, 
            from_storage=False, 
            data=newdata, 
            attack_label=True
        )
        X_cont, X_cat, y_l3, y_l7, y_type, num_cont, cat_dims = build_supervised_feature_matrix(country, df=raw_df)
        print(f"[INFO] Supervised feature matrix for {country} ready")
        return X_cont, X_cat, y_l3, y_l7, y_type, num_cont, cat_dims
    
    raw_df = build_country_dataframe(country, from_storage=False, data=newdata)
    X_cont, X_cat, num_cont, cat_dims = build_feature_matrix(country, df=raw_df)
    print(f"[INFO] Feature matrix for {country} ready")
    return X_cont, X_cat, num_cont, cat_dims