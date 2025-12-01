#!/usr/bin/env python3
"""
Builds and saves feature matrix for all countries:
- loads processed time-series data
- constructs merged feature dataframe for each country
- splits into:
        a) continuous features (scaled)
        b) categorical index features
- fits and saves a RobustScaler per country
- saves feature matrices for use during training and inference

Outputs:
    feature matrix : datasets/featured/features_<COUNTRY_CODE>.pkl
    scaler : datasets/scalers/scaler_<COUNTRY_CODE>.pkl

Usage:
    python -m src.pipelines.build_features
"""

from pathlib import Path
import pandas as pd
import pickle
from sklearn.base import TransformerMixin
from app.src.data.feature_engineering import (
    COUNTRIES,
    build_feature_matrix,
)


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
FEATURE_DIR = PROJECT_ROOT / "datasets" / "featured"
SCALER_DIR = PROJECT_ROOT / "datasets" / "scalers"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##               HELPER                ##
#########################################

def save_feature_matrix(country: str, X_cont: pd.DataFrame, X_cat: pd.DataFrame, num_cont: int, cat_dims: list[int], scaler: TransformerMixin):
    """
    Save:
      - continuous scaled features (float32)
      - categorical index features (int)
      - metadata: num_cont and cat_dims
      - scaler for validation and testing
    """
    fpath = FEATURE_DIR / f"features_{country}.pkl"
    with open(fpath, "wb") as f:
        pickle.dump(
            {
                "continuous": X_cont,
                "categorical": X_cat,
                "num_cont": num_cont,
                "cat_dims": cat_dims,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    spath = SCALER_DIR / f"scaler_{country}.pkl"
    with open(spath, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved {country} features and scaler.")


#########################################
##                 RUN                 ##
#########################################

def build_all_countries():
    for c in COUNTRIES:
        print(f"\n==============================")
        print(f"  BUILDING COUNTRY = {c}")
        print(f"==============================")

        try:
            X_cont, X_cat, num_cont, cat_dims, scaler = build_feature_matrix(c)
            save_feature_matrix(c, X_cont, X_cat, num_cont, cat_dims, scaler)
        except Exception as e:
            print(f"[ERROR] Could not build {c}: {e}")
            continue
    print("[DONE] All feature matrices and scaler build!")

if __name__ == "__main__":
    build_all_countries()
