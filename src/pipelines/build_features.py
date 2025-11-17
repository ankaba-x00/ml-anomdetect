#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime
from src.data.feature_engineering import (
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
##            BUILD HELPERS            ##
#########################################


def save_feature_matrix(country: str, X: pd.DataFrame, scaler):
    fpath = FEATURE_DIR / f"features_{country}.pkl"
    with open(fpath, "wb") as f:
        pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)

    spath = SCALER_DIR / f"scaler_{country}.pkl"
    with open(spath, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved {country} features and scaler.")


def build_all_countries():
    for c in COUNTRIES:
        print(f"\n==============================")
        print(f"  BUILDING COUNTRY = {c}")
        print(f"==============================")

        try:
            X, scaler = build_feature_matrix(c)
            #print(X.shape)
            #print(X.columns)
            #print("All floats:", set(X.apply(lambda col: col.apply(lambda x: isinstance(x, float)).all())))
            #print("Missing values in:", X.columns[X.isnull().any()].tolist())
            save_feature_matrix(c, X, scaler)
        except Exception as e:
            print(f"[ERROR] Could not build {c}: {e}")
            continue


if __name__ == "__main__":
    build_all_countries()
