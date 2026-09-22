#!/usr/bin/env python3
"""
Flattens nesting level of pkl files for specified key after fetching and 
preprocessing.

Usage:
    python -m app.src.data.processing.flatten [-k] [-o <FILE_NAME>] <all|FILE_KEY>
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from . import DSFILE_MAP


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def check_dsfiles_exist(file: str, folder: Path) -> Path:
    """Checks whether dataset file exists before proceeding."""
    file_path = folder / file
    if not file_path.is_file():
        raise FileNotFoundError(f"[ERROR] FileNotFound: {file}")
    return file_path

def conv_maxlayer_3(data: dict) -> pd.DataFrame:
    """
    Vectorized flattening of 3-layered data dictionary into dataframe. 
        Outer layer: regions
        Middle layer: dates
        Inner layer: timestamps, metric, values, ...
    """
    records = []
    for region, region_data in data.items():
        for date, details in region_data.items():
            if "timestamps" in details:
                ts = pd.to_datetime(details["timestamps"], errors="coerce")
                base = pd.DataFrame({
                    "regions": region,
                    "dates": date,
                    "timestamps": ts
                })
                for metric_name, metric_values in details.items():
                    if metric_name == "timestamps":
                        continue
                    if len(metric_values) != len(ts):
                        raise ValueError(
                            f"[ERROR] Length mismatch: metric '{metric_name}' in region={region}, date={date}"
                        )
                    base[metric_name] = pd.to_numeric(metric_values, errors="coerce")
                long_df = base.melt(
                    id_vars=["regions", "dates", "timestamps"],
                    var_name="metric",
                    value_name="_tmp_value"
                )
                long_df = long_df.rename(columns={"_tmp_value": "values"})
                records.append(long_df)
            else:
                countries = details.get("countries", [])
                values = details.get("values", [])
                ranks = details.get("ranks", [])
                if not ranks or len(ranks) != len(countries):
                    ranks = [np.nan] * len(countries)
                records.append(pd.DataFrame({
                    "regions": region,
                    "dates": date,
                    "countries": countries,
                    "values": values,
                    "ranks": ranks
                }))

    return pd.concat(records, ignore_index=True)

def conv_maxlayer_2(data: dict) -> pd.DataFrame:
    """
    Vectorized flattening of 2-layered data dictionary into dataframe. 
        Outer layer: dates
        Inner layer: countries, values, types, ...
    """
    records = []
    for date, details in data.items():
        df = pd.DataFrame(details)
        df["dates"] = date
        records.append(df)
        
    return pd.concat(records, ignore_index=True)

def _detect_nesting_level(data: dict) -> int:
    """Detects required nesting layer number for flattening dataset."""
    if not data:
        return 1
    
    first_val = next(iter(data.values()))
    if not first_val:
        return 2
    if isinstance(first_val, dict):
        inner_val = next(iter(first_val.values()), None)
        if isinstance(inner_val, dict):
            return 3
        else:
            return 2
    else:
        return 1

def _load_dsfile(key: str, folder: Path) -> dict:
    """Loads dataset file after sanity check."""
    path = check_dsfiles_exist(f"{key}.pkl", folder)
    with open(path, "rb") as f:
        data: dict = pickle.load(f)
    return data

def conv_pkltodf(
    key: str, 
    folder: Path, 
    data: dict | None = None
) -> pd.DataFrame:
    """Converts data dictionary or pkl file to internal dataframe."""
    if data is None:
        data = _load_dsfile(key, folder)

    max_layer = _detect_nesting_level(data)
    if max_layer == 3:
        return conv_maxlayer_3(data)
    elif max_layer == 2:
        return conv_maxlayer_2(data)
    else:
        raise ValueError("[ERROR] Data dict layering not valid. Aborting dataframe conversion!")

def flatten_single(key: str, file_name: str, data: dict | None = None) -> None:
    """Runs pipeline for flattening single dataset."""
    if DSFILE_MAP[key][0]:
        print(f"[INFO] Processing {key}...")
    
        df = conv_pkltodf(key, PROCESSED_DIR, data)

        file_path = PROCESSED_DIR / file_name
        df.to_hdf(file_path, key=key, format="table", complevel=9, index=True, mode="a")
        print(f"[OK] {key} saved to {file_path}")

        print(f"[OK] {key} successfully flattened")

def flatten_all(file_name: str) -> None:
    """Runs pipeline for flattening all datasets."""
    for key in DSFILE_MAP.keys():
        if DSFILE_MAP[key][0]:
            flatten_single(key, file_name)

    print("[DONE] All datasets processed!")


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Flatten pickled dataset files and return dataframe"
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available dataset keys and exit"
    )

    parser.add_argument(
        "-o", "--out",
        nargs="?",
        default=None,
        help="output file [default: <file_key>.h5]"
    )

    parser.add_argument(
        "file_key",
        help="<all|aibots_crawlers_time, anomalies, ...> file key to process"
    )


    args = parser.parse_args()

    time_dsfiles =  [k for k, v in DSFILE_MAP.items() if v[0]]
    if args.keys:
        print("Available file keys:")
        for key in time_dsfiles:
            print(f"\t- {key}")
        exit(0)

    if args.out is None:
        args.out = f"{args.file_key.lower()}.h5"

    if args.file_key not in ["all", *time_dsfiles]:
        print(f"[Error] file_key {args.file_key} cannot be processed\n")
        parser.print_help()
        exit(1)

    if args.file_key.lower() == 'all':
        flatten_all(args.out)
    else:
        flatten_single(args.file_key.lower(), args.out)