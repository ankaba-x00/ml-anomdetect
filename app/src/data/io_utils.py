#!/usr/bin/env python3

import os, pickle
import pandas as pd
import numpy as np


#########################################
##                PARAMS               ##
#########################################

DSFILE_MAP = {
    "aibots_crawlers_time": "aibots_crawlers_time.pkl",
    "anomalies": "anomalies.pkl",
    "bots_time": "bots_time.pkl",
    "httpreq_automated_time": "httpreq_automated_time.pkl",
    "httpreq_human_time": "httpreq_human_time.pkl",
    "httpreq_time": "httpreq_time.pkl",
    "httpreq": "httpreq.pkl",
    "iq_bandwidth_time": "iq_bandwidth_time.pkl",
    "iq_dns_time": "iq_dns_time.pkl",
    "iq_latency_time": "iq_latency_time.pkl",
    "l3_origin_bitrate_time": "l3_origin_bitrate_time.pkl",
    "l3_origin_duration_time": "l3_origin_duration_time.pkl",
    "l3_origin_protocol_time": "l3_origin_protocol_time.pkl",
    "l3_origin_time": "l3_origin_time.pkl",
    "l3_origin": "l3_origin.pkl",
    "l3_target": "l3_target.pkl",
    "l3_target_bitrate_time": "l3_target_bitrate_time.pkl",
    "l3_target_duration_time": "l3_target_duration_time.pkl",
    "l3_target_protocol_time": "l3_target_protocol_time.pkl",
    "l3_target_time": "l3_target_time.pkl",
    "l7_mitigations_time": "l7_mitigations_time.pkl",
    "l7_origin": "l7_origin.pkl",
    "l7_target": "l7_target.pkl",
    "l7_time": "l7_time.pkl",
    "traffic_time": "traffic_time.pkl",
    "traffic": "traffic.pkl",
}


#########################################
##            SANITY CHECK             ##
#########################################

def check_dsfiles_exist(file: str, folder: str) -> str:
    path = os.path.join(folder, file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] FileNotFound: {file}")
    return path


#########################################
##             FLATTENING              ##
#########################################

def conv_maxlayer_3(data: dict) -> pd.DataFrame:
    """
    Vectorized flattening of 3-layered data dictionary into dataframe. 
        Outer layer: regions -> regions
        Middle layer: dates -> dates
        Inner layer: timestamps, ..., values -> timestamps, metric, values
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
        Outer layer: dates -> dates
        Inner layer: countries, values, types, ... -> countries, values, types, ...
    """

    records = []
    for date, details in data.items():
        df = pd.DataFrame(details)
        df["dates"] = date
        records.append(df)
        
    return pd.concat(records, ignore_index=True)


#########################################
##               HELPER                ##
#########################################

def _detect_nesting_level(data):
    if not isinstance(data, dict) or not data:
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

def _load_dsfile(file: str, folder: str) -> dict:
    path = check_dsfiles_exist(DSFILE_MAP[file], folder)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


#########################################
##                MAIN                 ##
#########################################

def conv_pkltodf(file: str, folder: str) -> pd.DataFrame:
    data = _load_dsfile(file, folder)
    max_layer = _detect_nesting_level(data)
    if max_layer == 3:
        return conv_maxlayer_3(data)
    elif max_layer == 2:
        return conv_maxlayer_2(data)
    else:
        raise ValueError("[ERROR] Data dict layering not valid. Aborting dataframe conversion!")


if __name__=="__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and flatten pickled dataset files."
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available file keys and exit"
    )

    parser.add_argument(
        "file_key",
        help="file key to process [aibots_crawlers_time, anomalies, ...]"
    )

    args = parser.parse_args()

    if args.keys:
        print("Available file keys:")
        for key in DSFILE_MAP.keys():
            print(f"\t- {key}")
        exit(0)

    FILE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = FILE_DIR.parents[1]
    DSDIR = PROJECT_ROOT / "datasets" / "processed"

    df = conv_pkltodf(args.file_key, DSDIR)
    print(df)