#!/usr/bin/env python3
"""
Preprocesses all or single raw dataset file as fetched from Cloudflare.
- converts raw dataset as defined in DSFILE_MAP
- outputs pkl file

Output:
    pkl files : app/datasets/processed/<dataset>.pkl

Usage: 
    python -m app.src.data.processing.preprocess [-k] <all|FILE_KEY>
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Any

from . import DSFILE_MAP


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def dsfile_exists(prefix: str) -> None:
    match = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not match:
        raise FileNotFoundError(f"[ERROR] No JSON file starting with '{prefix}' found. Aborting preprocessing stage.")

def find_latest_pull(prefix: str) -> Path:
    matches = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"[ERROR] No files found starting with: {prefix}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def nontemp_extraction(
    data: dict, 
    field_map: dict, 
    result_key: str = "main"
) -> dict:
    data_dict: dict[str, Any] = {}

    for day_entry in data:
        fetch = day_entry.get("fetch", {})
        day_str = fetch.get("start")
        day_data = fetch.get("value", {})

        if not day_str or not isinstance(day_data, dict):
            continue
        if not day_data.get("success"):
            continue

        result = day_data.get("result", {}).get(result_key, [])
        if not result:
            continue

        extracted = {}
        for out_field, logic in field_map.items():
            if callable(logic):
                extracted[out_field] = [logic(item) for item in result]
            elif isinstance(logic, str):
                extracted[out_field] = [item.get(logic) for item in result if logic in item]
            else:
                raise ValueError(f"[ERROR] Unsupported field logic: {out_field}")
            
        if any(extracted.values()):
            data_dict[day_str] = extracted

    return data_dict

def nontemp_csplit_extraction(
    data: dict, 
    field_map: dict, 
    result_key: str = "main"
) -> dict:
    data_dict: dict[str, Any] = {}

    for region, region_data in data.items():
        region_result = nontemp_extraction(region_data, field_map, result_key)
        if region_result:
            data_dict[region] = region_result

    return data_dict

def temp_csplit_extraction(
    data: dict, 
    fields: list, 
    result_key: str = "main"
) -> dict:
    data_dict: dict[str, Any] = {}

    for region, region_data in data.items():
        data_dict[region] = {}

        for day_entry in region_data:
            fetch = day_entry.get("fetch", {})
            day_str = fetch.get("start")
            day_data = fetch.get("value", {})

            if not day_str or not isinstance(day_data, dict):
                continue
            if not day_data.get("success"):
                continue

            result = day_data.get("result", {}).get(result_key, {})
            timestamps = result.get("timestamps", [])
            if not timestamps:
                continue

            extracted = {"timestamps": timestamps}
            valid = True # data found, False = data not found

            for field in fields or []:
                vals = result.get(field, [])
                if not vals:
                    valid = False
                    break
                extracted[field] = [
                    float(v) if v not in (None, "null", "") else np.nan for v in vals
                ]

            if valid and any(extracted.values()):
                data_dict[region][day_str] = extracted

    return data_dict

def read_json_notime(data: dict, key: str) -> dict:
    print(f"[INFO] {key} preprocessed as non-temporal data...")
    
    if key == "anomalies":
        data_dict = nontemp_extraction(
            data, 
            result_key="trafficAnomalies", 
            field_map={
                "types": "type",
                "status": "status",
                "startDates": "startDate",
                "endDates": "endDate",
                "location": lambda item: (
                    (item.get("asnDetails", {}) or {}).get("location", {}).get("code")
                    or (item.get("locationDetails", {}) or {}).get("code")
                ) if isinstance(item, dict) and (item.get("asnDetails") or item.get("locationDetails")) else np.nan,
            }
        )

    elif key in ["httpreq", "traffic"]:
        data_dict = nontemp_extraction(
            data,
            field_map={
                "countries": "clientCountryAlpha2",
                "values": lambda item: float(item["value"]) if "value" in item and item["value"] not in (None, "null", "") else np.nan,
            }
        )
    else:
        raise KeyError(f"[ERROR] Key {key} not found, aborting!")
    
    return data_dict

def read_json_notime_csplit(data: dict, key: str) -> dict:
    print(f"[INFO] {key} preprocessed as non-temporal, country-resolved data...")
        
    if "target" in key:
        country_field = "targetCountryAlpha2"
    else:
        country_field = "originCountryAlpha2"

    data_dict = nontemp_csplit_extraction(
        data,
        field_map={
            "countries": country_field,
            "values": lambda item: float(item["value"]) if "value" in item and item["value"] not in (None, "null", "") else np.nan,
            "ranks": lambda item: item["rank"] if "rank" in item else np.nan
        }
    )
    
    return data_dict

def read_json_time_csplit(data: dict, key: str) -> dict:
    print(f"[INFO] {key} preprocessed as temporal, country-resolved data...")
    
    if key.startswith("iq"):
        data_dict = temp_csplit_extraction(
            data, 
            fields=["p25", "p50", "p75"]
        )
    elif "bitrate" in key:
        data_dict = temp_csplit_extraction(
            data, 
            fields=[
                "UNDER_500_MBPS",
                "_500_MBPS_TO_1_GBPS",
                "_1_GBPS_TO_10_GBPS",
                "_10_GBPS_TO_100_GBPS",
                "OVER_100_GBPS",
            ]
        )
    elif "duration" in key:
        data_dict = temp_csplit_extraction(
            data, 
            fields=[
                "UNDER_10_MINS",
                "_10_MINS_TO_20_MINS",
                "_20_MINS_TO_40_MINS",
                "_40_MINS_TO_1_HOUR",
                "_1_HOUR_TO_3_HOURS",
                "OVER_3_HOURS",
            ]
        )
    elif "protocol" in key:
        data_dict = temp_csplit_extraction(
            data, 
            fields=["UDP", "TCP", "ICMP", "GRE"]
        )
    elif "mitigations" in key:
        data_dict = temp_csplit_extraction(
            data, 
            fields=[
                "WAF",
                "DDOS",
                "ACCESS_RULES",
                "IP_REPUTATION",
                "BOT_MANAGEMENT",
                "API_SHIELD",
                "DATA_LOSS_PREVENTION",
            ]
        )
    elif "time" in key:
        data_dict = temp_csplit_extraction(
            data, 
            fields=["values"]
        )
    else:
        raise KeyError(f"[ERROR] Key {key} not found, aborting!")
    
    return data_dict

def save_data(data: dict, key: str) -> None:
    outfile = PROCESSED_DIR / f"{key}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] {key} saved to {outfile}")

def preprocess_single(
    key: str,
    data: dict | None = None,
    save_pkl: bool = True,
    return_data: bool = False
) -> dict | None:
    print(f"[INFO] Preprocessing {key}...")
    value = DSFILE_MAP[key]
    is_time, is_csplit, prefix = value
    dsfile_exists(str(prefix))

    if data is None:
        latest_data = find_latest_pull(str(prefix))
        with open(latest_data, 'r') as f:
            data = json.load(f)

    if is_time:
        conv_data = read_json_time_csplit(data, key)
    else:
        if is_csplit:
            conv_data = read_json_notime_csplit(data, key)
        else: 
            conv_data = read_json_notime(data, key)
    print(f"[OK] {key} successfully preprocessed")

    if conv_data and save_pkl:
        save_data(conv_data, key)

    if return_data:
        return conv_data
    else:
        return None

def preprocess_all() -> None:
    for key in DSFILE_MAP:
        preprocess_single(key)

    print("[DONE] All datasets preprocessed!")


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess all or single dataset file"
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available dataset keys and exit"
    )

    parser.add_argument(
        "file_key",
        help="<all|[aibots_crawlers_time, anomalies, ...]> file key to process"
    )

    args = parser.parse_args()

    if args.keys:
        print("Available file keys:")
        for key in DSFILE_MAP.keys():
            print(f"\t- {key}")
        exit(0)

    if args.file_key not in ["all", *DSFILE_MAP.keys()]:
        print(f"[ERROR] file_key {args.file_key} cannot be preprocessed\n")
        parser.print_help()
        exit(1)

    if args.file_key.lower() == "all":
        preprocess_all()
    else:
        preprocess_single(args.file_key)
