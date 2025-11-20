#!/usr/bin/env python3
"""
Converts all or single raw dataset file as fetched from Cloudflare to internal df structure
- converts raw dataset as defined in DSFILE_MAP
- outputs pkl file

Outputs:
    pkl files : datasets/processed/<dataset>.pkl

Usage: 
    python -m src.data.preprocess <all|key>
"""

import json
import numpy as np
import pickle
from pathlib import Path


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
DSFILE_MAP = {
    "aibots_crawlers_time": [True, None, "aibots_crawlers_time_pull"],
    "anomalies": [False, False, "anomalies_pull"],
    "bots_time": [True, None, "bots_time_pull"],
    "httpreq_time": [True, None, "httpreq_time_pull"],
    "httpreq_automated_time": [True, None, "httpreq_automated_time_pull"],
    "httpreq_human_time": [True, None, "httpreq_human_time_pull"],
    "httpreq": [False, False, "httpreq_pull"],
    "traffic_time": [True, None, "traffic_time_pull"],
    "traffic": [False, False, "traffic_pull"],
    "iq_bandwidth_time": [True, None, "inetqal_bandwidth_time_pull"],
    "iq_dns_time": [True, None, "inetqal_dns_time_pull"],
    "iq_latency_time": [True, None, "inetqal_latency_time_pull"],
    "l3_origin": [False, True, "l3attack_origin_pull"],
    "l3_origin_time": [True, None, "l3attack_origin_time_pull"],
    "l3_origin_bitrate_time": [True, None, "l3attack_origin_bitrate_time_pull"],
    "l3_origin_protocol_time": [True, None, "l3attack_origin_protocol_time_pull"],
    "l3_origin_duration_time": [True, None, "l3attack_origin_duration_time_pull"],
    "l3_target": [False, True, "l3attack_target_pull"],
    "l3_target_time": [True, None, "l3attack_target_time_pull"],
    "l3_target_bitrate_time": [True, None, "l3attack_target_bitrate_time_pull"],
    "l3_target_duration_time": [True, None, "l3attack_target_duration_time_pull"],
    "l3_target_protocol_time": [True, None, "l3attack_target_protocol_time_pull"],
    "l7_target": [False, True, "l7attack_target_pull"],
    "l7_time": [True, None, "l7attack_time_pull"],
    "l7_mitigations_time": [True, None, "l7attack_mitigations_time_pull"],
    "l7_origin": [False, True, "l7attack_origin_pull"],
} # name: time_data, csplit, file


#########################################
##            INTIAL CHECK             ##
#########################################

def _dsfile_exists(value: list):
    prefix = value[-1]
    match = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not match:
        raise FileNotFoundError(f"[Error] No JSON file starting with '{prefix}' found. Aborting preprocessing stage.")

def check_dsfiles_exist(file_map: dict):
    for name in file_map:
        _dsfile_exists(file_map[name])
    print("All dataset prefixes validated. Starting preprocessing stage...")

def find_latest_pull(prefix: str) -> Path:
    matches = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"[Error] No files found starting with: {prefix}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


#########################################
##              EXTRACTION             ##
#########################################

def nontemp_extraction(data: dict, field_map: dict, result_key: str = "main") -> dict:
    data_dict = {}

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
                raise ValueError(f"[Error] Unsupported field logic: {out_field}")
            
        if any(extracted.values()):
            data_dict[day_str] = extracted

    return data_dict

def nontemp_csplit_extraction(data: dict, field_map: dict, result_key: str = "main") -> dict:
    data_dict = {}

    for region, region_data in data.items():
        region_result = nontemp_extraction(region_data, field_map, result_key)
        if region_result:
            data_dict[region] = region_result

    return data_dict

def temp_csplit_extraction(data: dict, fields: list, result_key: str = "main") -> dict:
    data_dict = {}

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


#########################################
##               CONVERTER             ##
#########################################

def _read_file(file: Path) -> dict:
    with open(file,'r') as f:
        return json.load(f)

def read_json_notime(data: dict, name: str) -> dict:
    print(f"{name}\t processed as non-temporal data...")
    
    if name == "anomalies":
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

    elif name in ["httpreq", "traffic"]:
        data_dict = nontemp_extraction(
            data,
            field_map={
                "countries": "clientCountryAlpha2",
                "values": lambda item: float(item["value"]) if "value" in item and item["value"] not in (None, "null", "") else np.nan,
            }
        )
    else:
        raise KeyError(f"[Error] Key {name} not found, aborting!")
    
    return data_dict

def read_json_notime_csplit(data: dict, name: str) -> dict:
    print(f"{name}\t processed as non-temporal, country-resolved data...")
        
    if "target" in name:
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

def read_json_time_csplit(data: dict, name: str) -> dict:
    print(f"{name}\t processed as temporal, country-resolved data...")
    
    if name.startswith("iq"):
        data_dict = temp_csplit_extraction(
            data, 
            fields=["p25", "p50", "p75"]
        )
    elif "bitrate" in name:
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
    elif "duration" in name:
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
    elif "protocol" in name:
        data_dict = temp_csplit_extraction(
            data, 
            fields=["UDP", "TCP", "ICMP", "GRE"]
        )
    elif "mitigations" in name:
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
    elif "time" in name:
        data_dict = temp_csplit_extraction(
            data, 
            fields=["values"]
        )
    else:
        raise KeyError(f"[Error] Key {name} not found, aborting!")
    
    return data_dict


#########################################
##             PRINT-OUT               ##
#########################################

def save_data(data: dict, name: str):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outfile = PROCESSED_DIR / f"{name}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name}\t saved to {outfile}!")


#########################################
##                MAIN                 ##
#########################################

def preprocess_all():
    check_dsfiles_exist(DSFILE_MAP)
    
    for key in DSFILE_MAP:
        preprocess_single(key, check=False)


def preprocess_single(name: str, check: bool = True):
    value = DSFILE_MAP[name]

    if check:
        _dsfile_exists(value)
    
    is_time = value[0]
    is_csplit = bool(value[1])
    prefix = value[-1]

    path = find_latest_pull(prefix)
    data = _read_file(path)

    if is_time:
        conv_data = read_json_time_csplit(data, name)
    else:
        if is_csplit:
            conv_data = read_json_notime_csplit(data, name)
        else: 
            conv_data = read_json_notime(data, name)

    if conv_data:
        save_data(conv_data, name)
    else:
        print(f"[Error] No data extracted for {name}, skipping save.")


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess all or single dataset file."
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available file keys and exit"
    )

    parser.add_argument(
        "file_key",
        nargs="?",
        help="file key to process <all|[aibots_crawlers_time, anomalies, ...]>"
    )

    args = parser.parse_args()

    if args.keys:
        print("Available file keys:")
        for key in DSFILE_MAP.keys():
            print(f"\t- {key}")
        exit(0)

    if not args.file_key:
        parser.print_help()
        exit(1)

    key = args.file_key

    if key.lower() == "all":
        preprocess_all()
    else:
        preprocess_single(key)
