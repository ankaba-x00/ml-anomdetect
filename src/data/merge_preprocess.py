"""
Merges 2 timeseries datasets fetched from Cloudflare and preprocesses to 
internal df structure
- merges 2 timeseries datasets and orders by timestamps
- converts raw datasets as defined in DSFILE_MAP
- outputs pkl files

Outputs:
    pkl files : datasets/processed/<dataset>.pkl

Usage: 
    python -m src.data.merge_preprocess
"""

import json
import numpy as np
import pickle
from pathlib import Path
from src.data.preprocess import  read_json_time_csplit, save_data

#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
TIME_DSFILE_MAP = {
    "aibots_crawlers_time": [True, None, "aibots_crawlers_time_pull"],
    "bots_time": [True, None, "bots_time_pull"],
    "httpreq_time": [True, None, "httpreq_time_pull"],
    "httpreq_automated_time": [True, None, "httpreq_automated_time_pull"],
    "httpreq_human_time": [True, None, "httpreq_human_time_pull"],
    "traffic_time": [True, None, "traffic_time_pull"],
    # "iq_bandwidth_time": [True, None, "inetqal_bandwidth_time_pull"],
    # "iq_dns_time": [True, None, "inetqal_dns_time_pull"],
    # "iq_latency_time": [True, None, "inetqal_latency_time_pull"],
    "l3_origin_time": [True, None, "l3attack_origin_time_pull"],
    "l3_origin_bitrate_time": [True, None, "l3attack_origin_bitrate_time_pull"],
    "l3_origin_protocol_time": [True, None, "l3attack_origin_protocol_time_pull"],
    "l3_origin_duration_time": [True, None, "l3attack_origin_duration_time_pull"],
    "l3_target_time": [True, None, "l3attack_target_time_pull"],
    # "l3_target_bitrate_time": [True, None, "l3attack_target_bitrate_time_pull"],
    # "l3_target_duration_time": [True, None, "l3attack_target_duration_time_pull"],
    # "l3_target_protocol_time": [True, None, "l3attack_target_protocol_time_pull"],
    "l7_time": [True, None, "l7attack_time_pull"],
    # "l7_mitigations_time": [True, None, "l7attack_mitigations_time_pull"],
} # name: time_data, csplit, file

#########################################
##            INTIAL CHECK             ##
#########################################

def _pullversions_exist(value: list):
    prefix = value[-1]
    match = list(RAW_DIR.glob(f"{prefix}*.json"))
    if len(match) <= 1:
        raise FileNotFoundError(f"[Error] Pull versions starting with '{prefix}' not found. Aborting preprocessing stage.")

def check_pullversions_exist(file_map: dict):
    for name in file_map:
        _pullversions_exist(file_map[name])
    print("All dataset prefix pull versions validated. Starting merging stage...")

def check_ts_order(l1: list, l2: list):
    ts1 = l1[-1]["fetch"]["value"]["result"]["main"]["timestamps"][-1]
    ts2 = l2[0]["fetch"]["value"]["result"]["main"]["timestamps"][0]
    if ts1 >= ts2:
        raise ValueError(
            "[Error] Timestamp continuity error" \
            f"  last timestamp in file1:  {ts1}\n" \
            f"  first timestamp in file2: {ts2}\n" \
            "Expected: last_ts_file1 < first_ts_file2. Use different merge dir."
        )

#########################################
##               HELPER                ##
#########################################

def find_latest_pulls(prefix: str, dir: int) -> tuple[Path, Path]:
    matches = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"[Error] No files found starting with: {prefix}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if dir:
        return matches[0], matches[1]
    return matches[1], matches[0]

def _read_file(file: Path) -> dict:
    with open(file,'r') as f:
        return json.load(f)

def save_merged_data(data: dict, name: str):
    outfile = RAW_DIR / f"{name}_merged.json"
    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{name}\t saved to {outfile}!")

def preprocess_merged_data(data: dict, name: str):
    conv_data = read_json_time_csplit(data, name)
    if conv_data:
        save_data(conv_data, name)
    else:
        print(f"[Error] No data extracted for {name}, skipping save.")

#########################################
##                MERGE                ##
#########################################

def merge_single(name: str, dir: int, save_only: bool = False, check: bool = True):
    value = TIME_DSFILE_MAP[name]
    if check:
        _pullversions_exist(value)

    p1, p2 = find_latest_pulls(value[-1], dir)
    f1, f2 = _read_file(p1), _read_file(p2)

    merged = {}

    for region in f1.keys():
        if region not in f2.keys():
            raise ValueError(f"[Error] Region {region} missing; files do not match. Aborting merge.")
        
        list1, list2 = f1[region], f2[region]
        if region == "worldwide":
            check_ts_order(list1, list2)
        
        combined = list1 + list2
        for i, entry in enumerate(combined, start=1):
            entry["idx"] = i
        
        merged[region] = combined
    
    print(f"{name}\t successfully merged!")
    if save_only:
        save_merged_data(merged, name)
    else:
        preprocess_merged_data(merged, name)


def merge_all(dir: int, save_only: bool = False):
    check_pullversions_exist(TIME_DSFILE_MAP)

    for key in TIME_DSFILE_MAP:
        merge_single(name=key, dir=dir, save_only=save_only, check=False)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge and preprocess all or single dataset with multiple pull dates."
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available file keys and exit"
    )

    parser.add_argument(
        "-d", "--dir",
        action="store_true",
        help="show available merge directions and exit"
    )

    parser.add_argument(
        "-S", "--save",
        action="store_true",
        help="save-only mode; merged data will be stored and not further preprocessed"
    )

    parser.add_argument(
        "file_key",
        nargs="?",
        help="file key to process <all|[aibots_crawlers_time, bots_time, ...]>"
    )

    parser.add_argument(
        "merge_dir",
        nargs="?",
        help="merge direction <0=consecutively|1=non-consecutively>"
    )

    args = parser.parse_args()

    if args.keys:
        print("Available file keys:")
        for key in TIME_DSFILE_MAP.keys():
            print(f"   - {key}")
        exit(0)

    if args.dir:
        print("Available merge directions:" \
        "\n  - 0 : consecutively = later pull adds subsequent timestamps to previous pull" \
        "\n  - 1 : non-consecutively = later pull adds proceeding timestamps to previous pull")
        
        exit(0)

    if args.file_key is None or args.file_key not in ["all", *TIME_DSFILE_MAP.keys()]:
        print(f"[Error] file_key {args.file_key} cannot be processed.\n")
        parser.print_help()
        exit(1)

    if args.merge_dir is None or args.merge_dir not in ["0", "1"]:
        print(f"[Error] merge_dir {args.merge_dir} cannot be processed.\n")
        parser.print_help()
        exit(1)

    key, dir, save_only = args.file_key, int(args.merge_dir), args.save

    if key.lower() == "all":
        merge_all(dir, save_only)
    else:
        merge_single(key, dir, save_only)
