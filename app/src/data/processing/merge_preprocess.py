#!/usr/bin/env python3
"""
Merges and preprocesses raw datasets fetched in multiple pulls from Cloudflare
- merges 2 datasets and orders by timestamps
- converts raw datasets as defined in DSFILE_MAP
- outputs pkl files

Output:
    pkl files : datasets/processed/<dataset>.pkl

Usage: 
    python -m app.src.data.processing.merge_preprocess [-k] [-S] [-N <int>] <all|FILE_KEY>
"""

import json
from pathlib import Path

from . import DSFILE_MAP
from .preprocess import dsfile_exists, read_json_time_csplit, save_data


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


def check_ts_order(l1: list, l2: list) -> None:
    """Checks timestamp order to verify merge direction."""
    ts1 = l1[-1]["fetch"]["value"]["result"]["main"]["timestamps"][-1]
    ts2 = l2[0]["fetch"]["value"]["result"]["main"]["timestamps"][0]
    if ts1 >= ts2:
        raise ValueError(
            "[Error] Timestamp continuity error: last_ts_file1 > first_ts_file2. Use different merge dir."
        )

def _find_latest_pulls(prefix: str, i: int, j: int) -> tuple[Path, Path]:
    """Retrieves two dataset fetches from internal file naming convention."""
    matches = list(RAW_DIR.glob(f"{prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"[Error] No files found starting with: {prefix}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[i], matches[j]

def _read_file(file: Path) -> dict:
    """Reads dataset json file."""
    with open(file,'r') as f:
        data: dict = json.load(f)
    return data

def _merge_dicts(d1: dict, d2: dict) -> dict:
    """Generates merged dataset dictionary from two datasets."""
    merged = {}
    for region in d1.keys():
        if region not in d2.keys():
            raise ValueError(f"[Error] Region {region} missing; files do not match. Aborting merge.")
        
        list1, list2 = d1[region], d2[region]
        if region == "worldwide":
            check_ts_order(list1, list2)
        
        combined = list1 + list2
        for i, entry in enumerate(combined, start=1):
            entry["idx"] = i
        
        merged[region] = combined
    return merged

def merge_pulls(
    merge_dir: int, 
    n_pulls: int, 
    prefix: str
) -> dict:
    """Calls merger for specified number of fetch rounds (n_pulls)."""
    for n in range(1, n_pulls):
        merged: dict = run_merger(
            prefix=prefix,
            merge_dir=merge_dir,
            n=n,
            d=merged if n > 1 else None
        )
        print(f"[INFO] ... completed merge round {n}")
    return merged

def run_merger(
        prefix: str, 
        merge_dir: int, 
        n: int, 
        d: dict | None = None
    ) -> dict:
    """Performs merger of two subsequent fetch pulls."""
    if n == 1:
        if merge_dir:
            p1, p2 = _find_latest_pulls(prefix, n-1, n)
        else:
            p1, p2 = _find_latest_pulls(prefix, n, n-1)
        d1, d2 = _read_file(p1), _read_file(p2)
        return _merge_dicts(d1, d2)
    
    if d is None:
        raise ValueError("[ERROR] Multi-pull merger failed. Merge dictionary is of NoneType!")
    if merge_dir:
        _, p2 = _find_latest_pulls(prefix, n-1, n)
        d2 = _read_file(p2)
        return _merge_dicts(d, d2)
    else:
        p1, _ = _find_latest_pulls(prefix, n, n-1)
        d1 = _read_file(p1)
        return _merge_dicts(d1, d)

def save_merged_data(data: dict, key: str) -> None:
    """Saves merged dataset as json file."""
    outfile = RAW_DIR / f"{key}_merged.json"
    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] {key} saved to {outfile}")

def preprocess_merged_data(
    data: dict, 
    key: str,
) -> None:
    """Preprocesses merged dataset."""
    conv_data = read_json_time_csplit(data, key)

    if conv_data:
        save_data(conv_data, key)
    else:
        print(f"[Error] No data extracted for {key}, skipping save")

def merge_single(
    key: str, 
    n_pulls: int, 
    save_only: bool, 
) -> None:
    """Runs pipeline for merging and preprocessing single dataset."""
    print(f"[INFO] Merging {key}...")

    prefix = str(DSFILE_MAP[key][-1])
    dsfile_exists(prefix)

    try:
        merged = merge_pulls(0, n_pulls, prefix)
    except ValueError:
        merged = merge_pulls(1, n_pulls, prefix)
    
    print(f"[OK] {key} successfully merged")
    if save_only:
        save_merged_data(merged, key)
    else:
        preprocess_merged_data(merged, key)

def merge_all(
    n_pulls: int,
    save_only: bool
) -> None:
    """Runs pipeline for merging and preprocessing all datasets."""
    for key, value in DSFILE_MAP.items():
        if value[0]:
            merge_single(key, n_pulls, save_only)
        
    print(f"[DONE] All datasets merged and preprocessed!")


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge and preprocess all or single dataset with multiple pull dates"
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available dataset keys and exit"
    )
    
    parser.add_argument(
        "-S", "--save",
        action="store_true",
        help="save-only mode; merged data will be stored and not further preprocessed"
    )

    parser.add_argument(
        "-N",
        type=int,
        default=2,
        help="number of pulls [default: 2] e.g. 3 merges 3 files"
    )

    parser.add_argument(
        "file_key",
        help="<all|[aibots_crawlers_time, bots_time, ...]> file key to process "
    )

    args = parser.parse_args()

    time_dsfiles = [k for k, v in DSFILE_MAP.items() if v[0]]
    if args.keys:
        print("Available file keys:")
        for key in time_dsfiles:
            print(f"\t- {key}")
        exit(0)

    if args.file_key not in ["all", *time_dsfiles]:
        print(f"[Error] file_key {args.file_key} cannot be processed\n")
        parser.print_help()
        exit(1)

    if args.file_key.lower() == "all":
        merge_all(args.N, args.save)
    else:
        merge_single(args.file_key, args.N, args.save)
