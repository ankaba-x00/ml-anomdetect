#!/usr/bin/env python3
"""
Processes raw datasets ready for feature engineering
- merges multiple dataset pulls
- extracts data from json layering
- (optional) generates pkl files for data analysis
- for timeseries:
    - flattens nesting layers
    - generates hdf files for feature engineering

Output:
    FILES: app/datasets/processed/<all|FILE_KEY>.hdf
           app/datasets/processed/<FILE_KEY>.pkl

Usage:
    python -m app.src.pipelines.process_dataset [-k] [-S] [-o <FILE_NAME>] <all|FILE_KEY>
"""

from pathlib import Path

from app.src.data.processing import DSFILE_MAP
from app.src.data.processing.flatten import flatten_single
from app.src.data.processing.merge_preprocess import merge_pulls
from app.src.data.processing.preprocess import preprocess_single


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


def get_npulls(prefix: str) -> int:
    "Retrieves number of pulls per dataset"

    return len(list(RAW_DIR.glob(f"{prefix}*.json")))

def process_dataset(
    key: str, 
    save_pkl: bool, 
    h5_fname: str
) -> None:
    """Runs process dataset pipeline for selected dataset type."""
    prefix = str(DSFILE_MAP[key][-1])
    n_pulls = get_npulls(prefix)
    data = None

    if n_pulls == 0:
        raise FileNotFoundError(f"[ERROR] No dataset found for {key}")
    elif n_pulls > 1:
        try:
            print(f"[INFO] Merging {key}...")
            data = merge_pulls(0, n_pulls, prefix)
        except ValueError:
            data = merge_pulls(1, n_pulls, prefix)
        print(f"[OK] {key} successfully merged")

    prep_data = preprocess_single(
        key, 
        data,
        save_pkl, 
        return_data=True
    )

    flatten_single(key, h5_fname, prep_data)

    print(f"[DONE] {key} successfully processed")

def process_all(
    save_pkl: bool, 
    h5_fname: str
) -> None:
    """Runs process dataset pipeline for all dataset types."""
    for key in DSFILE_MAP:
        process_dataset(key, save_pkl, h5_fname)

    print(f"[DONE] All datasets processed!")


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process raw datasets ready for feature engineering and data analysis"
    )

    parser.add_argument(
        "-k", "--keys",
        action="store_true",
        help="show available dataset keys and exit"
    )

    parser.add_argument(
        "-S", "--save",
        action="store_true",
        help="save preprocessed data as pkl files for data analysis"
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

    if args.keys:
        print("Available file keys:")
        for key in DSFILE_MAP.keys():
            print(f"\t- {key}")
        exit(0)

    if args.file_key not in ["all", *DSFILE_MAP.keys()]:
        print(f"[Error] file_key {args.file_key} cannot be processed\n")
        parser.print_help()
        exit(1)

    if args.out is None:
        args.out = f"{args.file_key.lower()}.h5"

    if args.file_key.lower() == "all":
        process_all(args.save, args.out)
    else:
        process_dataset(
            args.file_key.lower(), 
            args.save, 
            args.out
        )