import numpy.core, sys
import numpy as np
import pandas as pd
from pathlib import Path

from app.deployment import RegionTimeseriesFetchResult
from app.src.data.processing import FIELD_MAP
from app.src.data.building import ProcessedRegionTimeseries

sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def _load_from_disk(keys: set[str], country: str) -> RegionTimeseriesFetchResult:
    """Load timeseries for country region from h5 file on disk."""

    data: dict = {}
    for k in keys:
        if k == "timestamps":
            continue
        key = f"{k}_time"
        df = pd.read_hdf(PROCESSED_DIR / "all.h5", key)
        if "regions" in df.columns:
            df = df[df["regions"] == country]
            # order by timestamps

        dict_fields = {
            "l3_origin_bitrate_time": "bitrate", 
            "l3_origin_duration_time": "duration", 
            "l3_origin_protocol_time": "protocol"
        }
        if key in dict_fields.keys():
            df_red = df.loc[:,["metric", "values"]]
            group = FIELD_MAP[dict_fields[key]]
            data[key[:-5]] = {
                g: list(df_red.loc[df_red["metric"] == g]["values"])
                for g in group
            }
        else:
            if key == "l7_time":
                data["timestamps"] = list(df["timestamps"].values)
            data[key[:-5]] = list(df["values"].values)

    return RegionTimeseriesFetchResult.from_dict(data)

def _check_values(key: str, ts_len: int, values: list[str]) -> list[str] | None:
    val_len = len(values)
    if ts_len != 96:
        raise ValueError(f"[ERROR] {key} not processed in 15min chunks; aborting now")
    elif not ts_len or not val_len:
        raise ValueError(f"[ERROR] No data pulled for {key}; aborting now")
    elif ts_len > val_len:
        print(f"[WARN] Zero-fill required for {key} to proceed; check dataset")
        return (values + ["0.0"]) * (ts_len - val_len)
    elif ts_len < val_len:
        print(f"[WARN] Value truncation required for {key} to proceed; check dataset")
        return values[:ts_len]
    else:
        return values


def _conv_todf(
    values: list[str], 
    timestamps: list[str], 
    name: str
) -> pd.DataFrame:
    "Converts raw stringified dataset values and checks for data consistency."

    ts = pd.to_datetime(timestamps, errors="coerce")
    val = _check_values(name, len(ts), values)

    return pd.DataFrame(
        data={name: val}, 
        index=ts
    ).astype("float64")

def _conv_weigh_todf(
    values: dict[str, list[str]], 
    timestamps: list[str], 
    mids: dict,
    name: str
) -> pd.DataFrame:
    "Converts raw stringified value distributions to weighted averages per timestamp and checks for data consistency."

    ts = pd.to_datetime(timestamps, errors="coerce")
    cols = [c for c in mids.keys() if c in values.keys()]
    
    X = np.column_stack([
        np.asarray(_check_values(name, len(ts), values[c]), dtype=float)
        for c in cols
    ])
    W = np.asarray([mids[c] for c in cols], dtype=float)
    den = X.sum(axis=1)
    den = np.where(den <= 0, np.nan, den)
    weighted = (X * W).sum(axis=1) / den
    weighted = np.nan_to_num(weighted, nan=0.0)

    return pd.DataFrame(
        data={name: weighted}, 
        index=ts
    ).astype("float64")

def _conv_fract_todf(
    values: dict[str, list[str]], 
    timestamps: list[str]
) -> pd.DataFrame:
    "Converts raw stringified protocol bucket data to fractional shares and Shannon entropy per timestamp and checks for data consistency."

    ts = pd.to_datetime(timestamps, errors="coerce")
    pcols = ["udp", "tcp", "icmp", "gre"]

    X = np.column_stack([
        np.asarray(
            _check_values(p, len(ts), values[f"{p.upper()}"]), 
            dtype=float
        ) for p in pcols
    ])
    den = X.sum(axis=1)
    den = np.where(den <= 0, np.nan, den)
    frac = {}
    for i, p in enumerate(pcols):
        p_frac = X[:,i] / den
        frac[f"{p}_frac"] = np.nan_to_num(p_frac, nan=0.0)

    P = np.column_stack(list(frac.values()))
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    frac["protocol_entropy"] = -(P * np.log2(P)).sum(axis=1)

    return pd.DataFrame(frac, index=ts).astype("float64")

def load_base(
    country: str, 
    data: RegionTimeseriesFetchResult | None = None
) -> ProcessedRegionTimeseries:
    """Preprocesses raw timeseries data from memory or disk."""

    if data is None:
        keys = RegionTimeseriesFetchResult.__getatts__()
        data = _load_from_disk(keys, country)

    # weighted bitrate and duration avg
    bitrate_mids = {
        "UNDER_500_MBPS": 250,
        "_500_MBPS_TO_1_GBPS": 750,
        "_1_GBPS_TO_10_GBPS": 5500,
        "_10_GBPS_TO_100_GBPS": 55000,
        "OVER_100_GBPS": 100000
    }
    dur_mids = {
        "UNDER_10_MINS": 5,
        "_10_MINS_TO_20_MINS": 15,
        "_20_MINS_TO_40_MINS": 30,
        "_40_MINS_TO_1_HOUR": 50,
        "_1_HOUR_TO_3_HOURS": 120,
        "OVER_3_HOURS": 300
    }

    return ProcessedRegionTimeseries(
        l3o=_conv_todf(data.l3_origin, data.timestamps, "l3_origin"),
        l3t=_conv_todf(data.l3_target, data.timestamps, "l3_target"), 
        l7=_conv_todf(data.l7, data.timestamps, "l7_traffic"),
        http=_conv_todf(data.httpreq, data.timestamps, "http"),
        http_auto=_conv_todf(data.httpreq_automated, data.timestamps, "http_auto"),
        http_human=_conv_todf(data.httpreq_human, data.timestamps, "http_human"),
        netflow=_conv_todf(data.traffic, data.timestamps, "netflow"),
        bots=_conv_todf(data.bots, data.timestamps, "bots_total"),
        ai=_conv_todf(data.aibots_crawlers, data.timestamps, "ai_bots"),
        l3_bitrate=_conv_weigh_todf(data.l3_origin_bitrate, data.timestamps, bitrate_mids, "l3_bitrate_avg"),
        l3_duration=_conv_weigh_todf(data.l3_origin_duration, data.timestamps, dur_mids, "l3_duration_avg"),
        protocol=_conv_fract_todf(data.l3_origin_protocol, data.timestamps),
    )