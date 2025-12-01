import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from sklearn.base import TransformerMixin
from src.data.fetch import _headers, _requests_session, _print_range
from src.data.feature_engineering import build_feature_matrix
from src.exploration.core.time_utils import conv_iso_to_local, conv_iso_to_local_with_daytype, conv_iso_to_local_with_daytimes
from src.exploration.core.params import timezones

#########################################
##                CONFIG               ##
#########################################

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("[ERROR] API_TOKEN not found in environment (.env)")


#########################################
##                PARAMS               ##
#########################################

FILE_DIR = Path(__file__).resolve().parent
OUT_DIR = FILE_DIR / "data" / "featured"
OUT_DIR.mkdir(parents=True, exist_ok=True)


#########################################
##            HELPER FETCH             ##
#########################################

def _fetch_timedata(TITLE: str, URL: str, BASE_PARAMS: dict, date_from: datetime, date_to: datetime):
    """
    Fetches timeseries data in memory for country from Cloudflare API in 1-hour buckets.
    """
    DATE_MIN, DATE_MAX = date_from, date_to.replace(hour=23, minute=45)

    session = _requests_session()
    region_results = {}

    params = dict(BASE_PARAMS)
    params["dateStart"] = DATE_MIN.isoformat().replace("+00:00", "Z")
    params["dateEnd"]   = DATE_MAX.isoformat().replace("+00:00", "Z")

    resp = session.get(URL, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 200:
        try:
            FIELD_MAP = {
                "bitrate": [
                    "UNDER_500_MBPS",
                    "_500_MBPS_TO_1_GBPS",
                    "_1_GBPS_TO_10_GBPS",
                    "_10_GBPS_TO_100_GBPS",
                    "OVER_100_GBPS",
                ],
                "duration": [
                    "UNDER_10_MINS",
                    "_10_MINS_TO_20_MINS",
                    "_20_MINS_TO_40_MINS",
                    "_40_MINS_TO_1_HOUR",
                    "_1_HOUR_TO_3_HOURS",
                    "OVER_3_HOURS",
                ],
                "protocol": ["UDP", "TCP", "ICMP", "GRE"],
                "default": ["values"],
            }
            if "bitrate" in TITLE:
                group = "bitrate"
            elif "duration" in TITLE:
                group = "duration"
            elif "protocol" in TITLE:
                group = "protocol"
            else:
                group = "default"
            data = resp.json()
            main = data.get("result", {}).get("main", {})
            
            if TITLE == "httpreq_time":
                timestamps = main.get("timestamps") or []
                region_results["timestamps"] = timestamps
            
            region_results.update({
                key: (main.get(key) or [])
                for key in FIELD_MAP[group]
            })
        except Exception as e:
            print(f"[ERROR] JSON decode error for {params['dateStart']}:", e)
    else:
        print(f"HTTP {resp.status_code} for {params['dateStart']}")
        return

    return region_results


#########################################
##             MAIN FETCH              ##
#########################################

def run_fetch(country: str, date_from: datetime, date_to: datetime) -> dict:
    """
    Fetches all datasets for AE in memory and returns timestamps and values only.
        ASSUMES:
        - DATE_FROM: incl. start e.g. (2024-11-15); automatically sets time to 00:00Z 
        - DATE_TO: incl. end e.g. (2024-12-15); automatically sets time to 23:45
    """
    print(f"[INFO] Fetching data for {country}...")
    results = {}
    
    params = {"name": "main", "location": country}
    TITLE = "httpreq_time"
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    TITLE = "traffic_time"
    URL="https://api.cloudflare.com/client/v4/radar/netflows/timeseries"
    results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    TITLE = "aibots_crawlers_time"
    URL = "https://api.cloudflare.com/client/v4/radar/ai/bots/timeseries"
    results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    TITLE = "bots_time"
    URL = "https://api.cloudflare.com/client/v4/radar/bots/timeseries"
    results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    TITLE = f"l7attack_time"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries"
    results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    BOT_CLASS = ["Likely_Automated", "Likely_Human"]
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    for botcl in BOT_CLASS:
        TITLE = f"httpreq_{botcl.replace('Likely_', '').lower()}_time"
        params = {"name": "main", "location": country, "botClass": botcl}
        results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    DIRECTION = ["Origin", "Target"]
    for dir in DIRECTION:
        params = {"name": "main", "location": country, "direction": dir}
        TITLE = f"l3attack_{dir.lower()}_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries"
        results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
        if dir == "Origin":
            TITLE = f"l3attack_{dir.lower()}_bitrate_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/bitrate"
            results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
            TITLE = f"l3attack_{dir.lower()}_duration_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/duration"
            results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
            TITLE = f"l3attack_{dir.lower()}_protocol_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/protocol"
            results[TITLE] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
    
    return results

#########################################
##             HELPER BUILD            ##
#########################################

def _conf_todf(data: dict, key: str, rename: str) -> pd.DataFrame:
    """Converts raw data to internal pd.DataFrame format consistent with model train/val/tune/test nomenclature."""
    values = list(data[key]["values"])
    ts = pd.to_datetime(data["timestamps"], errors="coerce")
    TS_RANGE = 96
    
    if len(ts) == len(values):
        pass
    elif len(ts) > len(values):
        diff = len(ts) - len(values)
        values = values + [0.0] * diff
    elif len(ts) < len(values):
        print(f"[WARN] Not enough ts data fetched for {key}.")
        start = pd.to_datetime(ts[0]).floor("D")
        ts = pd.date_range(start, periods=TS_RANGE, freq="15min")
        if len(values) < TS_RANGE:
            values = values + [0.0] * (TS_RANGE - len(values))
        else:
            values = values[:TS_RANGE]
    else:
        raise ValueError(f"[ERROR] No data fetched for {key}. Aborting now.")
    
    return pd.DataFrame(
            data={rename: values}, 
            index=ts
        ).astype("float64")
                                          
def _conv_weighted_todf(data: dict, key: str, rename: str, mids: dict) -> pd.Series:
    """Converts raw data which are value distributions to internal pd.DataFrame format which are weighted averages per timestamp consistent with model train/val/tune/test nomenclature."""
    ts = pd.to_datetime(data.get("timestamps", []), errors="coerce")
    cols = [c for c in mids.keys() if c in data[key]]
    if len(ts) == 0 or not cols:
        raise ValueError(f"[ERROR] No data fetched for {key}. Aborting now.")
    
    X = np.column_stack([
        np.asarray(data[key][c], dtype=float)
        for c in cols
    ])
    W = np.asarray([mids[c] for c in cols], dtype=float)
    den = X.sum(axis=1)
    den = np.where(den <= 0, np.nan, den)
    weighted = (X * W).sum(axis=1) / den
    weighted = np.nan_to_num(weighted, nan=0.0)

    return pd.Series(weighted, index=ts, name=rename, dtype="float64")

def _conv_fract_todf(data: dict, key: str) -> pd.DataFrame:
    """Converts raw data which are protocol bucket data to internal pd.DataFrame format which are fractional shares and Shannon entropy per timestamp consistent with model train/val/tune/test nomenclature."""
    ts = pd.to_datetime(data["timestamps"], errors="coerce")
    df = pd.DataFrame({
        "udp":  np.array(data[key].get("UDP",  []),  dtype=float),
        "tcp":  np.array(data[key].get("TCP",  []),  dtype=float),
        "icmp": np.array(data[key].get("ICMP", []), dtype=float),
        "gre":  np.array(data[key].get("GRE",  []),  dtype=float),
    }, index=ts).astype("float64")

    # Total traffic per timestamp and fractions
    pcols = ["udp", "tcp", "icmp", "gre"]
    df["total"] = df[pcols].sum(axis=1).replace(0, 1e-6)
    for c in pcols:
        df[f"{c}_frac"] = df[c] / df["total"]

    # Shannon entropy
    P = df[[f"{c}_frac" for c in pcols]].to_numpy()
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(axis=1, keepdims=True)
    entropy = -(P * np.log2(P)).sum(axis=1)

    return pd.DataFrame({
        "udp_frac": df["udp_frac"],
        "tcp_frac": df["tcp_frac"],
        "icmp_frac": df["icmp_frac"],
        "gre_frac": df["gre_frac"],
        "protocol_entropy": entropy,
    }, index=ts).astype("float64")


#########################################
##              MAIN BUILD             ##
#########################################

def run_build(country: str, data: dict) -> pd.DataFrame:
    """Builds raw feature matrix as df for a given country."""
    print(f"[INFO] Building raw feature matrix for {country}...")
    # ------------------------------------
    # 1. load all base series
    # ------------------------------------
    s_l3o =  _conf_todf(data, "l3attack_origin_time", "l3_origin")
    s_l3t = _conf_todf(data, "l3attack_target_time", "l3_target") 
    s_l7 = _conf_todf(data, "l7attack_time", "l7_traffic")
    s_http = _conf_todf(data, "httpreq_time", "http")
    s_http_auto = _conf_todf(data, "httpreq_automated_time", "http_auto")
    s_http_human = _conf_todf(data, "httpreq_human_time", "http_human")
    s_netflow = _conf_todf(data, "traffic_time", "netflow")
    s_bots = _conf_todf(data, "bots_time", "bots_total")
    s_ai = _conf_todf(data, "aibots_crawlers_time", "ai_bots")

    bitrate_mids = {
        "UNDER_500_MBPS": 250,
        "_500_MBPS_TO_1_GBPS": 750,
        "_1_GBPS_TO_10_GBPS": 5500,
        "_10_GBPS_TO_100_GBPS": 55000,
        "OVER_100_GBPS": 100000
    }
    s_l3_bitrate = _conv_weighted_todf(data, "l3attack_origin_bitrate_time", "l3_bitrate_avg", bitrate_mids)
    dur_mids = {
        "UNDER_10_MINS": 5,
        "_10_MINS_TO_20_MINS": 15,
        "_20_MINS_TO_40_MINS": 30,
        "_40_MINS_TO_1_HOUR": 50,
        "_1_HOUR_TO_3_HOURS": 120,
        "OVER_3_HOURS": 300
    }
    s_l3_duration = _conv_weighted_todf(data, "l3attack_origin_duration_time", "l3_duration_avg", dur_mids)
    
    s_protocol = _conv_fract_todf(data, "l3attack_origin_protocol_time")

    # ------------------------------------
    # 2. merge everything
    # ------------------------------------
    df = pd.concat([
            s_l3o, s_l3t, s_l7,
            s_http, s_http_auto, s_http_human,
            s_netflow,
            s_bots, s_ai,
            s_l3_bitrate, s_l3_duration,
        ], axis=1)
    df = df.join(s_protocol, how="outer")
    df = df.sort_index().interpolate().ffill().bfill()

    # ------------------------------------
    # 3. derived ratios
    # ------------------------------------
    eps = 1e-6
    df["ratio_l3_l7"] = df["l3_origin"] / (df["l7_traffic"] + eps)
    df["ratio_auto_human"] = df["http_auto"] / (df["http_human"] + eps)
    df["ratio_bots_http"] = df["bots_total"] / (df["http"] + eps)
    df["ratio_ai_bots_bots"] = df["ai_bots"] / (df["bots_total"] + eps)
    df["ratio_netflow_http"] = df["netflow"] / (df["http"] + eps)

    # ------------------------------------
    # 4a. local-time daytype + daytime
    # ------------------------------------
    iso_series = df.index.to_series().dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Daytype / weekday
    dt_df = conv_iso_to_local_with_daytype(iso_series, country, timezones)
    dt_df["daytype_bin"] = dt_df["daytype"].map({"Weekday": 0, "Weekend": 1}).astype("int64")

    # Daytime buckets
    daytimes = conv_iso_to_local_with_daytimes(iso_series, country, timezones)
    daytime_map = {
        "Deep night": 0,
        "Morning": 1,
        "Business hours": 2,
        "Evening": 3,
        "Early night": 4,
        "Unknown": 5,
    }
    daytimes["daytime_bin"] = daytimes["daytime"].map(daytime_map).astype("int64")

    df["weekday_idx"] = dt_df["weekday"].astype("int64")
    df["daytype_idx"] = dt_df["daytype_bin"]
    df["daytime_idx"] = daytimes["daytime_bin"]

    # ------------------------------------
    # 4b. month + week periodic encodings
    # ------------------------------------
    idx_local = conv_iso_to_local(iso_series, country, timezones)

    df["month_idx"] = idx_local.dt.month - 1
    df["week_idx"] = idx_local.dt.isocalendar().week.astype(int) - 1

    # ------------------------------------
    # 4c. time cyclic encoding
    # ------------------------------------
    # local hour-of-day and weekday cyclic
    df["hour_sin"] = np.sin(2 * np.pi * idx_local.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx_local.dt.hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["weekday_idx"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday_idx"] / 7)

    # month cyclic
    df["month_sin"] = np.sin(2 * np.pi * df["month_idx"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_idx"] / 12)

    # week cyclic
    df["week_sin"] = np.sin(2 * np.pi * df["week_idx"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_idx"] / 52)

    # ------------------------------------
    # 5. rolling aggregates
    # ------------------------------------
    df = df.sort_index()
    df = df.asfreq("h")

    roll_cols = ["l3_origin", "l3_target", "l7_traffic",
                 "http", "http_auto", "http_human",
                 "netflow", "bots_total", "ai_bots"]

    for col in roll_cols:
        df[f"{col}_roll3h"] = df[col].rolling(3).mean()
        df[f"{col}_roll24h"] = df[col].rolling(24).mean()

    df = df.bfill()
    return df

#########################################
##               MAIN RUN              ##
#########################################

def run_fetcher(country: str, date_from: datetime, date_to: datetime, scaler: TransformerMixin) -> tuple[pd.DataFrame, pd.DataFrame, int, dict, TransformerMixin]:
    """Fetches and processes raw data to return feature matrix objects ready for inference."""
    object = run_fetch(country, date_from, date_to)
    if "httpreq_time" in object:
        block = object["httpreq_time"]
        if isinstance(block, dict) and "timestamps" in block:
            object["timestamps"] = block.pop("timestamps")
    
    raw_df = run_build(country, object)
    
    X_cont, X_cat, num_cont, cat_dims, scaler = build_feature_matrix(country, raw_df, scaler)
    print(f"[INFO] Feature matrix for {country} ready.")

    return X_cont, X_cat, num_cont, cat_dims, scaler


if __name__=="__main__": 

    country = "US"
    DATE_FROM = datetime(2025, 11, 14, tzinfo=timezone.utc)
    DATE_TO   = datetime(2025, 11, 14, tzinfo=timezone.utc)
    run_fetcher(country, DATE_FROM, DATE_TO)