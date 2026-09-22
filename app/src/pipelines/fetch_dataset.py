#!/usr/bin/env python3
"""
Fetches raw datasets from Cloudflare.
- outputs json files

Output:
    raw files : app/datasets/raw/<dataset>.json

Usage: 
    python -m app.src.data.fetching.fetch [-S <date_str>] [-E <date_str>] [-A] [-T] [-N] [-hr] [-hrt] [-hra] [-t] [-tt] [-at] [-bt] [-iqt] [-a] [-l3or] [-l3ta] [-l3ort] [-l3tat] [-l7or] [-l7ta] [l7t] 
"""
import argparse, sys
from datetime import datetime, timezone

from app.src.data.fetching.fetch import (
    pull_notime_data,
    pull_notime_datapercountry,
    pull_time_datapercountry
)

def fetch_dataset(args: argparse.Namespace) -> None:
    start, end = args.start, args.end 
    try:
        start_dt = datetime.strptime(start, "%m/%d/%Y").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end, "%m/%d/%Y").replace(tzinfo=timezone.utc)
        if start_dt and end_dt:
            DATES = (start_dt, end_dt)
    except Exception:
        print(f"Date format not accepted: {start} - {end}")
        print("Required format: MM/DD/YYYY")
        sys.exit(1)
    
    if args.all or args.notime or args.httpreq:
        TITLE = "httpreq"
        URL = "https://api.cloudflare.com/client/v4/radar/http/top/locations"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_data(TITLE, URL, params, DATES)

    if args.all or args.time or args.httpreq_time:
        TITLE = "httpreq_time"
        URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.httpreq_automated:
        BOT_CLASS = ["Likely_Automated", "Likely_Human"]
        URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
        for botcl in BOT_CLASS:
            TITLE = f"httpreq_{botcl.replace('Likely_', '').lower()}_time"
            params = {"name": "main", "botClass": botcl}
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.notime or args.traffic:
        TITLE = "traffic"
        URL = "https://api.cloudflare.com/client/v4/radar/netflows/top/locations"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_data(TITLE, URL, params, DATES)

    if args.all or args.time or args.traffic_time:
        TITLE = "traffic_time"
        URL="https://api.cloudflare.com/client/v4/radar/netflows/timeseries"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.aibots_time:
        TITLE = "aibots_crawlers_time"
        URL = "https://api.cloudflare.com/client/v4/radar/ai/bots/timeseries"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.bots_time:
        TITLE = "bots_time"
        URL = "https://api.cloudflare.com/client/v4/radar/bots/timeseries"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.iq_time:
        URL = "https://api.cloudflare.com/client/v4/radar/quality/iqi/timeseries_groups"
        METRICS = ["bandwidth","dns","latency"]
        for metric in METRICS:
            TITLE = f"inetqal_{metric}_time"
            params = {"name": "main", "metric": metric}
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.notime or args.l3_origin:
        TITLE = "l3attack_origin"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/top/locations/origin"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.notime or args.l3_target:
        TITLE = "l3attack_target"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/top/locations/target"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.l3_origin_time or args.l3_target_time:
        if args.all or args.time:
            DIRECTION = ["Origin", "Target"]
        elif args.l3_origin_time:
            DIRECTION = ["Origin"]
        elif args.l3_target_time:
            DIRECTION = ["Target"]
        for dir in DIRECTION:
            params = {"name": "main", "direction": dir}
            TITLE = f"l3attack_{dir.lower()}_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries"
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)
            TITLE = f"l3attack_{dir.lower()}_bitrate_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/bitrate"
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)
            TITLE = f"l3attack_{dir.lower()}_duration_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/duration"
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)
            TITLE = f"l3attack_{dir.lower()}_protocol_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/protocol"
            print(f"[INFO] Fetching data {TITLE}...")
            pull_time_datapercountry(TITLE, URL, params, DATES)
    
    if args.all or args.notime or args.l7_origin:
        TITLE = "l7attack_origin"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/top/locations/origin"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.notime or args.l7_target:
        TITLE = "l7attack_target"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/top/locations/target"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.time or args.l7_time:
        TITLE = f"l7attack_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

        TITLE = f"l7attack_mitigations_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries_groups/mitigation_product"
        params = {"name": "main"}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_time_datapercountry(TITLE, URL, params, DATES)

    if args.all or args.notime or args.anomalies:
        TITLE = "anomalies"
        URL = "https://api.cloudflare.com/client/v4/radar/traffic_anomalies"
        params = {"name": "main", "limit": 200}
        print(f"[INFO] Fetching data {TITLE}...")
        pull_notime_data(TITLE, URL, params, DATES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Data from Cloudflare Radar API"
    )

    parser.add_argument(
        "-S", "--start",
        type=str, 
        help="start date of fetch range in format MM/DD/YYYY"
    )

    parser.add_argument(
        "-E", "--end",
        type=str, 
        help="end date of fetch range in format MM/DD/YYYY"
    )

    parser.add_argument(
        "-A", "--all",
        action="store_true",
        help="fetch all datasets"
    )

    parser.add_argument(
        "-T", "--time",
        action="store_true",
        help="fetch all timeseries datasets"
    )

    parser.add_argument(
        "-N", "--notime",
        action="store_true",
        help="fetch all none timeseries datasets"
    )

    parser.add_argument(
        "-hr", "--httpreq",
        action="store_true",
        help="fetch HTTP request traffic intensities"
    )

    parser.add_argument(
        "-hrt", "--httpreq-time",
        action="store_true",
        help="fetch HTTP request traffic timeseries"
    )

    parser.add_argument(
        "-hra", "--httpreq-automated",
        action="store_true",
        help="fetch HTTP request traffic data dissected in likely automated and likely human"
    )

    parser.add_argument(
        "-t", "--traffic",
        action="store_true",
        help="fetch Netflow traffic data in bytes"
    )

    parser.add_argument(
        "-tt", "--traffic-time",
        action="store_true",
        help="fetch Netflow traffic data in bytes as timeseries"
    )

    parser.add_argument(
        "-at", "--aibots-time",
        action="store_true",
        help="fetch ai-bots and crawlers timeseries"
    )

    parser.add_argument(
        "-bt", "--bots-time",
        action="store_true",
        help="fetch bots timeseries"
    )

    parser.add_argument(
        "-iqt", "--iq-time",
        action="store_true",
        help="fetch internet quality timeseries dissected in dns, latency, bandwidth"
    )

    parser.add_argument(
        "-a", "--anomalies",
        action="store_true",
        help="fetch anomaly data"
    )

    parser.add_argument(
        "-l3or", "--l3-origin",
        action="store_true",
        help="fetch l3 attack origin ranking per country"
    )

    parser.add_argument(
        "-l3ta", "--l3-target",
        action="store_true",
        help="fetch l3 attack target ranking per country"
    )

    parser.add_argument(
        "-l3ort", "--l3-origin-time",
        action="store_true",
        help="fetch l3 attack origin timeseries bundle with raw timeseries, bitrate, duration and protocol"
    )

    parser.add_argument(
        "-l3tat", "--l3-target-time",
        action="store_true",
        help="fetch l3 attack target timeseries bundle with raw timeseries, bitrate, duration and protocol"
    )

    parser.add_argument(
        "-l7or", "--l7-origin",
        action="store_true",
        help="fetch l7 attack origin ranking per country"
    )

    parser.add_argument(
        "-l7ta", "--l7-target",
        action="store_true",
        help="fetch l7 attack target ranking per country"
    )

    parser.add_argument(
        "-l7t", "--l7-time",
        action="store_true",
        help="fetch l7 attack timeseries bundle with raw timeseries and mitigations"
    )

    args = parser.parse_args()

    fetch_dataset(args)