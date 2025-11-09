#!/usr/bin/env python3

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import time
from dateutil.relativedelta import relativedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os


#########################################
##                CONFIG               ##
#########################################

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("API_TOKEN not found in environment (.env)")


#########################################
##                PARAMS               ##
#########################################

ISO_3166_alpha2 = [
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM", "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW", "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW"] # added Kosovo = XK for CF


#########################################
##               FETCHER               ##
#########################################

def _get_output_path(title: str):
    PULL_DATE = datetime.now(timezone.utc).strftime("%m-%d-%Y")
    output_dir = "./datasets"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{title}_pull-{PULL_DATE}.json")


def pull_notime_data(TITLE, URL, PARAMS):
    PULL_DATE = "11-6-2025"
    HEADERS = {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "application/json",
    }
    DATE_RANGE_MIN = datetime.fromisoformat("2025-07-02T00:00:00.000Z".replace("Z", "+00:00"))
    DATE_RANGE_MAX = datetime.fromisoformat("2025-11-05T00:00:00.000Z".replace("Z", "+00:00"))

    output_path = _get_output_path(TITLE)

    results = []
    current_end = DATE_RANGE_MAX
    idx = 1

    while current_end > DATE_RANGE_MIN:
        current_start = current_end - timedelta(days=1)

        date_end_str = current_end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        date_start_str = current_start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

        PARAMS["dateStart"] = date_start_str
        PARAMS["dateEnd"] = date_end_str

        resp = requests.get(URL, headers=HEADERS, params=PARAMS)
        print(f"[{idx:02}] {date_start_str} → {date_end_str} | status = {resp.status_code}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                #print("Pulled no. of countries:", len(data["result"]["main"]))
                results.append({
                    "idx": idx,
                    "fetch": {
                        "start": date_start_str,
                        "end": date_end_str,
                        "value": data
                    }
                })
            except Exception as e:
                print(f"JSON decode error for {date_start_str}: {e}")
        else:
            print(f"HTTP {resp.status_code} for {date_start_str}")

        current_end = current_start
        idx += 1

    results.sort(key=lambda x: x["fetch"]["start"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} days to {output_path}")


def pull_notime_datapercountry(TITLE, URL, PARAMS, COUNTRIES=ISO_3166_alpha2):
    PULL_DATE = "11-6-2025"
    HEADERS = {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "application/json",
    }

    DATE_RANGE_MIN = datetime.fromisoformat("2025-07-02T00:00:00.000Z".replace("Z", "+00:00"))
    DATE_RANGE_MAX = datetime.fromisoformat("2025-11-05T00:00:00.000Z".replace("Z", "+00:00"))

    output_path = _get_output_path(TITLE)

    final_results = {}

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch_for_region(region_name, PARAMS):
        print(f"\nFetching {region_name} ...")
        results = []
        current_end = DATE_RANGE_MAX
        idx = 1

        while current_end > DATE_RANGE_MIN:
            current_start = current_end - timedelta(days=1)

            date_start_str = current_start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            date_end_str = current_end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

            PARAMS["dateStart"] = date_start_str
            PARAMS["dateEnd"] = date_end_str

            try:
                resp = session.get(URL, headers=HEADERS, params=PARAMS, timeout=30)
            except requests.exceptions.SSLError as e:
                print(f"SSL error for {region_name} {date_start_str}: {e}, retrying after 2s")
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                break
            
            print(f"  [{idx:02}] {date_start_str} → {date_end_str} | status = {resp.status_code}")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print("Pulled no. of countries:", len(data["result"]["main"]))
                    results.append({
                        "idx": idx,
                        "fetch": {
                            "start": date_start_str,
                            "end": date_end_str,
                            "value": data
                        }
                    })
                except Exception as e:
                    print(f"JSON decode error for {date_start_str}: {e}")
            else:
                print(f"HTTP {resp.status_code} for {date_start_str}")

            current_end = current_start
            idx += 1

        results.sort(key=lambda x: x["fetch"]["start"], reverse=True)
        final_results[region_name] = results
        print(f"{region_name}: saved {len(results)} days")

    fetch_for_region("worldwide", dict(PARAMS))

    for code in COUNTRIES:
        region_params = dict(PARAMS)
        region_params["location"] = code
        fetch_for_region(code, region_params)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! Saved worldwide + {len(COUNTRIES)} countries to {output_path}")


def pull_time_datapercountry(TITLE, URL, PARAMS, COUNTRIES=ISO_3166_alpha2):
    PULL_DATE = "11-6-2025"
    HEADERS = {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "application/json",
    }

    DATE_RANGE_MIN = datetime.fromisoformat("2025-07-02T00:00:00.000Z".replace("Z", "+00:00"))
    DATE_RANGE_MAX = datetime.fromisoformat("2025-11-04T23:45:00.000Z".replace("Z", "+00:00"))

    output_path = _get_output_path(TITLE)

    final_results = {}

    def fetch_for_region(region_name, params):
        print(f"\nFetching {region_name} ...")
        results = []
        current_end = DATE_RANGE_MAX
        idx = 1

        while current_end > DATE_RANGE_MIN:
            current_start = (current_end - relativedelta(months=1)).replace(hour=0, minute=0)

            if current_start < DATE_RANGE_MIN:
                current_start = DATE_RANGE_MIN

            date_start_str = current_start.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            effective_end = current_end - timedelta(days=1)
            date_end_str  = effective_end.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

            params["dateStart"] = date_start_str
            params["dateEnd"]   = date_end_str

            resp = requests.get(URL, headers=HEADERS, params=params)
            print(f"  [{idx:02}] {date_start_str} → {date_end_str} | status = {resp.status_code}")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print("Pulled no. of 1h-buckets:", len(data["result"]["main"]["timestamps"]))
                    results.append({
                        "idx": idx,
                        "fetch": {
                            "start": date_start_str,
                            "end": date_end_str,
                            "value": data
                        }
                    })
                except Exception as e:
                    print(f"JSON decode error for {date_start_str}: {e}")
            else:
                print(f"HTTP {resp.status_code} for {date_start_str}")

            current_end = current_start - timedelta(minutes=15)
            idx += 1

        results.sort(key=lambda x: x["fetch"]["start"], reverse=True)
        final_results[region_name] = results
        print(f"{region_name}: saved {len(results)} months")

    fetch_for_region("worldwide", dict(PARAMS))

    for code in COUNTRIES:
        region_params = dict(PARAMS)
        region_params["location"] = code
        fetch_for_region(code, region_params)

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! Saved worldwide + {len(COUNTRIES)} countries to {output_path}")


if __name__ == "__main__":
    
    #--- HTTP reqests
    TITLE = "httpreq"
    URL = "https://api.cloudflare.com/client/v4/radar/http/top/locations"
    params = {"name": "main", "limit": 250}
    pull_notime_data(TITLE, URL, params)

    TITLE = "httpreq_time"
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    BOT_CLASS = ["Likely_Automated", "Likely_Human"]
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    for botcl in BOT_CLASS:
        TITLE = f"httpreq_{botcl.replace('Likely_', '').lower()}_time"
        params = {"name": "main", "botClass": botcl}
        pull_time_datapercountry(TITLE, URL, params)
    
    #--- Netflow traffic
    TITLE = "traffic"
    URL = "https://api.cloudflare.com/client/v4/radar/netflows/top/locations"
    params = {"name": "main", "limit": 200}
    pull_notime_data(TITLE, URL, params)

    TITLE = "traffic_time"
    URL="https://api.cloudflare.com/client/v4/radar/netflows/timeseries"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    #--- AI bots and crawlers
    TITLE = "aibots_crawlers_time"
    URL = "https://api.cloudflare.com/client/v4/radar/ai/bots/timeseries"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    #--- Bots
    TITLE = "bots_time"
    URL = "https://api.cloudflare.com/client/v4/radar/bots/timeseries"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    #--- Internet quality
    URL = "https://api.cloudflare.com/client/v4/radar/quality/iqi/timeseries_groups"
    METRICS = ["bandwidth","dns","latency"]
    for metric in METRICS:
        TITLE = f"inetqal_{metric}_time"
        params = {"name": "main", "metric": metric}
        pull_time_datapercountry(TITLE, URL, params)

    #--- Nework L3 attacks
    TITLE = "l3attack_origin"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/top/locations/origin"
    params = {"name": "main", "limit": 250}
    pull_notime_datapercountry(TITLE, URL, params)

    TITLE = "l3attack_target"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/top/locations/target"
    params = {"name": "main", "limit": 250}
    pull_notime_datapercountry(TITLE, URL, params)

    DIRECTION = ["Origin", "Target"]
    for dir in DIRECTION:
        params = {"name": "main", "direction": dir}
        TITLE = f"l3attack_{dir.lower()}_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries"
        pull_time_datapercountry(TITLE, URL, params)
        TITLE = f"l3attack_{dir.lower()}_bitrate_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/bitrate"
        pull_time_datapercountry(TITLE, URL, params)
        TITLE = f"l3attack_{dir.lower()}_duration_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/duration"
        pull_time_datapercountry(TITLE, URL, params)
        TITLE = f"l3attack_{dir.lower()}_protocol_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/protocol"
        pull_time_datapercountry(TITLE, URL, params)

    #--- WAF L7 attacks
    TITLE = "l7attack_origin"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/top/locations/origin"
    params = {"name": "main", "limit": 200}
    pull_notime_datapercountry(TITLE, URL, params)

    TITLE = "l7attack_target"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/top/locations/target"
    params = {"name": "main", "limit": 200}
    pull_notime_datapercountry(TITLE, URL, params)

    TITLE = f"l7attack_time"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    TITLE = f"l7attack_mitigations_time"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries_groups/mitigation_product"
    params = {"name": "main"}
    pull_time_datapercountry(TITLE, URL, params)

    #--- Traffic anomalies
    TITLE = "anomalies"
    URL = "https://api.cloudflare.com/client/v4/radar/traffic_anomalies"
    params = {"name": "main", "limit": 250}
    pull_notime_data(TITLE, URL, params)

