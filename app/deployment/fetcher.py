from datetime import datetime, timezone
from app.src.data.fetch import _headers, _requests_session


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
    
    if "httpreq_time" in results:
        block = results["httpreq_time"]
        if isinstance(block, dict) and "timestamps" in block:
            results["timestamps"] = block.pop("timestamps")

    return results


if __name__=="__main__": 

    country = "US"
    DATE_FROM = datetime(2025, 11, 14, tzinfo=timezone.utc)
    DATE_TO   = datetime(2025, 11, 14, tzinfo=timezone.utc)
    run_fetch(country, DATE_FROM, DATE_TO)