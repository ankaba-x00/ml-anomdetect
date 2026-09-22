import sys
from datetime import datetime, timezone
from typing import Any

from . import RegionTimeseriesFetchResult
from app.src.data.processing import FIELD_MAP
from app.src.data.fetching.fetch import _headers, _requests_session


def _fetch_timedata(
    TITLE: str, 
    URL: str, 
    BASE_PARAMS: dict[str, Any], 
    date_from: datetime, 
    date_to: datetime
) -> dict[str, list[str]]:
    """
    Fetches timeseries data in memory.
    
    ASSUMES
    =======
        - date_from: incl. start e.g. (2024-11-15); sets time to 00:00Z 
        - date_to: incl. end e.g. (2024-12-15); sets time to 00:00Z next day
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
        print(f"[ERROR] HTTP {resp.status_code} for {params['dateStart']}")
        sys.exit(1)

    return region_results

def run_fetch(
    country: str, 
    date_from: datetime, 
    date_to: datetime
) -> RegionTimeseriesFetchResult:
    """
    Fetches timeseries datasets from Cloudflare API in memory for specified 
    country in 15-min buckets.
    """
    print(f"[INFO] Fetching data for {country}...")
    
    results: dict[str, dict[str, list[str]] | list[str]] = {}
    
    params = {"name": "main", "location": country}
    TITLE = "httpreq_time"
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    results[TITLE[:-5]] = _fetch_timedata(TITLE, URL, params, date_from, date_to)

    TITLE = "traffic_time"
    URL="https://api.cloudflare.com/client/v4/radar/netflows/timeseries"
    results[TITLE[:-5]] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]

    TITLE = "aibots_crawlers_time"
    URL = "https://api.cloudflare.com/client/v4/radar/ai/bots/timeseries"
    results[TITLE[:-5]] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]

    TITLE = "bots_time"
    URL = "https://api.cloudflare.com/client/v4/radar/bots/timeseries"
    results[TITLE[:-5]] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]

    TITLE = f"l7attack_time"
    URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer7/timeseries"
    results["l7"] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]

    BOT_CLASS = ["Likely_Automated", "Likely_Human"]
    URL="https://api.cloudflare.com/client/v4/radar/http/timeseries"
    for botcl in BOT_CLASS:
        TITLE = f"httpreq_{botcl.replace('Likely_', '').lower()}_time"
        params = {"name": "main", "location": country, "botClass": botcl}
        results[TITLE[:-5]] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]

    DIRECTION = ["Origin", "Target"]
    for dir in DIRECTION:
        params = {"name": "main", "location": country, "direction": dir}
        TITLE = f"l3attack_{dir.lower()}_time"
        URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries"
        results[f"l3_{dir.lower()}"] = _fetch_timedata(TITLE, URL, params, date_from, date_to)["values"]
        if dir == "Origin":
            TITLE = f"l3attack_{dir.lower()}_bitrate_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/bitrate"
            results[f"l3_{dir.lower()}_bitrate"] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
            TITLE = f"l3attack_{dir.lower()}_duration_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/duration"
            results[f"l3_{dir.lower()}_duration"] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
            TITLE = f"l3attack_{dir.lower()}_protocol_time"
            URL = "https://api.cloudflare.com/client/v4/radar/attacks/layer3/timeseries_groups/protocol"
            results[f"l3_{dir.lower()}_protocol"] = _fetch_timedata(TITLE, URL, params, date_from, date_to)
    
    if "httpreq" in results:
        block = results["httpreq"]
        if isinstance(block, dict) and "timestamps" in block:
            results["timestamps"] = block.pop("timestamps")
            results["httpreq"] = block["values"]

    return RegionTimeseriesFetchResult.from_dict(results)


if __name__=="__main__": 

    country = "US"
    DATE_FROM = datetime(2026, 2, 14, tzinfo=timezone.utc)
    DATE_TO   = datetime(2026, 2, 14, tzinfo=timezone.utc)
    run_fetch(country, DATE_FROM, DATE_TO)
