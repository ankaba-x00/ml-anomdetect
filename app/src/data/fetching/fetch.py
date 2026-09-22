import requests, json, os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from requests.adapters import HTTPAdapter
from typing import Any
from urllib3.util.retry import Retry


env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(env_path)
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("[ERROR] API_TOKEN not found in environment (.env)")

ISO_3166_alpha2 = [
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", 
    "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", 
    "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY", 
    "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", 
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", 
    "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", 
    "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL", 
    "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM", 
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", 
    "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", 
    "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS", 
    "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK", 
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", 
    "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", 
    "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", 
    "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW", 
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", 
    "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", 
    "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", 
    "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", 
    "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW"] # added Kosovo = XK for CF


def _get_output_path(title: str) -> Path:
    """
    Generates output directory enabling multi-pull fetches with no data loss.
    """
    pull_date = datetime.now(timezone.utc).strftime("%m-%d-%Y")
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "datasets" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{title}_pull-{pull_date}.json"

def _headers() -> dict[str, str]:
    """Output header dictionary."""
    global API_TOKEN
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "application/json",
    }

def _requests_session() -> requests.Session:
    """Creates session with retry."""
    session = requests.Session()
    retries = Retry(
        total=5,
        allowed_methods=["GET"],
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def _print_range(dates: tuple[datetime, ...]) -> None:
    """Prints date range of fetch in stdout."""
    print(
        f"[INFO] Data will be fetched" \
        "\n\t FROM incl. {dates[0].strftime('%m/%d/%Y')}" \
        "\n\t TO incl. {dates[1].strftime('%m/%d/%Y')}\n"
    )

def pull_notime_data(
    TITLE: str, 
    URL: str, 
    BASE_PARAMS: dict[str, Any], 
    dates: tuple[datetime, ...]
) -> None:
    """
    Fetches dataseries with no specific location from Cloudflare API in 1-day 
    buckets (ergo worldwide and no timeseries).

    ASSUMES
    =======
        - dates[0]: incl. start e.g. (2024-11-15); sets time to 00:00Z 
        - dates[1]: incl. end e.g. (2024-12-15); sets time to 00:00Z next day
    """
    _print_range(dates)
    DATE_MIN, DATE_MAX = dates[0], dates[1] + timedelta(days=1)
    
    output_path = _get_output_path(TITLE)
    results = []
    session = _requests_session()
    current_start = DATE_MIN
    idx = 1

    while current_start < DATE_MAX:
        next_start = current_start + timedelta(days=1)
        window_end = min(next_start, DATE_MAX)

        params = dict(BASE_PARAMS)
        params["dateStart"] = current_start.isoformat().replace("+00:00", "Z")
        params["dateEnd"] = window_end.isoformat().replace("+00:00", "Z")

        resp = session.get(URL, headers=_headers(), params=params, timeout=30)
        print(f"  [{idx:02}] {params['dateStart']} to {params['dateEnd']} | status = {resp.status_code}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                try:
                    print("Pulled:", len(data["result"].get("main", [])))
                except:
                    pass
                results.append({
                    "idx": idx,
                    "fetch": {
                        "start": params["dateStart"],
                        "end": params["dateEnd"],
                        "value": data
                    }
                })
            except Exception as e:
                print(f"[ERROR] JSON decode error for {params['dateStart']}: {e}")
        else:
            print(f"HTTP {resp.status_code} for {params['dateStart']}")

        current_start = next_start
        idx += 1

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved {len(results)} days to {output_path}")

def pull_notime_datapercountry(
    TITLE: str, 
    URL: str, 
    BASE_PARAMS: dict[str, Any],
    dates: tuple[datetime, ...],
    COUNTRIES: list[str] = ISO_3166_alpha2
) -> None:
    """
    Fetches dataseries for worldwide and per country from Cloudflare API in 
    1-day buckets (no timeseries).

    ASSUMES
    =======
        - dates[0]: incl. start e.g. (2024-11-15); sets time to 00:00Z 
        - dates[1]: incl. end e.g. (2024-12-15); sets time to 00:00Z next day
    """
    _print_range(dates)
    DATE_MIN, DATE_MAX = dates[0], dates[1] + timedelta(days=1)
    
    output_path = _get_output_path(TITLE)
    results_all = {}
    session = _requests_session()

    def fetch_for_region(
        region_name: str, 
        params_base: dict[str, str | int]
    ) -> None:
        print(f"[INFO] Fetching {region_name}...")
        region_results = []
        current_start = DATE_MIN
        idx = 1

        while current_start < DATE_MAX:
            next_start = current_start + timedelta(days=1)
            window_end = min(next_start, DATE_MAX)

            params = dict(params_base)
            params["dateStart"] = current_start.isoformat().replace("+00:00", "Z")
            params["dateEnd"] = window_end.isoformat().replace("+00:00", "Z")

            try:
                resp = session.get(URL, headers=_headers(), params=params, timeout=30)
            except Exception as e:
                print("[ERROR] Request error:", e)
                break
            
            print(f"  [{idx:02}] {params['dateStart']} to {params['dateEnd']} | status = {resp.status_code}")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    try:
                        main = data.get("result", {}).get("main") or []
                        print("Pulled no. of countries:", len(main))
                    except:
                        pass
                    region_results.append({
                        "idx": idx,
                        "fetch": {
                            "start": params["dateStart"],
                            "end": params["dateEnd"],
                            "value": data
                        }
                    })
                except Exception as e:
                    print(f"JSON decode error for {params['dateStart']}: {e}")
            else:
                print(f"HTTP {resp.status_code} for {params['dateStart']}")

            current_start = next_start
            idx += 1

        results_all[region_name] = region_results
        print(f"[OK] {region_name}: saved {len(region_results)} days")

    fetch_for_region("worldwide", dict(BASE_PARAMS))

    for code in COUNTRIES:
        reg_params = dict(BASE_PARAMS)
        reg_params["location"] = code
        fetch_for_region(code, reg_params)

    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved worldwide + {len(COUNTRIES)} countries to {output_path}")

def pull_time_datapercountry(
    TITLE: str, 
    URL: str, 
    BASE_PARAMS: dict[str, Any],
    dates: tuple[datetime, ...],
    COUNTRIES: list[str] = ISO_3166_alpha2
) -> None:
    """
    Fetches timeseries data for worldwide and per country from Cloudflare API 
    in 15-min buckets.

    ASSUMES
    =======
        - dates[0]: incl. start e.g. (2024-11-15); sets time to 00:00Z 
        - dates[1]: incl. end e.g. (2024-12-15); sets time to 00:00Z next day
    """
    _print_range(dates)
    DATE_MIN, DATE_MAX = dates[0], dates[1].replace(hour=23, minute=45)

    output_path = _get_output_path(TITLE)
    results_all = {}
    session = _requests_session()

    def fetch_region(
        region_name: str, 
        params_base: dict[str, str]
    ) -> None:
        print(f"[INFO] Fetching {region_name} ...")
        region_results: list[dict[str, int | dict[str, Any]]] = []
        current_start = DATE_MIN
        idx = 1

        while current_start < DATE_MAX:
            # natural calendar month boundary + end window 15min before next_start
            next_start = current_start + relativedelta(months=1)
            window_end = next_start - timedelta(minutes=15)
            # clip to the global DATE_MAX end and break if clipped too far
            if window_end > DATE_MAX:
                window_end = DATE_MAX
            if window_end < current_start:
                break

            params = dict(params_base)
            params["dateStart"] = current_start.isoformat().replace("+00:00", "Z")
            params["dateEnd"]   = window_end.isoformat().replace("+00:00", "Z")

            resp = session.get(URL, headers=_headers(), params=params, timeout=30)
            print(f"  [{idx:02}] {params['dateStart']} to {params['dateEnd']} | status = {resp.status_code}")

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    timestamps = (
                        data.get("result", {})
                            .get("main", {})
                            .get("timestamps") or []
                    )
                    print("Pulled no. of 15m-buckets:", len(timestamps))
                    region_results.append({
                        "idx": idx,
                        "fetch": {
                            "start": params["dateStart"],
                            "end":   params["dateEnd"],
                            "value": data
                        }
                    })
                except Exception as e:
                    print(f"[ERROR] JSON decode error for {params['dateStart']}:", e)
            else:
                print(f"HTTP {resp.status_code} for {params['dateStart']}")
                break

            current_start = next_start
            idx += 1
        
        #region_results.sort(key=lambda x: x["fetch"]["start"])
        results_all[region_name] = region_results
        print(f"[OK] {region_name}: saved {len(region_results)} windows")

    fetch_region("worldwide", dict(BASE_PARAMS))

    for code in COUNTRIES:
        p = dict(BASE_PARAMS)
        p["location"] = code
        fetch_region(code, p)

    with open(output_path, "w") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved worldwide +", len(COUNTRIES), "countries to", output_path)
