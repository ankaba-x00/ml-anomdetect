from typing import Union
import pandas as pd
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMEIT] {func.__name__} executed in {end - start:.4f}s")
        return result
    return wrapper

TimeLike = Union[str, pd.Series, pd.Timestamp]

def conv_iso_to_utc(time: TimeLike) -> Union[pd.Timestamp, pd.Series]:
    """Converts ISO 8601 extended date-time format timestamp to UTC time format timestamp."""
    return pd.to_datetime(time, utc=True, errors="coerce")

def conv_utc_to_iso(time: TimeLike) -> Union[str, pd.Series]:
    """Converts UTC time format timestamp to ISO 8601 extended date-time format timestamp."""
    dt = pd.to_datetime(time, utc=True, errors="coerce")

    if isinstance(dt, pd.Series):
        return dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def conv_iso_to_local(
        time: TimeLike, 
        country: str, 
        tz: dict
    ) -> Union[pd.Timestamp, pd.Series]:
    """Converts ISO 8601 extended date-time format timestamp into local timestamp."""
    utc_time = conv_iso_to_utc(time)
    zone = tz[country]["zone"]

    if isinstance(utc_time, pd.Series):
        return utc_time.dt.tz_convert(zone)
    
    return utc_time.tz_convert(zone)

def conv_local_to_iso(
        time: TimeLike, 
        country: str, 
        tz: dict
    ) -> Union[str, pd.Series]:
    """Converts local timestamp to ISO 8601 extended date-time format timestamp."""
    local_tz = tz[country]["zone"]
    dt = pd.to_datetime(time, errors="coerce")

    if isinstance(dt, pd.Series):
        has_tz = dt.dt.tz is not None if not dt.empty else False
        if not has_tz:
            dt = dt.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT", errors="coerce")
        else: 
            dt = dt.dt.tz_convert(local_tz)
        utc_time = dt.dt.tz_convert("UTC")
        return utc_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    if dt.tzinfo is None:
        dt = dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
    else:
        dt = dt.tz_convert(local_tz)
    utc_time = dt.tz_convert("UTC")
    return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

def conv_iso_to_local_with_daytype(
        time: TimeLike, 
        country: str, 
        tz: dict
    ) -> Union[dict, pd.DataFrame]:
    """
    Converts an ISO 8601 timestamp to the country's local time and returns weekday/weekend info.
        Returns:
            dict: {
                "local_time": pandas.Timestamp (tz-aware),
                "weekday": int (0=Monday, 6=Sunday),
                "daytype": str ("weekday" or "weekend")
            }
    """
    utc_time = conv_iso_to_utc(time)
    zone = tz[country]["zone"]

    if isinstance(utc_time, pd.Series):
        local_time = utc_time.dt.tz_convert(zone)
        weekday = local_time.dt.dayofweek
        daytype = weekday.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        return pd.DataFrame({"local_time": local_time, "weekday": weekday, "daytype": daytype})

    try:
        local_time = utc_time.tz_convert(zone)
        weekday = local_time.dayofweek
        daytype = "Weekend" if weekday >= 5 else "Weekday"
        return {"local_time": local_time, "weekday": weekday, "daytype": daytype}
    except Exception:
        return {"local_time": pd.NaT, "weekday": None, "daytype": None}

def conv_iso_to_local_with_daytimes(
        time: TimeLike, 
        country: str, 
        tz: dict  
    ) -> Union[dict, pd.DataFrame]:
    """
    Converts an ISO 8601 timestamp to the country's local time and classifies into a daytime bucket.
        Returns:
            dict or DataFrame:
                local_time : pandas.Timestamp (tz-aware)
                local_hour : int (0–23)
                daytime : str ("Deep night", "Morning", "Business hours", "Evening", "Early night")
    """
    utc_time = conv_iso_to_utc(time)
    zone = tz[country]["zone"]

    if isinstance(utc_time, pd.Series):
        local_time = utc_time.dt.tz_convert(zone)
        local_hour = local_time.dt.hour

        def classify_hour(h):
            if 0 <= h < 6:
                return "Deep night"
            elif 6 <= h < 9:
                return "Morning"
            elif 9 <= h < 17:
                return "Business hours"
            elif 17 <= h < 22:
                return "Evening"
            elif 22 <= h < 24:
                return "Early night"
            return "Unknown"

        daytime = local_hour.apply(classify_hour)
        return pd.DataFrame({"local_time": local_time, "local_hour": local_hour, "daytime": daytime})

    try:
        local_time = utc_time.tz_convert(zone)
        local_hour = local_time.hour

        if 0 <= local_hour < 6:
            daytime = "Deep night"
        elif 6 <= local_hour < 9:
            daytime = "Morning"
        elif 9 <= local_hour < 17:
            daytime = "Business hours"
        elif 17 <= local_hour < 22:
            daytime = "Evening"
        elif 22 <= local_hour < 24:
            daytime = "Early night"
        else:
            daytime = "Unknown"

        return {"local_time": local_time, "local_hour": local_hour, "daytime": daytime}

    except Exception:
        return {"local_time": pd.NaT, "local_hour": None, "daytime": None}


#conv_iso_to_utc("2025-11-04T00:00:00Z")
#conv_utc_to_iso('2025-11-04 00:00:00+0000')
#conv_iso_to_local("2025-11-04T00:00:00Z", "US")
#conv_local_to_iso("2025-11-03 19:00:00-0500", "US")
#conv_iso_to_local_with_daytype("2025-11-04T00:00:00Z", "US")