from typing import Any, cast, overload
import pandas as pd


TimeLike = str | pd.Series | pd.Timestamp

@overload
def conv_iso_to_utc(time: str | pd.Timestamp) -> pd.Timestamp:
    ...

@overload
def conv_iso_to_utc(time: pd.Series) -> pd.Series:
    ...

def conv_iso_to_utc(time: TimeLike) -> pd.Timestamp | pd.Series:
    """
    Converts ISO 8601 extended date-time format timestamp to UTC time format 
    timestamp.
    """
    return pd.to_datetime(time, utc=True)

def conv_utc_to_iso(time: TimeLike) -> str | pd.Series:
    """
    Converts UTC time format timestamp to ISO 8601 extended date-time format 
    timestamp.
    """
    dt = pd.to_datetime(time, utc=True)

    if isinstance(dt, pd.Series):
        return dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def conv_iso_to_local(
    time: TimeLike, 
    country: str, 
    tz: dict
) -> pd.Timestamp | pd.Series:
    """
    Converts ISO 8601 extended date-time format timestamp into local timestamp.
    """
    utc_time = conv_iso_to_utc(time)
    zone = tz[country]["zone"]

    if isinstance(utc_time, pd.Series):
        time_series: pd.Series = utc_time.dt.tz_convert(zone)
        return time_series
    
    return utc_time.tz_convert(zone)

def conv_local_to_iso(
    time: TimeLike, 
    country: str, 
    tz: dict
) -> str | pd.Series:
    """
    Converts local timestamp to ISO 8601 extended date-time format timestamp.
    """
    local_tz = tz[country]["zone"]
    dt = pd.to_datetime(time)

    if isinstance(dt, pd.Series):
        has_tz = dt.dt.tz is not None if not dt.empty else False
        if not has_tz:
            dt = dt.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
        else: 
            dt = dt.dt.tz_convert(local_tz)
        utc_time = dt.dt.tz_convert("UTC")
        return utc_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    if dt.tzinfo is None:
        dt = dt.tz_localize(local_tz)
    else:
        dt = dt.tz_convert(local_tz)

    utc_time = dt.tz_convert("UTC")
    return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

def conv_iso_to_local_with_daytype(
    time: TimeLike, 
    country: str, 
    tz: dict
) -> dict[str, Any]:
    """
    Converts an ISO 8601 timestamp to the country's local time and returns 
    weekday/weekend info.
    """
    utc_time = conv_iso_to_utc(time)

    zone = tz[country]["zone"]
    
    try:
        if isinstance(time, pd.Series):
            utc_time = cast(pd.Series, utc_time)
            local_time = utc_time.dt.tz_convert(zone)
            weekday = local_time.dt.dayofweek
            daytype = weekday.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        else:
            utc_time = cast(pd.Timestamp, utc_time)
            local_time = utc_time.tz_convert(zone)
            weekday = local_time.dayofweek
            daytype = "Weekend" if weekday >= 5 else "Weekday"
        return {"local_time": local_time, "weekday": weekday, "daytype": daytype}
    except Exception:
        raise ValueError("[ERROR] Conversion failed for local with daytype")

def conv_iso_to_local_with_daytimes(
    time: TimeLike, 
    country: str, 
    tz: dict  
) -> dict[str, Any]:
    """
    Converts an ISO 8601 timestamp to the country's local time and classifies 
    into a daytime bucket.
    """
    utc_time = conv_iso_to_utc(time)
    zone = tz[country]["zone"]

    def classify_hour(h: int) -> str:
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

    try:
        if isinstance(time, pd.Series):
            utc_time = cast(pd.Series, utc_time)
            local_time = utc_time.dt.tz_convert(zone)
            local_hour = local_time.dt.hour
            daytime = local_hour.apply(classify_hour)
        else:
            utc_time = cast(pd.Timestamp, utc_time)
            local_time = utc_time.tz_convert(zone)
            local_hour = local_time.hour
            daytime = classify_hour(local_hour)

        return {"local_time": local_time, "local_hour": local_hour, "daytime": daytime}

    except Exception:
        raise ValueError("[ERROR] Conversion failed for local with daytime")


if __name__=='__main__':
    conv_iso_to_utc("2025-11-04T00:00:00Z")
    #conv_utc_to_iso('2025-11-04 00:00:00+0000')
    #conv_iso_to_local("2025-11-04T00:00:00Z", "US")
    #conv_local_to_iso("2025-11-03 19:00:00-0500", "US")
    #conv_iso_to_local_with_daytype("2025-11-04T00:00:00Z", "US")