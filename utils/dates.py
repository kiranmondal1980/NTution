"""
NSE MOMENTUM 5™ — Indian Standard Time (IST) & Trading Calendar Engine
================================================================================
Centralizes all time, date, and calendar operations for Indian Capital Markets.
Guarantees:
- Strict IST (UTC+05:30) timezone awareness
- NSE market session phase identification:
    * Pre-Market: 09:00 to 09:08 IST
    * Normal Regular Trading: 09:15 to 15:30 IST
    * Post-Market: 15:40 to 16:00 IST
- Accurate business/trading day counts excluding weekends
- Standardized ISO timestamp formatting and parsing
================================================================================
"""

from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Tuple, Union
import pandas as pd

# Define Indian Standard Time (IST) Zone: UTC + 5 hours 30 minutes
IST: timezone = timezone(timedelta(hours=5, minutes=30))

# Standard NSE Trading Session Times (IST)
PRE_OPEN_START: time = time(9, 0, 0)
PRE_OPEN_END: time = time(9, 8, 0)
MARKET_OPEN_TIME: time = time(9, 15, 0)
MARKET_CLOSE_TIME: time = time(15, 30, 0)
POST_MARKET_START: time = time(15, 40, 0)
POST_MARKET_END: time = time(16, 0, 0)


def get_ist_now() -> datetime:
    """
    Returns the current precise timestamp in Indian Standard Time (IST).
    """
    return datetime.now(IST)


def get_ist_date() -> date:
    """
    Returns the current date in Indian Standard Time (IST).
    """
    return get_ist_now().date()


def to_ist(dt: datetime) -> datetime:
    """
    Converts any datetime (naive or timezone-aware) to Indian Standard Time (IST).
    If naive, assumes it was recorded in UTC.
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def is_market_open_now() -> bool:
    """
    Determines if the NSE cash equity segment is currently in regular trading hours.
    Returns True only between 09:15:00 and 15:30:00 IST on Monday through Friday.
    """
    now_ist = get_ist_now()
    
    # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday
    if now_ist.weekday() >= 5:
        return False

    current_t = now_ist.time()
    return MARKET_OPEN_TIME <= current_t <= MARKET_CLOSE_TIME


def get_market_session_phase() -> str:
    """
    Identifies the current operational phase of the NSE market.
    Returns:
        'WEEKEND', 'CLOSED_PRE_DAWN', 'PRE_OPEN', 'NORMAL_TRADING',
        'POST_CLOSE_BUFFER', 'POST_MARKET', or 'NIGHT_CLOSED'
    """
    now_ist = get_ist_now()
    if now_ist.weekday() >= 5:
        return "WEEKEND"

    t = now_ist.time()
    if t < PRE_OPEN_START:
        return "CLOSED_PRE_DAWN"
    elif PRE_OPEN_START <= t < PRE_OPEN_END:
        return "PRE_OPEN"
    elif PRE_OPEN_END <= t < MARKET_OPEN_TIME:
        return "PRE_OPEN_BUFFER"
    elif MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME:
        return "NORMAL_TRADING"
    elif MARKET_CLOSE_TIME < t < POST_MARKET_START:
        return "POST_CLOSE_BUFFER"
    elif POST_MARKET_START <= t <= POST_MARKET_END:
        return "POST_MARKET"
    else:
        return "NIGHT_CLOSED"


def trading_days_between(
    start_dt: Union[datetime, date, pd.Timestamp],
    end_dt: Union[datetime, date, pd.Timestamp]
) -> int:
    """
    Calculates the exact number of business trading sessions between two dates
    using the standard Monday-to-Friday business day calendar.
    
    Args:
        start_dt: Beginning date or timestamp
        end_dt: Ending date or timestamp

    Returns:
        Integer count of sessions elapsed (0 if start >= end).
    """
    if pd.isna(start_dt) or pd.isna(end_dt):
        return 0

    # Normalize to pandas timestamps
    ts_start = pd.to_datetime(start_dt)
    ts_end = pd.to_datetime(end_dt)

    if ts_start >= ts_end:
        return 0

    # Business day range excludes Saturdays and Sundays
    bdate_range = pd.bdate_range(start=ts_start.date(), end=ts_end.date())
    # Exclude the starting day itself to return elapsed intervals
    return max(0, len(bdate_range) - 1)


def get_prior_trading_date(ref_date: Optional[date] = None, lookback_days: int = 1) -> date:
    """
    Calculates the calendar date of the Nth previous trading session (excluding weekends).
    """
    target = ref_date or get_ist_date()
    while lookback_days > 0:
        target -= timedelta(days=1)
        # Weekdays: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
        if target.weekday() < 5:
            lookback_days -= 1
    return target


def get_next_trading_date(ref_date: Optional[date] = None) -> date:
    """
    Calculates the date of the next consecutive trading session (skipping weekends).
    """
    target = ref_date or get_ist_date()
    target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def format_iso(dt: Optional[Union[datetime, pd.Timestamp]]) -> str:
    """
    Standardizes a datetime into an ISO 8601 formatted string.
    """
    if dt is None or pd.isna(dt):
        return ""
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d %H:%M:%S")
