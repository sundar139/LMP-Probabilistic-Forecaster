"""Calendar feature engineering for panel datasets."""

from __future__ import annotations

import math

import holidays
import pandas as pd
from holidays import HolidayBase

REQUIRED_DS = "ds"


def _coerce_ds(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().any():
        raise ValueError("Column 'ds' contains non-datetime values that cannot be parsed.")

    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return parsed.dt.tz_convert(timezone)


def add_calendar_features(
    frame: pd.DataFrame,
    *,
    ds_col: str = REQUIRED_DS,
    timezone: str = "America/New_York",
    copy: bool = True,
) -> pd.DataFrame:
    """Add calendar/cyclic covariates from a datetime column.

    Raises a ValueError when the datetime column is missing or invalid.
    """
    if ds_col not in frame.columns:
        raise ValueError("Column 'ds' is required for calendar feature generation.")

    out = frame.copy(deep=True) if copy else frame
    out[ds_col] = _coerce_ds(out[ds_col], timezone)

    ds = out[ds_col]
    out["hour"] = ds.dt.hour
    out["day_of_week"] = ds.dt.dayofweek
    out["day_of_month"] = ds.dt.day
    out["day_of_year"] = ds.dt.dayofyear
    out["month"] = ds.dt.month
    out["quarter"] = ds.dt.quarter
    out["year"] = ds.dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_month_start"] = ds.dt.is_month_start.astype(int)
    out["is_month_end"] = ds.dt.is_month_end.astype(int)

    years = sorted(out["year"].dropna().astype(int).unique().tolist())
    us_holidays: HolidayBase = holidays.country_holidays("US", years=years)
    out["is_holiday"] = ds.dt.date.map(lambda x: 1 if x in us_holidays else 0).astype(int)

    out["sin_hour"] = out["hour"].map(lambda x: math.sin(2.0 * math.pi * x / 24.0))
    out["cos_hour"] = out["hour"].map(lambda x: math.cos(2.0 * math.pi * x / 24.0))

    out["sin_day_of_week"] = out["day_of_week"].map(
        lambda x: math.sin(2.0 * math.pi * x / 7.0)
    )
    out["cos_day_of_week"] = out["day_of_week"].map(
        lambda x: math.cos(2.0 * math.pi * x / 7.0)
    )

    out["sin_day_of_year"] = out["day_of_year"].map(
        lambda x: math.sin(2.0 * math.pi * x / 366.0)
    )
    out["cos_day_of_year"] = out["day_of_year"].map(
        lambda x: math.cos(2.0 * math.pi * x / 366.0)
    )

    return out
