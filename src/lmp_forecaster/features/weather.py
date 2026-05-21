"""Weather feature normalization and alignment helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]


def normalize_weather_for_panel(
    weather: pd.DataFrame,
    *,
    ds_col: str = "ds",
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Normalize weather frame for panel alignment."""
    if ds_col not in weather.columns:
        raise ValueError("Weather frame missing required column: ds")

    out = weather.copy()
    out[ds_col] = pd.to_datetime(out[ds_col], errors="coerce", utc=False)
    if out[ds_col].isna().any():
        raise ValueError("Weather frame contains non-datetime ds values.")

    if out[ds_col].dt.tz is None:
        out[ds_col] = out[ds_col].dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        out[ds_col] = out[ds_col].dt.tz_convert(timezone)

    for col in WEATHER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.sort_values(ds_col).reset_index(drop=True)
    return out


def align_weather_to_lmp(
    lmp: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    exact_only: bool = True,
    allow_nearest_hour: bool = False,
    fill_weather_limit: int = 0,
) -> pd.DataFrame:
    """Align weather rows to hourly LMP timestamps."""
    if "ds" not in lmp.columns:
        raise ValueError("LMP frame missing required column: ds")

    lmp_sorted = lmp.sort_values(["unique_id", "ds"]).copy()
    weather_sorted = weather.sort_values("ds").copy()

    if exact_only or not allow_nearest_hour:
        merged = lmp_sorted.merge(weather_sorted, on="ds", how="left", suffixes=("", "_weather"))
    else:
        merged = pd.merge_asof(
            lmp_sorted.sort_values("ds"),
            weather_sorted.sort_values("ds"),
            on="ds",
            direction="nearest",
            tolerance=pd.Timedelta(hours=1),
        )

    metadata: dict[str, Any] = {}
    if fill_weather_limit > 0:
        before_missing = int(merged[WEATHER_COLUMNS].isna().sum().sum())
        merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].ffill(limit=fill_weather_limit)
        merged[WEATHER_COLUMNS] = merged[WEATHER_COLUMNS].bfill(limit=fill_weather_limit)
        after_missing = int(merged[WEATHER_COLUMNS].isna().sum().sum())
        metadata["weather_fill_applied"] = True
        metadata["weather_fill_before_missing"] = before_missing
        metadata["weather_fill_after_missing"] = after_missing
    else:
        metadata["weather_fill_applied"] = False

    for col in WEATHER_COLUMNS:
        merged[f"{col}_missing"] = merged[col].isna().astype(int)

    merged.attrs["weather_alignment_metadata"] = metadata
    return merged
