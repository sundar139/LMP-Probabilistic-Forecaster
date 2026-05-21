"""Validation helpers for normalized ingestion outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_LMP_COLUMNS = (
    "unique_id",
    "ds",
    "y",
    "market",
    "location_type",
    "source",
    "pulled_at",
)
REQUIRED_WEATHER_COLUMNS = (
    "ds",
    "latitude",
    "longitude",
    "timezone",
    "source",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "pulled_at",
)


@dataclass(frozen=True)
class DstDayLength:
    """Detected daily hour count for DST-aware checks."""

    day: str
    hours: int


def assert_no_required_nulls(frame: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise when required columns contain null values."""
    for col in required_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing required column: {col}")
        if frame[col].isna().any():
            raise ValueError(f"Required column has nulls: {col}")


def assert_monotonic_time_per_series(frame: pd.DataFrame, id_col: str = "unique_id") -> None:
    """Ensure timestamps are monotonic increasing within each series."""
    if "ds" not in frame.columns:
        raise ValueError("Missing required column: ds")
    if id_col not in frame.columns:
        raise ValueError(f"Missing required column: {id_col}")

    for series_id, grp in frame.groupby(id_col):
        if not grp["ds"].is_monotonic_increasing:
            raise ValueError(f"Non-monotonic timestamps for series: {series_id}")


def assert_expected_hourly_spacing(frame: pd.DataFrame, id_col: str = "unique_id") -> None:
    """Ensure hourly spacing where differences are measurable in UTC seconds."""
    if "ds" not in frame.columns:
        raise ValueError("Missing required column: ds")

    work = frame.copy()
    work["_utc"] = pd.to_datetime(work["ds"], utc=True)

    grouped = work.groupby(id_col) if id_col in work.columns else [("_single", work)]
    for series_id, grp in grouped:
        deltas = grp.sort_values("_utc")["_utc"].diff().dropna()
        if deltas.empty:
            continue
        bad = deltas[deltas != pd.Timedelta(hours=1)]
        if not bad.empty:
            raise ValueError(
                f"Unexpected spacing in series {series_id}; "
                "expected 1-hour intervals."
            )


def detect_dst_day_lengths(
    series: pd.Series,
    timezone: str = "America/New_York",
) -> list[DstDayLength]:
    """Return day-level hour counts to identify DST spring/fall behavior."""
    parsed = pd.to_datetime(series, utc=False)
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        localized = parsed.dt.tz_convert(timezone)

    counts = (
        pd.DataFrame({"ds": localized.dropna()})
        .assign(day=lambda x: x["ds"].dt.strftime("%Y-%m-%d"))
        .groupby("day", as_index=False)
        .size()
    )

    return [DstDayLength(day=row["day"], hours=int(row["size"])) for _, row in counts.iterrows()]


def validate_lmp_frame(frame: pd.DataFrame) -> None:
    """Validate normalized LMP frame schema and basic integrity."""
    assert_no_required_nulls(frame, list(REQUIRED_LMP_COLUMNS))
    assert_monotonic_time_per_series(frame, id_col="unique_id")
    assert_expected_hourly_spacing(frame, id_col="unique_id")


def validate_weather_frame(frame: pd.DataFrame) -> None:
    """Validate normalized weather frame schema and timestamp integrity."""
    assert_no_required_nulls(
        frame,
        ["ds", "latitude", "longitude", "timezone", "source", "pulled_at"],
    )
    missing = [col for col in REQUIRED_WEATHER_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required weather columns: {missing}")
    assert_expected_hourly_spacing(frame, id_col="source")
