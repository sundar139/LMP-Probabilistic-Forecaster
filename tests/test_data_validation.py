"""Tests for ingestion validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from lmp_forecaster.data.validation import (
    detect_dst_day_lengths,
    validate_lmp_frame,
)


def test_detect_dst_day_lengths_identifies_spring_and_fall_days() -> None:
    spring = pd.Series(
        pd.date_range(
            start="2024-03-10 00:00:00",
            end="2024-03-10 23:00:00",
            freq="h",
            tz="America/New_York",
        )
    )
    fall = pd.Series(
        pd.date_range(
            start="2024-11-03 00:00:00",
            end="2024-11-03 23:00:00",
            freq="h",
            tz="America/New_York",
        )
    )

    spring_lengths = {item.hours for item in detect_dst_day_lengths(spring)}
    fall_lengths = {item.hours for item in detect_dst_day_lengths(fall)}

    assert 23 in spring_lengths
    assert 25 in fall_lengths


def test_validate_lmp_frame_requires_monotonic_hourly_series() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP"],
            "ds": [
                pd.Timestamp("2024-01-01 01:00:00", tz="America/New_York"),
                pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York"),
            ],
            "y": [10.0, 11.0],
            "market": ["DAY_AHEAD", "DAY_AHEAD"],
            "location_type": ["ZONE", "ZONE"],
            "source": ["pjm", "pjm"],
            "pulled_at": [
                pd.Timestamp("2024-01-02T00:00:00Z"),
                pd.Timestamp("2024-01-02T00:00:00Z"),
            ],
        }
    )

    with pytest.raises(ValueError):
        validate_lmp_frame(frame)
