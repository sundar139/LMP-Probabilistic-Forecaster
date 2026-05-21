"""Tests for calendar feature engineering."""

from __future__ import annotations

import pandas as pd

from lmp_forecaster.features.calendar import add_calendar_features


def test_calendar_features_added_and_cyclic_ranges() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2024-07-03 00:00:00",
                periods=48,
                freq="h",
                tz="America/New_York",
            )
        }
    )

    out = add_calendar_features(df)

    required = {
        "hour",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "month",
        "quarter",
        "year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "sin_hour",
        "cos_hour",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_day_of_year",
        "cos_day_of_year",
    }
    assert required.issubset(out.columns)

    for col in [
        "sin_hour",
        "cos_hour",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_day_of_year",
        "cos_day_of_year",
    ]:
        assert out[col].between(-1.0, 1.0).all()


def test_holiday_feature_detects_july_fourth() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2024-07-04 00:00:00",
                periods=2,
                freq="h",
                tz="America/New_York",
            )
        }
    )

    out = add_calendar_features(df)
    assert out["is_holiday"].eq(1).all()
