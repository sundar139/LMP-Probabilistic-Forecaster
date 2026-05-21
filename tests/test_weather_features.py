"""Tests for weather normalization and alignment."""

from __future__ import annotations

import pandas as pd

from lmp_forecaster.features.weather import (
    align_weather_to_lmp,
    normalize_weather_for_panel,
)


def test_weather_aligns_exactly_by_ds() -> None:
    lmp = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP", "AEP"],
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=3,
                freq="h",
                tz="America/New_York",
            ),
            "y": [10.0, 11.0, 12.0],
        }
    )
    weather = pd.DataFrame(
        {
            "ds": lmp["ds"],
            "temperature_2m": [1.0, 2.0, 3.0],
            "relative_humidity_2m": [50, 51, 52],
            "dew_point_2m": [0.0, 1.0, 2.0],
            "apparent_temperature": [0.5, 1.5, 2.5],
            "precipitation": [0.0, 0.0, 0.1],
            "wind_speed_10m": [5.0, 6.0, 7.0],
            "cloud_cover": [10, 20, 30],
            "source": ["openmeteo", "openmeteo", "openmeteo"],
        }
    )

    norm = normalize_weather_for_panel(weather)
    out = align_weather_to_lmp(lmp, norm)

    assert out["temperature_2m"].tolist() == [1.0, 2.0, 3.0]


def test_missing_indicators_created_when_weather_missing() -> None:
    lmp = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP"],
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=2,
                freq="h",
                tz="America/New_York",
            ),
            "y": [10.0, 11.0],
        }
    )
    weather = pd.DataFrame(
        {
            "ds": lmp["ds"],
            "temperature_2m": [1.0, None],
            "relative_humidity_2m": [50, None],
            "dew_point_2m": [0.0, None],
            "apparent_temperature": [0.5, None],
            "precipitation": [0.0, None],
            "wind_speed_10m": [5.0, None],
            "cloud_cover": [10, None],
            "source": ["openmeteo", "openmeteo"],
        }
    )

    out = align_weather_to_lmp(lmp, normalize_weather_for_panel(weather))
    assert "temperature_2m_missing" in out.columns
    assert out["temperature_2m_missing"].tolist() == [0, 1]
