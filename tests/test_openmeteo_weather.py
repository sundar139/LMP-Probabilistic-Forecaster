"""Tests for Open-Meteo weather adapters."""

from __future__ import annotations

from datetime import date

import pandas as pd

from lmp_forecaster.data.openmeteo_weather import (
    OpenMeteoRequestConfig,
    build_historical_forecast_request,
    build_historical_weather_request,
    normalize_openmeteo_hourly,
)


def test_openmeteo_normalization_hourly_rows() -> None:
    payload = {
        "latitude": 39.96,
        "longitude": -82.99,
        "timezone": "America/New_York",
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [1.0, 2.0],
            "relative_humidity_2m": [60, 62],
            "dew_point_2m": [-1.0, 0.0],
            "apparent_temperature": [0.5, 1.5],
            "precipitation": [0.0, 0.1],
            "wind_speed_10m": [3.0, 4.0],
            "cloud_cover": [20, 30],
        },
    }

    frame = normalize_openmeteo_hourly(payload, source="openmeteo_historical_weather")

    assert len(frame) == 2
    assert "ds" in frame.columns
    assert "temperature_2m" in frame.columns
    assert frame["source"].iloc[0] == "openmeteo_historical_weather"
    assert not frame["ds"].isna().any()


def test_openmeteo_normalization_handles_dst_fallback_ambiguity() -> None:
    payload = {
        "latitude": 39.96,
        "longitude": -82.99,
        "timezone": "America/New_York",
        "hourly": {
            "time": [
                "2024-11-03T00:00",
                "2024-11-03T01:00",
                "2024-11-03T02:00",
                "2024-11-03T03:00",
            ],
            "temperature_2m": [1.0, 2.0, 2.5, 3.0],
            "relative_humidity_2m": [60, 62, 61, 63],
            "dew_point_2m": [-1.0, 0.0, 0.1, 0.2],
            "apparent_temperature": [0.5, 1.5, 1.7, 2.0],
            "precipitation": [0.0, 0.1, 0.0, 0.0],
            "wind_speed_10m": [3.0, 4.0, 4.1, 4.2],
            "cloud_cover": [20, 30, 35, 40],
        },
    }

    frame = normalize_openmeteo_hourly(payload, source="openmeteo_historical_weather")

    assert len(frame) == 4
    assert not frame["ds"].isna().any()
    assert str(frame["ds"].dt.tz) == "America/New_York"
    utc_deltas = frame["ds"].dt.tz_convert("UTC").diff().dropna()
    assert (utc_deltas == pd.Timedelta(hours=1)).all()


def test_historical_forecast_request_params() -> None:
    cfg = OpenMeteoRequestConfig(
        latitude=39.96,
        longitude=-82.99,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )

    weather_url, weather_params = build_historical_weather_request(cfg)
    forecast_url, forecast_params = build_historical_forecast_request(cfg)

    assert "archive-api" in weather_url
    assert "historical-forecast-api" in forecast_url
    assert weather_params["timezone"] == "America/New_York"
    assert "temperature_2m" in str(weather_params["hourly"])
    assert forecast_params["models"] == "best_match"
