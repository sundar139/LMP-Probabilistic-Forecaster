"""Tests for configured data source registry."""

from __future__ import annotations

from lmp_forecaster.data.source_registry import load_source_registry


def test_registry_contains_expected_sources() -> None:
    registry = load_source_registry()
    names = {item.name for item in registry.sources}

    assert "pjm_day_ahead_hourly_zonal_lmp" in names
    assert "open_meteo_historical_weather" in names
    assert "open_meteo_historical_forecast_api" in names
