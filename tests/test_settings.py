"""Tests for application settings defaults."""

from __future__ import annotations

from lmp_forecaster.config.settings import get_settings


def test_default_settings_values() -> None:
    settings = get_settings()

    assert settings.timezone == "America/New_York"
    assert settings.forecast_horizon == 24
    assert settings.input_size == 168
    assert settings.quantiles == [0.1, 0.5, 0.9]
    assert settings.default_zone == "AEP"
