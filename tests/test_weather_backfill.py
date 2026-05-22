"""Tests for real weather backfill command/helper."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app
from lmp_forecaster.data.weather_backfill import pull_real_weather


class _DummyZone:
    zone = "AEP"
    latitude = 39.96
    longitude = -82.99


def test_weather_backfill_dry_run_does_not_write(monkeypatch) -> None:
    runner = CliRunner()
    from lmp_forecaster import cli as cli_module

    monkeypatch.setattr(cli_module, "get_zone", lambda _z: _DummyZone)
    result = runner.invoke(
        app,
        [
            "pull-real-weather",
            "--zone",
            "AEP",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-12-31",
        ],
    )
    assert result.exit_code == 0
    assert "Dry-run only" in result.stdout


def test_weather_normalization_validates_required_columns(monkeypatch, tmp_path: Path) -> None:
    from lmp_forecaster.data import weather_backfill as module

    def fake_get_json(**_kwargs):
        return {
            "latitude": 39.96,
            "longitude": -82.99,
            "timezone": "America/New_York",
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [1.0, 2.0],
                "relative_humidity_2m": [50, 51],
                "dew_point_2m": [0.0, 0.1],
                "apparent_temperature": [0.5, 0.6],
                "precipitation": [0.0, 0.0],
                "wind_speed_10m": [3.0, 3.1],
                "cloud_cover": [10, 11],
            },
        }

    monkeypatch.setattr(module, "get_json", fake_get_json)

    from lmp_forecaster.config import paths as paths_module

    class DummyPaths:
        root = tmp_path

    monkeypatch.setattr(paths_module, "get_project_paths", lambda start=None: DummyPaths())
    monkeypatch.setattr(module, "get_project_paths", lambda: DummyPaths())

    result = pull_real_weather(
        zone="AEP",
        latitude=39.96,
        longitude=-82.99,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        write=True,
        overwrite=True,
    )
    assert result.output_path.exists()
    assert result.quality_report_path is not None
    loaded = pd.read_parquet(result.output_path)
    assert {"ds", "temperature_2m", "source"}.issubset(set(loaded.columns))
