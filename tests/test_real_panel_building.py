"""Tests for real-panel build behavior and validation constraints."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from lmp_forecaster.cli import app
from lmp_forecaster.data.build_panel import PanelBuildConfig, build_single_zone_panel


def _real_lmp(rows: int = 8784) -> pd.DataFrame:
    ds = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h", tz="America/New_York")
    return pd.DataFrame({"unique_id": "AEP", "ds": ds, "y": 30.0})


def _real_weather(ds: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": ds,
            "temperature_2m": 10.0,
            "relative_humidity_2m": 50.0,
            "dew_point_2m": 5.0,
            "apparent_temperature": 10.0,
            "precipitation": 0.0,
            "wind_speed_10m": 5.0,
            "cloud_cover": 20.0,
            "timezone": "America/New_York",
            "source": "openmeteo_historical_weather",
            "pulled_at": pd.Timestamp("2026-01-01T00:00:00Z"),
            "latitude": 39.96,
            "longitude": -82.99,
        }
    )


def test_real_panel_build_rejects_synthetic_source_when_real_required(tmp_path: Path) -> None:
    lmp = _real_lmp(300)
    weather = _real_weather(lmp["ds"])
    cfg = PanelBuildConfig(zone="AEP", start_date=date(2024, 1, 1), end_date=date(2024, 1, 13))

    panel = build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)
    assert panel["source_label"].eq("real").all()

    with pytest.raises(ValueError, match="Real panel build requires real LMP and real weather"):
        bad_cfg = PanelBuildConfig(
            zone="AEP",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 13),
            input_weather_path=tmp_path / "missing_weather.parquet",
            allow_synthetic_weather=True,
            require_real_sources=True,
        )
        build_single_zone_panel(bad_cfg, lmp_frame=lmp)


def test_real_panel_row_count_after_warmup() -> None:
    lmp = _real_lmp(8784)
    weather = _real_weather(lmp["ds"])
    cfg = PanelBuildConfig(
        zone="AEP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        min_history_hours=168,
        drop_warmup_rows=True,
    )
    panel = build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)
    assert len(panel) == 8784 - 168


def test_panel_summary_source_labels_real(tmp_path: Path, monkeypatch) -> None:
    lmp = _real_lmp(8784)
    weather = _real_weather(lmp["ds"])

    from lmp_forecaster import cli as cli_module

    def fake_build(cfg):
        return build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)

    monkeypatch.setattr(cli_module, "build_single_zone_panel", fake_build)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "build-single-zone-panel",
                "--zone",
                "AEP",
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-12-31",
                "--write",
                "--summary",
            ],
        )
        assert result.exit_code == 0
        assert "Source labels: ['real']" in result.stdout
