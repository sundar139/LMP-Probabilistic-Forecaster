"""Tests for single-zone panel builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lmp_forecaster.data.build_panel import (
    PanelBuildConfig,
    build_single_zone_panel,
    validate_panel_schema,
)


def _lmp(rows: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": [float(i) for i in range(rows)],
        }
    )


def _weather_from_lmp(lmp: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": lmp["ds"],
            "temperature_2m": 1.0,
            "relative_humidity_2m": 50.0,
            "dew_point_2m": 0.0,
            "apparent_temperature": 0.5,
            "precipitation": 0.0,
            "wind_speed_10m": 4.0,
            "cloud_cover": 10.0,
            "source": "synthetic_weather",
        }
    )


def test_panel_builder_creates_required_schema_and_drops_warmup(tmp_path: Path) -> None:
    lmp = _lmp(240)
    weather = _weather_from_lmp(lmp)

    cfg = PanelBuildConfig(zone="AEP", drop_warmup_rows=True, min_history_hours=168)
    panel = build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)

    validate_panel_schema(panel, cfg)
    assert len(panel) == 240 - 168
    assert panel["lmp_lag_168"].notna().all()


def test_panel_builder_rejects_missing_y(tmp_path: Path) -> None:
    lmp = _lmp(240)
    lmp.loc[0, "y"] = None
    weather = _weather_from_lmp(lmp.fillna(0.0))

    cfg = PanelBuildConfig(zone="AEP")
    with pytest.raises(ValueError):
        build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)


def test_panel_builder_rejects_duplicate_rows() -> None:
    lmp = _lmp(10)
    lmp = pd.concat([lmp, lmp.iloc[[0]]], ignore_index=True)
    weather = _weather_from_lmp(_lmp(11))

    cfg = PanelBuildConfig(zone="AEP", min_history_hours=1)
    with pytest.raises(ValueError):
        build_single_zone_panel(cfg, lmp_frame=lmp, weather_frame=weather)


def test_builder_fails_without_lmp_when_synthetic_disabled(tmp_path: Path) -> None:
    cfg = PanelBuildConfig(
        zone="AEP",
        input_lmp_path=tmp_path / "missing.parquet",
        input_weather_path=tmp_path / "missing_weather.parquet",
        allow_synthetic_lmp=False,
        allow_synthetic_weather=False,
    )

    with pytest.raises(FileNotFoundError):
        build_single_zone_panel(cfg)


def test_synthetic_fallback_requires_explicit_flags() -> None:
    cfg = PanelBuildConfig(
        zone="AEP",
        allow_synthetic_lmp=True,
        allow_synthetic_weather=True,
    )
    panel = build_single_zone_panel(cfg)
    assert "source_label" in panel.columns
    assert panel["source_label"].iloc[0] in {"synthetic", "mixed"}
