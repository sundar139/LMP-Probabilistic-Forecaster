"""Single-zone panel builder from cached LMP/weather or synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.synthetic_panel import SyntheticPanelConfig, make_synthetic_panel
from lmp_forecaster.features.calendar import add_calendar_features
from lmp_forecaster.features.lags import (
    add_lag_features,
    add_rolling_features,
    validate_no_lag_leakage,
)
from lmp_forecaster.features.weather import (
    WEATHER_COLUMNS,
    align_weather_to_lmp,
    normalize_weather_for_panel,
)

REQUIRED_OUTPUT_COLUMNS = [
    "unique_id",
    "ds",
    "y",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_day_of_year",
    "cos_day_of_year",
    *WEATHER_COLUMNS,
    "lmp_lag_1",
    "lmp_lag_2",
    "lmp_lag_3",
    "lmp_lag_24",
    "lmp_lag_48",
    "lmp_lag_168",
    "lmp_rolling_mean_24",
    "lmp_rolling_std_24",
    "lmp_rolling_min_24",
    "lmp_rolling_max_24",
    "lmp_rolling_mean_168",
    "lmp_rolling_std_168",
]


@dataclass(frozen=True)
class PanelBuildConfig:
    zone: str = "AEP"
    timezone: str = "America/New_York"
    input_lmp_path: Path | None = None
    input_weather_path: Path | None = None
    output_path: Path | None = None
    start_date: date | None = None
    end_date: date | None = None
    min_history_hours: int = 168
    allow_synthetic_lmp: bool = False
    allow_synthetic_weather: bool = False
    drop_warmup_rows: bool = True
    fill_weather_limit: int = 3


def _default_lmp_cache_path(zone: str) -> Path:
    root = get_project_paths().root
    base = root / "data" / "cache" / "pjm" / "da_hrl_lmps"
    matches = sorted(base.glob(f"*_{zone}.parquet")) if base.exists() else []
    if matches:
        return matches[-1]
    return base / f"da_hrl_lmps_default_{zone}.parquet"


def _default_weather_cache_path() -> Path:
    root = get_project_paths().root
    base = root / "data" / "cache" / "weather" / "openmeteo"
    matches = sorted(base.glob("historical_weather_*.parquet")) if base.exists() else []
    if matches:
        return matches[-1]
    return base / "historical_weather_default.parquet"


def _coerce_lmp_frame(frame: pd.DataFrame, zone: str, timezone: str) -> pd.DataFrame:
    required = ["unique_id", "ds", "y"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"LMP frame missing required columns: {missing}")

    out = frame.copy()
    out = out[out["unique_id"].astype(str).str.upper() == zone.upper()].copy()
    if out.empty:
        raise ValueError(f"No rows for zone {zone} in LMP frame.")

    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", utc=False)
    if out["ds"].isna().any():
        raise ValueError("LMP frame contains invalid ds values.")

    if out["ds"].dt.tz is None:
        out["ds"] = out["ds"].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    else:
        out["ds"] = out["ds"].dt.tz_convert(timezone)

    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if out["y"].isna().any():
        raise ValueError("LMP frame has null/non-numeric target y.")

    if out.duplicated(subset=["unique_id", "ds"]).any():
        raise ValueError("LMP frame has duplicate unique_id+ds rows.")

    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out[["unique_id", "ds", "y"]]


def _synthetic_lmp(zone: str, timezone: str, min_rows: int) -> pd.DataFrame:
    cfg = SyntheticPanelConfig(zones=(zone,), periods=max(240, min_rows), timezone=timezone)
    frame = make_synthetic_panel(cfg)[["unique_id", "ds", "y"]].copy()
    frame["source_label"] = "synthetic"
    return frame


def _synthetic_weather_from_lmp(lmp: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(lmp), dtype=float)
    weather = pd.DataFrame(
        {
            "ds": lmp["ds"],
            "temperature_2m": 60 + 10 * np.sin(2 * np.pi * idx / 24),
            "relative_humidity_2m": 50 + 20 * np.cos(2 * np.pi * idx / 24),
            "dew_point_2m": 45 + 5 * np.sin(2 * np.pi * idx / 24),
            "apparent_temperature": 58 + 9 * np.sin(2 * np.pi * idx / 24),
            "precipitation": np.where((idx % 24) < 2, 0.1, 0.0),
            "wind_speed_10m": 7 + 2 * np.cos(2 * np.pi * idx / 24),
            "cloud_cover": 40 + 30 * np.sin(2 * np.pi * idx / 12),
            "source": "synthetic_weather",
        }
    )
    return weather


def load_lmp_frame(config: PanelBuildConfig) -> tuple[pd.DataFrame, str]:
    """Load LMP data from cache or synthetic fallback."""
    path = config.input_lmp_path or _default_lmp_cache_path(config.zone)
    if path.exists():
        frame = pd.read_parquet(path)
        return _coerce_lmp_frame(frame, config.zone, config.timezone), "real"

    if not config.allow_synthetic_lmp:
        raise FileNotFoundError(
            f"No LMP parquet found at {path}. Use --allow-synthetic-lmp for smoke fallback."
        )

    synth = _synthetic_lmp(config.zone, config.timezone, config.min_history_hours + 72)
    return _coerce_lmp_frame(synth, config.zone, config.timezone), "synthetic"


def load_weather_frame(
    config: PanelBuildConfig,
    lmp_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Load weather data from cache or synthetic fallback."""
    path = config.input_weather_path or _default_weather_cache_path()
    if path.exists():
        weather = pd.read_parquet(path)
        return normalize_weather_for_panel(weather, timezone=config.timezone), "real"

    if not config.allow_synthetic_weather:
        raise FileNotFoundError(
            f"No weather parquet found at {path}. Use --allow-synthetic-weather for smoke fallback."
        )

    synth = _synthetic_weather_from_lmp(lmp_frame)
    return normalize_weather_for_panel(synth, timezone=config.timezone), "synthetic"


def _apply_date_window(frame: pd.DataFrame, config: PanelBuildConfig) -> pd.DataFrame:
    out = frame.copy()
    if config.start_date is not None:
        start_ts = pd.Timestamp(config.start_date).tz_localize(config.timezone)
        out = out[out["ds"] >= start_ts]
    if config.end_date is not None:
        end_ts = pd.Timestamp(config.end_date).tz_localize(config.timezone) + pd.Timedelta(hours=23)
        out = out[out["ds"] <= end_ts]
    return out.reset_index(drop=True)


def validate_panel_schema(panel: pd.DataFrame, config: PanelBuildConfig) -> None:
    """Validate panel output constraints."""
    missing = [col for col in REQUIRED_OUTPUT_COLUMNS if col not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required output columns: {missing}")

    if panel[["unique_id", "ds", "y"]].isna().any().any():
        raise ValueError("Panel has nulls in required fields unique_id/ds/y.")

    if panel["unique_id"].astype(str).str.upper().nunique() != 1:
        raise ValueError("Panel must contain exactly one unique_id.")
    if panel["unique_id"].astype(str).str.upper().iloc[0] != config.zone.upper():
        raise ValueError("Panel unique_id does not match requested zone.")

    if panel.duplicated(subset=["unique_id", "ds"]).any():
        raise ValueError("Panel has duplicate unique_id+ds rows.")

    if not pd.api.types.is_numeric_dtype(panel["y"]):
        raise ValueError("Panel y must be numeric.")

    sorted_panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if not sorted_panel["ds"].equals(panel.reset_index(drop=True)["ds"]):
        raise ValueError("Panel ds must be sorted within unique_id.")

    validate_no_lag_leakage(panel)

    if config.drop_warmup_rows:
        lag_roll = [
            col
            for col in panel.columns
            if col.startswith("lmp_lag_") or col.startswith("lmp_rolling_")
        ]
        null_counts = panel[lag_roll].isna().sum().sum()
        if int(null_counts) > 0:
            raise ValueError("Lag/rolling nulls remain after warmup drop.")

    for col in WEATHER_COLUMNS:
        if panel[col].isna().any() and f"{col}_missing" not in panel.columns:
            raise ValueError(f"Missing weather indicator for nullable column: {col}")

    tz_name = str(panel["ds"].dt.tz)
    if tz_name != config.timezone:
        raise ValueError(f"Panel timezone mismatch: expected {config.timezone}, got {tz_name}")


def summarize_panel(panel: pd.DataFrame) -> dict[str, Any]:
    """Summarize high-level panel characteristics."""
    return {
        "rows": int(len(panel)),
        "start_ds": str(panel["ds"].min()),
        "end_ds": str(panel["ds"].max()),
        "y_mean": float(panel["y"].mean()),
        "y_std": float(panel["y"].std()),
    }


def build_single_zone_panel(
    config: PanelBuildConfig,
    *,
    lmp_frame: pd.DataFrame | None = None,
    weather_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build cleaned single-zone panel with leakage-safe features."""
    lmp_source = "real"
    weather_source = "real"

    if lmp_frame is None:
        lmp_frame, lmp_source = load_lmp_frame(config)
    else:
        lmp_frame = _coerce_lmp_frame(lmp_frame, config.zone, config.timezone)

    if weather_frame is None:
        weather_frame, weather_source = load_weather_frame(config, lmp_frame)
    else:
        weather_frame = normalize_weather_for_panel(weather_frame, timezone=config.timezone)

    lmp_frame = _apply_date_window(lmp_frame, config)
    aligned = align_weather_to_lmp(
        lmp_frame,
        weather_frame,
        exact_only=True,
        allow_nearest_hour=False,
        fill_weather_limit=config.fill_weather_limit,
    )

    panel = add_calendar_features(aligned, timezone=config.timezone)
    panel = add_lag_features(panel)
    panel = add_rolling_features(panel)

    if config.drop_warmup_rows:
        panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        panel = panel.iloc[config.min_history_hours :].reset_index(drop=True)

    source_label = "real"
    if lmp_source == "synthetic" and weather_source == "synthetic":
        source_label = "synthetic"
    elif lmp_source == "real" and weather_source == "real":
        source_label = "real"
    else:
        source_label = "mixed"

    panel["source_label"] = source_label

    validate_panel_schema(panel, config)
    return panel


def write_panel(panel: pd.DataFrame, config: PanelBuildConfig) -> Path:
    """Write panel parquet to processed single-zone path."""
    root = get_project_paths().root
    output = config.output_path or (
        root / "data" / "processed" / "panel" / "single_zone" / f"{config.zone}_panel.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    return output
