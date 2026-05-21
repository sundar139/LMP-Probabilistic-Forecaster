"""Application settings loaded from environment and YAML config."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lmp_forecaster.config.paths import ProjectPaths, get_project_paths


class ForecastDefaults(BaseModel):
    """Forecast defaults shared across training and serving."""

    horizon: int = 24
    input_size: int = 168
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    default_zone: str = "AEP"


class ProjectDefaults(BaseModel):
    """Project-level defaults."""

    name: str = "lmp-forecaster"
    timezone: str = "America/New_York"


class AppSettings(BaseSettings):
    """Environment-overridable settings."""

    model_config = SettingsConfigDict(
        env_prefix="LMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    timezone: str = "America/New_York"
    forecast_horizon: int = 24
    input_size: int = 168
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    default_zone: str = "AEP"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must parse to a mapping: {path}")
    return raw


@lru_cache(maxsize=1)
def load_defaults(paths: ProjectPaths | None = None) -> tuple[ProjectDefaults, ForecastDefaults]:
    """Load immutable defaults from conf/base.yaml."""
    active_paths = paths or get_project_paths()
    payload = _load_yaml(active_paths.conf / "base.yaml")

    project_data = payload.get("project", {})
    forecast_data = payload.get("forecast", {})

    return ProjectDefaults(**project_data), ForecastDefaults(**forecast_data)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Build runtime settings using YAML defaults + env overrides."""
    project_defaults, forecast_defaults = load_defaults()
    return AppSettings(
        timezone=project_defaults.timezone,
        forecast_horizon=forecast_defaults.horizon,
        input_size=forecast_defaults.input_size,
        quantiles=forecast_defaults.quantiles,
        default_zone=forecast_defaults.default_zone,
    )
