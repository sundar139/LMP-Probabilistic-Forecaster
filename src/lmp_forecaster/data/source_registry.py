"""Data source registry for documented upstream providers."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from lmp_forecaster.config.paths import get_project_paths


class DataSource(BaseModel):
    """Single source descriptor."""

    name: str
    provider: str
    purpose: str
    expected_frequency: str
    notes: str


class SourceRegistry(BaseModel):
    """Collection of upstream sources."""

    sources: list[DataSource] = Field(default_factory=list)


def load_source_registry(path: Path | None = None) -> SourceRegistry:
    """Load source metadata from YAML."""
    registry_path = path or (get_project_paths().conf / "data_sources.yaml")
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Source registry must parse to a mapping: {registry_path}")

    return SourceRegistry(**payload)
