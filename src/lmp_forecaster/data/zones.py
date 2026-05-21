"""Zone metadata registry used for weather proxy mapping."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from lmp_forecaster.config.paths import get_project_paths


class ZoneMetadata(BaseModel):
    """Project-level zone metadata.

    Coordinates are approximate weather proxy centroids and can be refined later.
    """

    zone: str
    display_name: str
    latitude: float
    longitude: float
    region_cluster: str
    type: str = "ZONE"


class ZoneRegistry(BaseModel):
    """Collection of configured zones."""

    zones: list[ZoneMetadata]



def load_zone_registry(path: Path | None = None) -> ZoneRegistry:
    """Load zone metadata YAML."""
    active_path = path or (get_project_paths().conf / "zones.yaml")
    if not active_path.exists():
        raise FileNotFoundError(f"Zone registry not found: {active_path}")

    with active_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Zone registry must parse to a mapping: {active_path}")

    registry = ZoneRegistry(**payload)
    return registry


def get_zone(zone: str, path: Path | None = None) -> ZoneMetadata:
    """Get zone metadata by code."""
    needle = zone.upper()
    registry = load_zone_registry(path=path)
    for item in registry.zones:
        if item.zone.upper() == needle:
            return item
    raise KeyError(f"Zone not found in registry: {zone}")
