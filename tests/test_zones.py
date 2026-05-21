"""Tests for zone metadata registry."""

from __future__ import annotations

from lmp_forecaster.data.zones import get_zone, load_zone_registry

EXPECTED_ZONES = {"AEP", "AECO", "ATSI", "BGE", "COMED", "DOM", "PECO", "PEPCO", "PPL", "PSEG"}


def test_zone_registry_contains_initial_ten_zones() -> None:
    registry = load_zone_registry()
    observed = {item.zone for item in registry.zones}

    assert EXPECTED_ZONES.issubset(observed)
    assert len(registry.zones) == 10


def test_get_zone_returns_metadata() -> None:
    zone = get_zone("AEP")
    assert zone.zone == "AEP"
    assert zone.type == "ZONE"
    assert isinstance(zone.latitude, float)
    assert isinstance(zone.longitude, float)
