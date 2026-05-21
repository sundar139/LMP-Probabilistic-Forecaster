"""Tests for project path utilities."""

from __future__ import annotations

from pathlib import Path

from lmp_forecaster.config.paths import get_project_paths


def test_project_paths_resolve() -> None:
    paths = get_project_paths(Path(__file__).resolve().parents[1])

    assert paths.root.exists()
    assert paths.conf.name == "conf"
    assert paths.data.name == "data"
    assert paths.raw.name == "raw"
    assert paths.cache.name == "cache"
    assert paths.processed.name == "processed"
