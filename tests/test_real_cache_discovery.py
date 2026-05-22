"""Tests for real cache discovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lmp_forecaster.data.real_cache_discovery import (
    locate_latest_lmp_cache,
    locate_latest_lmp_quality_report,
    locate_latest_weather_cache,
)


def _write_lmp_chunk(path: Path, start: str, end: str, zone: str = "AEP") -> None:
    ds = pd.date_range(f"{start} 00:00:00", f"{end} 23:00:00", freq="h", tz="America/New_York")
    frame = pd.DataFrame({"unique_id": zone, "ds": ds, "y": 1.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_real_lmp_cache_discovery_prefers_full_daily_coverage(tmp_path: Path) -> None:
    root = tmp_path / "cache" / "pjm" / "da_hrl_lmps"
    _write_lmp_chunk(root / "aep_da_lmp_2024-01-01_2024-01-07.parquet", "2024-01-01", "2024-01-07")
    _write_lmp_chunk(root / "aep_da_lmp_2024-01-08_2024-01-14.parquet", "2024-01-08", "2024-01-14")

    found = locate_latest_lmp_cache(
        zone="AEP",
        start_date="2024-01-01",
        end_date="2024-01-14",
        cache_root=root,
    )
    assert len(found) == 2
    assert found[0].name == "aep_da_lmp_2024-01-01_2024-01-07.parquet"
    assert found[1].name == "aep_da_lmp_2024-01-08_2024-01-14.parquet"


def test_real_lmp_cache_discovery_handles_overlap_covering_current_day(tmp_path: Path) -> None:
    root = tmp_path / "cache" / "pjm" / "da_hrl_lmps"
    _write_lmp_chunk(root / "aep_da_lmp_2024-01-01_2024-01-31.parquet", "2024-01-01", "2024-01-31")
    _write_lmp_chunk(root / "aep_da_lmp_2024-02-01_2024-03-02.parquet", "2024-02-01", "2024-03-02")
    _write_lmp_chunk(root / "aep_da_lmp_2024-03-03_2024-04-02.parquet", "2024-03-03", "2024-04-02")
    _write_lmp_chunk(root / "aep_da_lmp_2024-04-03_2024-05-03.parquet", "2024-04-03", "2024-05-03")
    _write_lmp_chunk(root / "aep_da_lmp_2024-04-29_2024-05-05.parquet", "2024-04-29", "2024-05-05")
    _write_lmp_chunk(root / "aep_da_lmp_2024-05-06_2024-05-12.parquet", "2024-05-06", "2024-05-12")

    found = locate_latest_lmp_cache(
        zone="AEP",
        start_date="2024-01-01",
        end_date="2024-05-12",
        cache_root=root,
    )

    assert found[-2].name == "aep_da_lmp_2024-04-29_2024-05-05.parquet"
    assert found[-1].name == "aep_da_lmp_2024-05-06_2024-05-12.parquet"


def test_real_lmp_cache_discovery_missing_returns_empty(tmp_path: Path) -> None:
    found = locate_latest_lmp_cache(
        zone="AEP",
        start_date="2024-01-01",
        end_date="2024-01-07",
        cache_root=tmp_path / "missing",
    )
    assert found == []


def test_real_lmp_quality_report_discovery_latest_match(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    p1 = reports / "lmp_quality_AEP_20260101T000000Z.json"
    p2 = reports / "lmp_quality_AEP_20260102T000000Z.json"
    payload = {
        "zone": "AEP",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "data_source_label": "real",
        "generated_at": "2026-01-01T00:00:00Z",
    }
    p1.write_text(json.dumps(payload), encoding="utf-8")
    payload["generated_at"] = "2026-01-02T00:00:00Z"
    p2.write_text(json.dumps(payload), encoding="utf-8")

    found = locate_latest_lmp_quality_report(
        zone="AEP",
        start_date="2024-01-01",
        end_date="2024-12-31",
        report_root=reports,
    )
    assert found == p2


def test_real_weather_cache_discovery_full_window(tmp_path: Path) -> None:
    cache = tmp_path / "weather" / "openmeteo"
    cache.mkdir(parents=True, exist_ok=True)

    ds = pd.date_range(
        "2024-01-01 00:00:00",
        "2024-12-31 23:00:00",
        freq="h",
        tz="America/New_York",
    )
    frame = pd.DataFrame(
        {
            "ds": ds,
            "timezone": "America/New_York",
            "source": "openmeteo_historical_weather",
        }
    )
    out = cache / "openmeteo_AEP_2024-01-01_2024-12-31.parquet"
    frame.to_parquet(out, index=False)

    hit = locate_latest_weather_cache(
        zone="AEP",
        start_date="2024-01-01",
        end_date="2024-12-31",
        cache_root=cache,
    )
    assert hit is not None
    assert hit.path == out
    assert hit.row_count == len(frame)
