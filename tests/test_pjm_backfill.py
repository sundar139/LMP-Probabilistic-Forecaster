"""Tests for PJM backfill orchestration."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import lmp_forecaster.data.pjm_backfill as backfill
from lmp_forecaster.data.pjm_api import PjmApiConfig, PjmApiError
from lmp_forecaster.data.pjm_backfill import (
    PjmBackfillConfig,
    plan_backfill_chunks,
    run_da_lmp_backfill,
    validate_backfill_completeness,
)


def _frame_for_range(start: date, end: date, zone: str = "AEP") -> pd.DataFrame:
    ds = pd.date_range(
        f"{start.isoformat()} 00:00:00",
        f"{end.isoformat()} 23:00:00",
        freq="h",
        tz="America/New_York",
    )
    return pd.DataFrame(
        {
            "unique_id": [zone.upper()] * len(ds),
            "ds": ds,
            "y": [float(i) for i in range(len(ds))],
            "market": ["DAY_AHEAD"] * len(ds),
            "location_type": ["ZONE"] * len(ds),
            "source": ["pjm_api_da_hrl_lmps"] * len(ds),
            "pulled_at": [pd.Timestamp("2024-01-01T00:00:00Z")] * len(ds),
            "pnode_name": [zone.upper()] * len(ds),
            "pnode_type": ["ZONE"] * len(ds),
        }
    )


def _mock_api_config() -> PjmApiConfig:
    return PjmApiConfig(
        api_base_url="https://api.pjm.com/api/v1",
        api_key="dummy",
        timeout_seconds=30.0,
        row_count=50000,
        max_connections_per_minute=5,
        timezone="America/New_York",
    )


def test_plan_backfill_chunks_monthly_like_windows() -> None:
    cfg = PjmBackfillConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 15),
        chunk_days=7,
    )
    chunks = plan_backfill_chunks(cfg)
    assert len(chunks) >= 11
    assert chunks[0][0] == date(2024, 1, 1)
    assert chunks[-1][1] == date(2024, 3, 15)


def test_validate_backfill_completeness_flags_duplicates_and_missing() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP", "AEP"],
            "ds": pd.to_datetime(
                [
                    "2024-01-01 00:00:00-05:00",
                    "2024-01-01 00:00:00-05:00",
                    "2024-01-01 02:00:00-05:00",
                ]
            ),
            "y": [10.0, 11.0, 12.0],
            "market": ["DA", "DA", "DA"],
            "location_type": ["ZONE", "ZONE", "ZONE"],
            "source": ["pjm", "pjm", "pjm"],
            "pulled_at": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"], utc=True),
        }
    )
    out = validate_backfill_completeness(df, zone="AEP")
    assert out["duplicate_timestamp_count"] == 1
    assert out["missing_hour_count"] >= 1


def test_validate_backfill_requires_required_columns() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_backfill_completeness(pd.DataFrame({"x": [1]}), zone="AEP")


def test_plan_backfill_chunks_default_is_weekly_like() -> None:
    cfg = PjmBackfillConfig(start_date=date(2024, 1, 1), end_date=date(2024, 1, 8))
    chunks = plan_backfill_chunks(cfg)
    assert chunks == [(date(2024, 1, 1), date(2024, 1, 7)), (date(2024, 1, 8), date(2024, 1, 8))]


def test_backfill_resume_skips_existing_valid_chunks(monkeypatch, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    norm_root = tmp_path / "norm"
    reports_root = tmp_path / "reports"
    raw_root.mkdir(parents=True)
    norm_root.mkdir(parents=True)
    reports_root.mkdir(parents=True)

    start_a, end_a = date(2024, 1, 1), date(2024, 1, 7)
    existing = _frame_for_range(start_a, end_a)
    raw_a = raw_root / f"aep_raw_{start_a.isoformat()}_{end_a.isoformat()}.json"
    norm_a = norm_root / f"aep_da_lmp_{start_a.isoformat()}_{end_a.isoformat()}.parquet"
    existing.to_json(raw_a, orient="records", date_format="iso", indent=2)
    existing.to_parquet(norm_a, index=False)

    monkeypatch.setattr(backfill, "resolve_pjm_api_config", _mock_api_config)
    sleeps: list[float] = []
    monkeypatch.setattr(backfill.time, "sleep", lambda s: sleeps.append(float(s)))

    calls: list[tuple[date, date]] = []

    def fake_fetch(*, config: PjmApiConfig, zone: str, start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        return _frame_for_range(start, end, zone=zone)

    monkeypatch.setattr(backfill, "fetch_da_lmp_for_zone", fake_fetch)

    cfg = PjmBackfillConfig(
        zone="AEP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 8),
        chunk_days=7,
        output_root=raw_root,
        normalized_output_root=norm_root,
        report_output_root=reports_root,
        dry_run=False,
    )
    result = run_da_lmp_backfill(cfg)

    assert calls == [(date(2024, 1, 8), date(2024, 1, 8))]
    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunks_planned"] == 2
    assert manifest["chunks_skipped_existing"] == 1
    assert manifest["chunks_completed"] == 1
    assert manifest["chunks_failed"] == 0


def test_backfill_resume_redownloads_invalid_existing_chunk(monkeypatch, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    norm_root = tmp_path / "norm"
    reports_root = tmp_path / "reports"
    raw_root.mkdir(parents=True)
    norm_root.mkdir(parents=True)
    reports_root.mkdir(parents=True)

    start_a, end_a = date(2024, 11, 1), date(2024, 11, 1)
    bad = _frame_for_range(start_a, end_a)
    bad.loc[0, "ds"] = pd.NaT
    raw_a = raw_root / f"aep_raw_{start_a.isoformat()}_{end_a.isoformat()}.json"
    norm_a = norm_root / f"aep_da_lmp_{start_a.isoformat()}_{end_a.isoformat()}.parquet"
    bad.to_json(raw_a, orient="records", date_format="iso", indent=2)
    bad.to_parquet(norm_a, index=False)

    monkeypatch.setattr(backfill, "resolve_pjm_api_config", _mock_api_config)
    monkeypatch.setattr(backfill.time, "sleep", lambda _s: None)

    calls: list[tuple[date, date]] = []

    def fake_fetch(*, config: PjmApiConfig, zone: str, start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        return _frame_for_range(start, end, zone=zone)

    monkeypatch.setattr(backfill, "fetch_da_lmp_for_zone", fake_fetch)

    cfg = PjmBackfillConfig(
        zone="AEP",
        start_date=start_a,
        end_date=end_a,
        chunk_days=1,
        output_root=raw_root,
        normalized_output_root=norm_root,
        report_output_root=reports_root,
        dry_run=False,
    )

    result = run_da_lmp_backfill(cfg)
    assert result.combined_rows == 24
    assert calls == [(start_a, end_a)]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunks_skipped_existing"] == 0
    assert manifest["chunks_completed"] == 1


def test_boundary_error_multi_day_chunk_splits_and_preserves_coverage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    norm_root = tmp_path / "norm"
    reports_root = tmp_path / "reports"

    monkeypatch.setattr(backfill, "resolve_pjm_api_config", _mock_api_config)
    monkeypatch.setattr(backfill.time, "sleep", lambda _s: None)

    calls: list[tuple[date, date]] = []

    def fake_fetch(*, config: PjmApiConfig, zone: str, start: date, end: date) -> pd.DataFrame:
        calls.append((start, end))
        if start < end:
            raise PjmApiError("PJM API request failed (400): spans over archived and standard data")
        return _frame_for_range(start, end, zone=zone)

    monkeypatch.setattr(backfill, "fetch_da_lmp_for_zone", fake_fetch)

    cfg = PjmBackfillConfig(
        zone="AEP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 4),
        chunk_days=4,
        output_root=raw_root,
        normalized_output_root=norm_root,
        report_output_root=reports_root,
        dry_run=False,
    )

    result = run_da_lmp_backfill(cfg)
    assert result.combined_rows == 96
    success_days = {s for s, e in calls if s == e}
    assert success_days == {date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunks_planned"] == 4
    assert manifest["chunks_completed"] == 4
    assert manifest["chunks_failed"] == 0


def test_boundary_error_single_day_raises_clear_error(monkeypatch, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    norm_root = tmp_path / "norm"
    reports_root = tmp_path / "reports"

    monkeypatch.setattr(backfill, "resolve_pjm_api_config", _mock_api_config)
    monkeypatch.setattr(backfill.time, "sleep", lambda _s: None)

    def always_boundary(*, config: PjmApiConfig, zone: str, start: date, end: date) -> pd.DataFrame:
        raise PjmApiError("PJM API request failed (400): spans over archived and standard data")

    monkeypatch.setattr(backfill, "fetch_da_lmp_for_zone", always_boundary)

    cfg = PjmBackfillConfig(
        zone="AEP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        chunk_days=1,
        output_root=raw_root,
        normalized_output_root=norm_root,
        report_output_root=reports_root,
        dry_run=False,
    )

    with pytest.raises(PjmApiError, match="single day"):
        run_da_lmp_backfill(cfg)

    manifest_path = reports_root / "pjm_backfill_manifest_AEP_2024.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunks_failed"] == 1


def test_429_retries_apply_cooldown_and_throttle(monkeypatch, tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    norm_root = tmp_path / "norm"
    reports_root = tmp_path / "reports"

    monkeypatch.setattr(backfill, "resolve_pjm_api_config", _mock_api_config)

    sleeps: list[float] = []
    monkeypatch.setattr(backfill.time, "sleep", lambda s: sleeps.append(float(s)))

    attempts = {"count": 0}

    def flaky_fetch(*, config: PjmApiConfig, zone: str, start: date, end: date) -> pd.DataFrame:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PjmApiError("PJM API request failed (429): rate limit")
        return _frame_for_range(start, end, zone=zone)

    monkeypatch.setattr(backfill, "fetch_da_lmp_for_zone", flaky_fetch)

    cfg = PjmBackfillConfig(
        zone="AEP",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        chunk_days=1,
        output_root=raw_root,
        normalized_output_root=norm_root,
        report_output_root=reports_root,
        dry_run=False,
    )

    result = run_da_lmp_backfill(cfg)
    assert result.combined_rows == 24

    cooldown_sleeps = [s for s in sleeps if s >= 90.0]
    assert len(cooldown_sleeps) == 2
    throttle_sleeps = [s for s in sleeps if 12.0 <= s < 90.0]
    assert len(throttle_sleeps) >= 1

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["effective_max_connections_per_minute"] <= 5
