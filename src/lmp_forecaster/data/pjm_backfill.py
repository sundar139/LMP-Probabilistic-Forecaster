"""PJM backfill orchestration for AEP day-ahead hourly LMP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.pjm_api import (
    PjmApiConfig,
    fetch_da_lmp_for_zone,
    resolve_pjm_api_config,
    write_lmp_cache,
)
from lmp_forecaster.eval.data_quality import build_lmp_quality_report, write_lmp_quality_report


@dataclass(frozen=True)
class PjmBackfillConfig:
    zone: str = "AEP"
    start_date: date | None = None
    end_date: date | None = None
    chunk_days: int = 31
    row_count: int = 50000
    output_root: Path | None = None
    normalized_output_root: Path | None = None
    overwrite: bool = False
    dry_run: bool = True


@dataclass(frozen=True)
class PjmBackfillResult:
    raw_paths: list[Path]
    normalized_paths: list[Path]
    quality_report_path: Path | None
    combined_rows: int


def plan_backfill_chunks(config: PjmBackfillConfig) -> list[tuple[date, date]]:
    if config.start_date is None or config.end_date is None:
        raise ValueError("start_date and end_date are required for backfill chunk planning.")
    if config.start_date > config.end_date:
        raise ValueError("start_date must be <= end_date")

    chunks: list[tuple[date, date]] = []
    cur = config.start_date
    while cur <= config.end_date:
        nxt = min(config.end_date, cur + timedelta(days=config.chunk_days - 1))
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


def validate_backfill_completeness(frame: pd.DataFrame, *, zone: str) -> dict[str, Any]:
    required = ["unique_id", "ds", "y", "market", "location_type", "source", "pulled_at"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Backfill frame missing required columns: {missing}")

    out = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    ds = pd.to_datetime(out["ds"], errors="coerce", utc=False)
    if ds.isna().any():
        raise ValueError("Backfill frame contains invalid ds values.")

    min_ds = ds.min()
    max_ds = ds.max()
    expected = len(pd.date_range(min_ds, max_ds, freq="h")) if len(out) > 0 else 0
    duplicates = int(out.duplicated(subset=["unique_id", "ds"]).sum())
    observed_unique_hours = int(out.drop_duplicates(subset=["unique_id", "ds"]).shape[0])

    return {
        "zone": zone,
        "row_count": int(len(out)),
        "expected_hour_count": int(expected),
        "missing_hour_count": int(max(0, expected - observed_unique_hours)),
        "duplicate_timestamp_count": duplicates,
    }


def run_da_lmp_backfill(config: PjmBackfillConfig) -> PjmBackfillResult:
    chunks = plan_backfill_chunks(config)
    if config.dry_run:
        return PjmBackfillResult(
            raw_paths=[],
            normalized_paths=[],
            quality_report_path=None,
            combined_rows=0,
        )

    api_cfg = resolve_pjm_api_config()
    api_cfg = PjmApiConfig(
        api_base_url=api_cfg.api_base_url,
        api_key=api_cfg.api_key,
        timeout_seconds=api_cfg.timeout_seconds,
        row_count=config.row_count,
        max_connections_per_minute=api_cfg.max_connections_per_minute,
        timezone=api_cfg.timezone,
    )

    paths = get_project_paths()
    raw_root = config.output_root or (
        paths.root / "data" / "raw" / "pjm" / "da_hrl_lmps"
    )
    norm_root = config.normalized_output_root or (
        paths.root / "data" / "cache" / "pjm" / "da_hrl_lmps"
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    norm_root.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    norm_paths: list[Path] = []
    pieces: list[pd.DataFrame] = []

    for start, end in chunks:
        df = fetch_da_lmp_for_zone(config=api_cfg, zone=config.zone, start=start, end=end)
        raw_path = (
            raw_root / f"{config.zone.lower()}_raw_{start.isoformat()}_{end.isoformat()}.json"
        )
        norm_path = write_lmp_cache(
            df,
            zone=config.zone,
            start=start,
            end=end,
            output_root=norm_root,
        )

        if not raw_path.exists() or config.overwrite:
            df.to_json(raw_path, orient="records", date_format="iso", indent=2)

        raw_paths.append(raw_path)
        norm_paths.append(norm_path)
        pieces.append(df)

    combined = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["unique_id", "ds", "y"])
    )
    quality = build_lmp_quality_report(combined, zone=config.zone)
    report_path = write_lmp_quality_report(quality)

    return PjmBackfillResult(
        raw_paths=raw_paths,
        normalized_paths=norm_paths,
        quality_report_path=report_path,
        combined_rows=int(len(combined)),
    )
