"""PJM backfill orchestration for AEP day-ahead hourly LMP."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.pjm_api import (
    PjmApiConfig,
    PjmApiError,
    effective_max_connections_per_minute,
    effective_pjm_throttle_seconds,
    fetch_da_lmp_for_zone,
    resolve_pjm_api_config,
    write_lmp_cache,
)
from lmp_forecaster.data.validation import validate_lmp_frame
from lmp_forecaster.eval.data_quality import build_lmp_quality_report, write_lmp_quality_report

BOUNDARY_ERROR_TEXT = "spans over archived and standard data"
DEFAULT_RETRY_ATTEMPTS = 4
MIN_429_COOLDOWN_SECONDS = 90.0


@dataclass(frozen=True)
class PjmBackfillConfig:
    zone: str = "AEP"
    start_date: date | None = None
    end_date: date | None = None
    chunk_days: int = 7
    row_count: int = 50000
    output_root: Path | None = None
    normalized_output_root: Path | None = None
    report_output_root: Path | None = None
    overwrite: bool = False
    dry_run: bool = True
    allow_partial_completion: bool = False
    max_retries: int = DEFAULT_RETRY_ATTEMPTS
    rate_limit_cooldown_seconds: float = MIN_429_COOLDOWN_SECONDS


@dataclass(frozen=True)
class PjmBackfillResult:
    raw_paths: list[Path]
    normalized_paths: list[Path]
    quality_report_path: Path | None
    manifest_path: Path | None
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


def _chunk_raw_path(root: Path, *, zone: str, start: date, end: date) -> Path:
    return root / f"{zone.lower()}_raw_{start.isoformat()}_{end.isoformat()}.json"


def _chunk_norm_path(root: Path, *, zone: str, start: date, end: date) -> Path:
    return root / f"{zone.lower()}_da_lmp_{start.isoformat()}_{end.isoformat()}.parquet"


def _manifest_path(reports_root: Path, *, zone: str, start: date, end: date) -> Path:
    suffix = str(start.year) if start.year == end.year else f"{start.isoformat()}_{end.isoformat()}"
    return reports_root / f"pjm_backfill_manifest_{zone.upper()}_{suffix}.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["generated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_existing_chunk(raw_path: Path, norm_path: Path) -> pd.DataFrame | None:
    if not raw_path.exists() or not norm_path.exists():
        return None
    if raw_path.stat().st_size <= 2 or norm_path.stat().st_size <= 0:
        return None

    try:
        frame = pd.read_parquet(norm_path)
    except Exception:
        return None

    if frame.empty:
        return None

    required = {"unique_id", "ds", "y", "market", "location_type", "source", "pulled_at"}
    if not required.issubset(set(frame.columns)):
        return None

    try:
        validate_lmp_frame(frame)
    except ValueError:
        return None

    return frame


def _is_rate_limit_error(exc: Exception) -> bool:
    return "429" in str(exc)


def _is_boundary_error(exc: Exception) -> bool:
    return BOUNDARY_ERROR_TEXT in str(exc).lower()


def _split_chunk(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    span_days = (end - start).days
    mid = start + timedelta(days=span_days // 2)
    return (start, mid), (mid + timedelta(days=1), end)


def _record_failed_chunk(
    manifest: dict[str, Any],
    *,
    start: date,
    end: date,
    error: Exception,
) -> None:
    manifest["chunks_failed"] = int(manifest["chunks_failed"]) + 1
    failed = list(manifest["failed_chunk_details"])
    failed.append(
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "error": str(error),
        }
    )
    manifest["failed_chunk_details"] = failed


def _apply_inter_request_throttle(state: dict[str, int], *, throttle_seconds: float) -> None:
    if state["request_count"] > 0:
        time.sleep(throttle_seconds)
    state["request_count"] += 1


def _fetch_chunk_with_retries(
    *,
    api_cfg: PjmApiConfig,
    zone: str,
    start: date,
    end: date,
    max_retries: int,
    cooldown_seconds: float,
    throttle_seconds: float,
    request_state: dict[str, int],
) -> pd.DataFrame:
    attempts = max(1, int(max_retries))

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        _apply_inter_request_throttle(request_state, throttle_seconds=throttle_seconds)
        try:
            return fetch_da_lmp_for_zone(config=api_cfg, zone=zone, start=start, end=end)
        except PjmApiError as exc:
            last_error = exc
            if _is_rate_limit_error(exc) and attempt < attempts:
                time.sleep(max(MIN_429_COOLDOWN_SECONDS, cooldown_seconds))
                continue
            raise

    raise PjmApiError(
        f"PJM chunk failed after retries for {start.isoformat()} to {end.isoformat()}: {last_error}"
    )


def run_da_lmp_backfill(config: PjmBackfillConfig) -> PjmBackfillResult:
    chunks = plan_backfill_chunks(config)
    if config.dry_run:
        return PjmBackfillResult(
            raw_paths=[],
            normalized_paths=[],
            quality_report_path=None,
            manifest_path=None,
            combined_rows=0,
        )

    api_cfg = resolve_pjm_api_config()
    effective_connections = effective_max_connections_per_minute(api_cfg)
    api_cfg = PjmApiConfig(
        api_base_url=api_cfg.api_base_url,
        api_key=api_cfg.api_key,
        timeout_seconds=api_cfg.timeout_seconds,
        row_count=config.row_count,
        max_connections_per_minute=effective_connections,
        timezone=api_cfg.timezone,
    )
    throttle_seconds = effective_pjm_throttle_seconds(api_cfg)

    paths = get_project_paths()
    raw_root = config.output_root or (
        paths.root / "data" / "raw" / "pjm" / "da_hrl_lmps"
    )
    norm_root = config.normalized_output_root or (
        paths.root / "data" / "cache" / "pjm" / "da_hrl_lmps"
    )
    reports_root = config.report_output_root or (paths.root / "data" / "cache" / "reports")
    raw_root.mkdir(parents=True, exist_ok=True)
    norm_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    if config.start_date is None or config.end_date is None:
        raise ValueError("start_date and end_date are required.")

    manifest_path = _manifest_path(
        reports_root,
        zone=config.zone,
        start=config.start_date,
        end=config.end_date,
    )

    manifest: dict[str, Any] = {
        "zone": config.zone.upper(),
        "run_start": config.start_date.isoformat(),
        "run_end": config.end_date.isoformat(),
        "chunks_planned": len(chunks),
        "chunks_completed": 0,
        "chunks_skipped_existing": 0,
        "chunks_failed": 0,
        "effective_max_connections_per_minute": effective_connections,
        "throttle_seconds": throttle_seconds,
        "failed_chunk_details": [],
        "generated_at": _now_iso(),
    }
    _write_manifest(manifest_path, manifest)

    raw_paths: list[Path] = []
    norm_paths: list[Path] = []
    pieces: list[pd.DataFrame] = []
    request_state: dict[str, int] = {"request_count": 0}
    queue: list[tuple[date, date]] = list(chunks)

    while queue:
        start, end = queue.pop(0)
        raw_path = _chunk_raw_path(raw_root, zone=config.zone, start=start, end=end)
        norm_path = _chunk_norm_path(norm_root, zone=config.zone, start=start, end=end)

        existing = None if config.overwrite else _load_existing_chunk(raw_path, norm_path)
        if existing is not None:
            raw_paths.append(raw_path)
            norm_paths.append(norm_path)
            pieces.append(existing)
            manifest["chunks_skipped_existing"] = int(manifest["chunks_skipped_existing"]) + 1
            _write_manifest(manifest_path, manifest)
            continue

        try:
            frame = _fetch_chunk_with_retries(
                api_cfg=api_cfg,
                zone=config.zone,
                start=start,
                end=end,
                max_retries=config.max_retries,
                cooldown_seconds=config.rate_limit_cooldown_seconds,
                throttle_seconds=throttle_seconds,
                request_state=request_state,
            )
        except PjmApiError as exc:
            if _is_boundary_error(exc) and start < end:
                left, right = _split_chunk(start, end)
                manifest["chunks_planned"] = int(manifest["chunks_planned"]) + 1
                _write_manifest(manifest_path, manifest)
                queue = [left, right, *queue]
                continue

            error: Exception = exc
            if _is_boundary_error(exc) and start >= end:
                error = PjmApiError(
                    "PJM request crossed archived/standard boundary for a single day "
                    f"({start.isoformat()}) and could not be split further."
                )

            _record_failed_chunk(manifest, start=start, end=end, error=error)
            _write_manifest(manifest_path, manifest)
            if config.allow_partial_completion:
                continue
            raise error from exc

        if frame.empty:
            empty_error = PjmApiError(
                f"PJM chunk produced no rows for {start.isoformat()} to {end.isoformat()}."
            )
            _record_failed_chunk(manifest, start=start, end=end, error=empty_error)
            _write_manifest(manifest_path, manifest)
            if config.allow_partial_completion:
                continue
            raise empty_error

        frame.to_json(raw_path, orient="records", date_format="iso", indent=2)
        written_norm_path = write_lmp_cache(
            frame,
            zone=config.zone,
            start=start,
            end=end,
            output_root=norm_root,
        )

        raw_paths.append(raw_path)
        norm_paths.append(written_norm_path)
        pieces.append(frame)
        manifest["chunks_completed"] = int(manifest["chunks_completed"]) + 1
        _write_manifest(manifest_path, manifest)

    combined = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(
            columns=[
                "unique_id",
                "ds",
                "y",
                "market",
                "location_type",
                "source",
                "pulled_at",
                "pnode_name",
                "pnode_type",
            ]
        )
    )

    if not combined.empty:
        validate_lmp_frame(combined)

    quality = build_lmp_quality_report(combined, zone=config.zone)
    report_path = write_lmp_quality_report(quality, output_dir=reports_root)

    _write_manifest(manifest_path, manifest)

    return PjmBackfillResult(
        raw_paths=raw_paths,
        normalized_paths=norm_paths,
        quality_report_path=report_path,
        manifest_path=manifest_path,
        combined_rows=int(len(combined)),
    )
