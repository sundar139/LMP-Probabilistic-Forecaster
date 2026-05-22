"""Discovery helpers for real cached LMP/weather inputs and quality reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths

_LMP_CHUNK_RE = re.compile(
    r"^(?P<zone>[a-z0-9]+)_da_lmp_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CacheChunk:
    path: Path
    zone: str
    start: date
    end: date


@dataclass(frozen=True)
class WeatherCacheHit:
    path: Path
    row_count: int
    start_ds: str
    end_ds: str


def _parse_lmp_chunk(path: Path) -> CacheChunk | None:
    m = _LMP_CHUNK_RE.match(path.name)
    if not m:
        return None
    return CacheChunk(
        path=path,
        zone=m.group("zone").upper(),
        start=date.fromisoformat(m.group("start")),
        end=date.fromisoformat(m.group("end")),
    )


def _default_lmp_cache_root() -> Path:
    return get_project_paths().root / "data" / "cache" / "pjm" / "da_hrl_lmps"


def _default_weather_cache_root() -> Path:
    return get_project_paths().root / "data" / "cache" / "weather" / "openmeteo"


def _default_report_root() -> Path:
    return get_project_paths().root / "data" / "cache" / "reports"


def locate_latest_lmp_cache(
    *,
    zone: str,
    start_date: str,
    end_date: str,
    cache_root: Path | None = None,
) -> list[Path]:
    """Locate deterministic LMP chunk paths covering a date window.

    Strategy: walk day-by-day and choose the best chunk covering each day,
    preferring fresher cache writes over stale legacy overlap fragments.
    """
    root = cache_root or _default_lmp_cache_root()
    zone_u = zone.upper()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if not root.exists():
        return []

    chunks: list[CacheChunk] = []
    for path in sorted(root.glob("*.parquet")):
        parsed = _parse_lmp_chunk(path)
        if parsed is None or parsed.zone != zone_u:
            continue
        if parsed.end < start or parsed.start > end:
            continue
        chunks.append(parsed)

    selected: list[Path] = []
    cur = start
    while cur <= end:
        candidates = [c for c in chunks if c.start <= cur <= c.end and c.end <= end]
        if not candidates:
            return []

        candidates.sort(
            key=lambda c: (
                -c.path.stat().st_mtime,
                (c.end - c.start).days,
                c.path.name.lower(),
            )
        )
        chosen = candidates[0]
        selected.append(chosen.path)
        cur = chosen.end + timedelta(days=1)

    return selected


def _pick_latest_report(
    *,
    prefix: str,
    zone: str,
    start_date: str,
    end_date: str,
    report_root: Path | None = None,
    source_label_keys: tuple[str, ...] = (),
) -> Path | None:
    root = report_root or _default_report_root()
    if not root.exists():
        return None

    zone_u = zone.upper()
    matches: list[tuple[pd.Timestamp, Path]] = []

    for path in root.glob(f"{prefix}_{zone_u}_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if str(payload.get("zone", "")).upper() != zone_u:
            continue
        if str(payload.get("start_date", "")) != start_date:
            continue
        if str(payload.get("end_date", "")) != end_date:
            continue

        if source_label_keys:
            labels = {str(payload.get(k, "")).lower() for k in source_label_keys}
            if "real" not in labels and "mixed-real" not in labels:
                continue

        stamp = pd.to_datetime(payload.get("generated_at"), errors="coerce", utc=True)
        if pd.isna(stamp):
            stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        matches.append((stamp, path))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def locate_latest_lmp_quality_report(
    *,
    zone: str,
    start_date: str,
    end_date: str,
    report_root: Path | None = None,
) -> Path | None:
    return _pick_latest_report(
        prefix="lmp_quality",
        zone=zone,
        start_date=start_date,
        end_date=end_date,
        report_root=report_root,
        source_label_keys=("data_source_label",),
    )


def locate_latest_weather_quality_report(
    *,
    zone: str,
    start_date: str,
    end_date: str,
    report_root: Path | None = None,
) -> Path | None:
    return _pick_latest_report(
        prefix="weather_quality",
        zone=zone,
        start_date=start_date,
        end_date=end_date,
        report_root=report_root,
        source_label_keys=("data_source_label", "source_label"),
    )


def locate_latest_weather_cache(
    *,
    zone: str,
    start_date: str,
    end_date: str,
    cache_root: Path | None = None,
) -> WeatherCacheHit | None:
    root = cache_root or _default_weather_cache_root()
    if not root.exists():
        return None

    zone_u = zone.upper()
    direct = root / f"openmeteo_{zone_u}_{start_date}_{end_date}.parquet"
    candidates = [direct] if direct.exists() else sorted(root.glob("*.parquet"))

    requested_start = pd.Timestamp(f"{start_date} 00:00:00", tz="America/New_York")
    requested_end = pd.Timestamp(f"{end_date} 23:00:00", tz="America/New_York")

    best: WeatherCacheHit | None = None
    for path in candidates:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if "ds" not in frame.columns:
            continue
        ds = pd.to_datetime(frame["ds"], errors="coerce", utc=False)
        if ds.isna().any():
            continue
        if ds.dt.tz is None:
            ds = ds.dt.tz_localize(
                "America/New_York",
                ambiguous="infer",
                nonexistent="shift_forward",
            )
        else:
            ds = ds.dt.tz_convert("America/New_York")

        if ds.min() > requested_start or ds.max() < requested_end:
            continue

        hit = WeatherCacheHit(
            path=path,
            row_count=int(len(frame)),
            start_ds=str(ds.min()),
            end_ds=str(ds.max()),
        )
        if best is None or path.stat().st_mtime > best.path.stat().st_mtime:
            best = hit

    return best
