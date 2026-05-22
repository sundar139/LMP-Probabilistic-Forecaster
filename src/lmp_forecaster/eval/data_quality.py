"""Data quality reporting for real LMP ingestion outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.validation import detect_dst_day_lengths


def build_lmp_quality_report(
    frame: pd.DataFrame,
    *,
    zone: str,
    source: str = "pjm_api_da_hrl_lmps",
    extreme_threshold: float = 500.0,
) -> dict[str, Any]:
    required = ["unique_id", "ds", "y"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"LMP quality input missing required columns: {missing}")

    out = frame.copy()
    for col in required:
        if out[col].isna().any():
            raise ValueError(f"LMP quality input has null values in required column: {col}")

    out["ds"] = pd.to_datetime(out["ds"], errors="coerce", utc=False)
    if out["ds"].isna().any():
        raise ValueError("LMP quality input has invalid ds values.")

    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if out["y"].isna().any():
        raise ValueError("LMP quality input has invalid y values.")

    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    duplicates = int(out.duplicated(subset=["unique_id", "ds"]).sum())
    min_ds = out["ds"].min()
    max_ds = out["ds"].max()
    unique_hours = int(out.drop_duplicates(subset=["unique_id", "ds"]).shape[0])
    expected_hours = len(pd.date_range(min_ds, max_ds, freq="h")) if len(out) > 0 else 0
    missing_hours = int(max(0, expected_hours - unique_hours))

    dst = [
        {"day": d.day, "hours": d.hours}
        for d in detect_dst_day_lengths(out["ds"], timezone="America/New_York")
        if d.hours in {23, 25}
    ]

    pnode_name_series = out.get("pnode_name", pd.Series(dtype=str)).dropna()
    pnode_type_series = out.get("pnode_type", pd.Series(dtype=str)).dropna()
    pnode_names = sorted({str(v) for v in pnode_name_series.unique()})
    pnode_types = sorted({str(v) for v in pnode_type_series.unique()})

    return {
        "source": source,
        "zone": zone,
        "start_date": str(min_ds.date()) if len(out) else None,
        "end_date": str(max_ds.date()) if len(out) else None,
        "row_count": int(len(out)),
        "expected_hour_count": int(expected_hours),
        "missing_hour_count": missing_hours,
        "duplicate_timestamp_count": duplicates,
        "min_ds": str(min_ds) if len(out) else None,
        "max_ds": str(max_ds) if len(out) else None,
        "timezone": str(out["ds"].dt.tz),
        "dst_day_lengths": dst,
        "y_min": float(out["y"].min()) if len(out) else None,
        "y_max": float(out["y"].max()) if len(out) else None,
        "y_mean": float(out["y"].mean()) if len(out) else None,
        "y_std": float(out["y"].std()) if len(out) else None,
        "negative_price_count": int((out["y"] < 0).sum()),
        "zero_price_count": int((out["y"] == 0).sum()),
        "extreme_price_count": int((out["y"].abs() > extreme_threshold).sum()),
        "pnode_names": pnode_names,
        "pnode_types": pnode_types,
        "columns_observed": list(out.columns),
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source_label": "real",
    }


def write_lmp_quality_report(report: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = get_project_paths().root
    target = output_dir or (root / "data" / "cache" / "reports")
    target.mkdir(parents=True, exist_ok=True)

    zone = str(report.get("zone", "unknown"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"lmp_quality_{zone}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
