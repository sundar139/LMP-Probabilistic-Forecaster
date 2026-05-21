"""Panel summary reporting utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.validation import detect_dst_day_lengths
from lmp_forecaster.features.weather import WEATHER_COLUMNS


def build_panel_summary(panel: pd.DataFrame, *, zone: str) -> dict[str, Any]:
    """Build summary statistics for a single-zone panel."""
    weather_missingness = {
        col: int(panel[col].isna().sum()) if col in panel.columns else 0 for col in WEATHER_COLUMNS
    }
    feature_null_counts = {col: int(panel[col].isna().sum()) for col in panel.columns}

    dst_summary = [
        {"day": item.day, "hours": item.hours}
        for item in detect_dst_day_lengths(panel["ds"], timezone="America/New_York")
        if item.hours in {23, 25}
    ]

    source_labels = sorted({str(v) for v in panel.get("source_label", pd.Series(["unknown"]))})

    return {
        "zone": zone,
        "row_count": int(len(panel)),
        "start_ds": str(panel["ds"].min()) if len(panel) > 0 else None,
        "end_ds": str(panel["ds"].max()) if len(panel) > 0 else None,
        "duplicate_timestamps": int(panel.duplicated(subset=["unique_id", "ds"]).sum()),
        "missing_target_values": int(panel["y"].isna().sum()) if "y" in panel.columns else 0,
        "weather_missingness_counts": weather_missingness,
        "feature_null_counts": feature_null_counts,
        "dst_day_lengths": dst_summary,
        "y_min": float(panel["y"].min()) if "y" in panel.columns else None,
        "y_max": float(panel["y"].max()) if "y" in panel.columns else None,
        "y_mean": float(panel["y"].mean()) if "y" in panel.columns else None,
        "y_std": float(panel["y"].std()) if "y" in panel.columns else None,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_labels": source_labels,
    }


def write_panel_summary(summary: dict[str, Any], output_dir: Path | None = None) -> Path:
    """Write summary JSON to cache report directory."""
    root = get_project_paths().root
    out_dir = output_dir or (root / "data" / "cache" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    zone = str(summary.get("zone", "unknown"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"panel_summary_{zone}_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
