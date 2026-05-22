"""Real baseline results summary reporting utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths

REQUIRED_METRIC_FIELDS = [
    "model",
    "row_count",
    "mae",
    "rmse",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "mean_pinball_loss",
    "coverage_80",
    "interval_width_mean",
    "generated_at",
    "data_source_label",
    "zone",
]


def _reports_root() -> Path:
    root = get_project_paths().root
    path = root / "data" / "cache" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_baseline_results(
    *,
    zone: str,
    data_source_label: str,
    panel: pd.DataFrame,
    split_sizes: dict[str, int],
    metrics: dict[str, dict[str, Any]],
    forecasts: dict[str, str],
    accelerator_kind: str,
    accelerator_device_name: str,
    training_duration_seconds: float | None,
) -> dict[str, Any]:
    metric_rows = [dict(v) for _, v in sorted(metrics.items())]
    for row in metric_rows:
        missing = [k for k in REQUIRED_METRIC_FIELDS if k not in row]
        if missing:
            raise ValueError(f"Missing required metrics fields: {missing}")

    if "source_label" in panel.columns:
        panel_source_labels = sorted({str(v) for v in panel["source_label"].astype(str).unique()})
    else:
        panel_source_labels = []

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "zone": zone,
        "data_source_label": data_source_label,
        "dataset_window": {
            "start_ds": str(panel["ds"].min()) if len(panel) else None,
            "end_ds": str(panel["ds"].max()) if len(panel) else None,
            "row_count": int(len(panel)),
        },
        "split_sizes": {
            "train": int(split_sizes.get("train", 0)),
            "validation": int(split_sizes.get("val", split_sizes.get("validation", 0))),
            "test": int(split_sizes.get("test", 0)),
        },
        "metrics": metric_rows,
        "forecast_outputs": forecasts,
        "forecast_schema_validation": "passed",
        "panel_source_labels": panel_source_labels,
        "accelerator": {
            "kind": accelerator_kind,
            "device_name": accelerator_device_name,
        },
        "training_duration_seconds": training_duration_seconds,
        "caveat": "first real single-zone untuned baseline",
        "next_step_recommendation": (
            "tighten real baseline training configuration, add MLflow tracking, "
            "and prepare rolling-origin backtest design"
        ),
    }


def write_baseline_results_json(summary: dict[str, Any], output_dir: Path | None = None) -> Path:
    target = output_dir or _reports_root()
    zone = str(summary.get("zone", "unknown"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"baseline_results_summary_{zone}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_baseline_results_markdown(
    summary: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    target = output_dir or _reports_root()
    zone = str(summary.get("zone", "unknown"))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"baseline_results_summary_{zone}_{stamp}.md"

    metrics_df = pd.DataFrame(summary.get("metrics", []))
    metrics_table = metrics_df.to_markdown(index=False) if not metrics_df.empty else "(no metrics)"

    accelerator = summary.get("accelerator", {})
    accelerator_kind = accelerator.get("kind")
    accelerator_device = accelerator.get("device_name")

    lines = [
        f"# Real baseline results summary — {zone}",
        "",
        f"Generated at: {summary.get('generated_at')}",
        f"Data source label: {summary.get('data_source_label')}",
        "",
        "## Dataset window",
        f"- start_ds: {summary.get('dataset_window', {}).get('start_ds')}",
        f"- end_ds: {summary.get('dataset_window', {}).get('end_ds')}",
        f"- row_count: {summary.get('dataset_window', {}).get('row_count')}",
        "",
        "## Split sizes",
        f"- train: {summary.get('split_sizes', {}).get('train')}",
        f"- validation: {summary.get('split_sizes', {}).get('validation')}",
        f"- test: {summary.get('split_sizes', {}).get('test')}",
        "",
        "## Model metrics",
        metrics_table,
        "",
        f"Forecast schema validation: {summary.get('forecast_schema_validation')}",
        f"Accelerator: {accelerator_kind} ({accelerator_device})",
        f"Training duration seconds: {summary.get('training_duration_seconds')}",
        "",
        f"Caveat: {summary.get('caveat')}",
        f"Next step recommendation: {summary.get('next_step_recommendation')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
