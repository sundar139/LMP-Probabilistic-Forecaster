"""Tests for baseline results summary report utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lmp_forecaster.eval.results_report import (
    summarize_baseline_results,
    write_baseline_results_json,
    write_baseline_results_markdown,
)


def test_results_summary_includes_required_metrics_and_fields(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP"],
            "ds": pd.date_range("2024-12-01", periods=2, freq="h", tz="America/New_York"),
            "y": [20.0, 22.0],
            "source_label": ["real", "real"],
        }
    )
    metrics = {
        "TFT": {
            "model": "TFT",
            "zone": "AEP",
            "row_count": 24,
            "mae": 1.0,
            "rmse": 1.5,
            "pinball_p10": 0.3,
            "pinball_p50": 0.5,
            "pinball_p90": 0.7,
            "mean_pinball_loss": 0.5,
            "coverage_80": 0.8,
            "interval_width_mean": 3.0,
            "generated_at": "2026-01-01T00:00:00Z",
            "data_source_label": "real",
        }
    }
    summary = summarize_baseline_results(
        zone="AEP",
        data_source_label="real",
        panel=panel,
        split_sizes={"train": 100, "val": 20, "test": 24},
        metrics=metrics,
        forecasts={"TFT": "x.parquet"},
        accelerator_kind="cpu",
        accelerator_device_name="CPU",
        training_duration_seconds=12.3,
    )

    assert summary["data_source_label"] == "real"
    assert summary["split_sizes"]["validation"] == 20
    assert summary["forecast_schema_validation"] == "passed"

    out_json = write_baseline_results_json(summary, output_dir=tmp_path)
    out_md = write_baseline_results_markdown(summary, output_dir=tmp_path)
    assert out_json.exists()
    assert out_md.exists()

    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["zone"] == "AEP"
    assert loaded["metrics"][0]["model"] == "TFT"
