"""Tests for panel summary reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lmp_forecaster.eval.panel_report import build_panel_summary, write_panel_summary


def test_panel_summary_includes_expected_fields(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP"],
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=2,
                freq="h",
                tz="America/New_York",
            ),
            "y": [10.0, 11.0],
            "temperature_2m": [1.0, None],
            "temperature_2m_missing": [0, 1],
            "source_label": ["synthetic", "synthetic"],
        }
    )

    summary = build_panel_summary(panel, zone="AEP")
    assert summary["zone"] == "AEP"
    assert summary["row_count"] == 2
    assert "start_ds" in summary
    assert "end_ds" in summary
    assert "weather_missingness_counts" in summary
    assert "source_labels" in summary

    path = write_panel_summary(summary, output_dir=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["row_count"] == 2
