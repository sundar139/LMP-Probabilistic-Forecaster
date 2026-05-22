"""Tests for LMP data quality reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lmp_forecaster.eval.data_quality import build_lmp_quality_report, write_lmp_quality_report


def _sample_lmp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP", "AEP"],
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=3,
                freq="h",
                tz="America/New_York",
            ),
            "y": [-10.0, 0.0, 600.0],
            "pnode_name": ["AEP", "AEP", "AEP"],
            "pnode_type": ["ZONE", "ZONE", "ZONE"],
            "source": ["pjm_api", "pjm_api", "pjm_api"],
        }
    )


def test_quality_report_has_expected_fields_and_counts() -> None:
    report = build_lmp_quality_report(_sample_lmp(), zone="AEP")
    assert report["source"] == "pjm_api_da_hrl_lmps"
    assert report["zone"] == "AEP"
    assert report["row_count"] == 3
    assert report["expected_hour_count"] == 3
    assert report["missing_hour_count"] == 0
    assert report["negative_price_count"] == 1
    assert report["zero_price_count"] == 1
    assert report["extreme_price_count"] == 1
    assert report["duplicate_timestamp_count"] == 0
    assert report["timezone"] == "America/New_York"
    assert report["pnode_names"] == ["AEP"]
    assert report["pnode_types"] == ["ZONE"]
    assert report["data_source_label"] == "real"


def test_quality_report_rejects_null_required_columns() -> None:
    frame = _sample_lmp()
    frame.loc[0, "unique_id"] = None
    with pytest.raises(ValueError, match="null"):
        build_lmp_quality_report(frame, zone="AEP")


def test_write_quality_report_creates_json(tmp_path: Path) -> None:
    report = build_lmp_quality_report(_sample_lmp(), zone="AEP")
    out = write_lmp_quality_report(report, output_dir=tmp_path)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["zone"] == "AEP"
