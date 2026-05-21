"""Tests for PJM LMP ingestion adapter."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lmp_forecaster.data.pjm_lmp import (
    PjmLmpRequestConfig,
    build_day_ahead_lmp_request,
    normalize_pjm_lmp_rows,
    plan_pjm_smoke_output_path,
)
from lmp_forecaster.data.validation import validate_lmp_frame


def test_pjm_normalization_produces_required_core_columns() -> None:
    records = [
        {
            "pnode_name": "AEP",
            "datetime_beginning_ept": "2024-01-01 00:00:00 EPT",
            "total_lmp_da": "31.5",
            "market_type": "DA",
            "location_type": "ZONE",
        },
        {
            "pnode_name": "AEP",
            "datetime_beginning_ept": "2024-01-01 01:00:00 EPT",
            "total_lmp_da": "32.7",
            "market_type": "DA",
            "location_type": "ZONE",
        },
    ]

    frame = normalize_pjm_lmp_rows(records, fallback_location_type="ZONE")

    assert {"unique_id", "ds", "y"}.issubset(frame.columns)
    assert frame["unique_id"].tolist() == ["AEP", "AEP"]
    assert frame["y"].tolist() == [31.5, 32.7]
    validate_lmp_frame(frame)


def test_pjm_validator_catches_missing_y() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["AEP"],
            "ds": [pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York")],
            "y": [pd.NA],
            "market": ["DAY_AHEAD"],
            "location_type": ["ZONE"],
            "source": ["pjm"],
            "pulled_at": [pd.Timestamp("2024-01-02T00:00:00Z")],
        }
    )

    with pytest.raises(ValueError, match="Required column has nulls: y"):
        validate_lmp_frame(frame)


def test_build_request_and_output_path_deterministic() -> None:
    cfg = PjmLmpRequestConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        locations=["AEP", "ATSI"],
        max_rows=100,
    )

    url, params = build_day_ahead_lmp_request(cfg)
    path = plan_pjm_smoke_output_path(cfg)

    assert "da_hrl_lmps" in url
    assert params["location_type"] == "ZONE"
    assert params["rowCount"] == 100
    assert "data" in str(path)
    assert "cache" in str(path)
    assert path.suffix == ".parquet"
