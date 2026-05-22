"""Tests for PJM backfill orchestration."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from lmp_forecaster.data.pjm_backfill import (
    PjmBackfillConfig,
    plan_backfill_chunks,
    validate_backfill_completeness,
)


def test_plan_backfill_chunks_monthly_like_windows() -> None:
    cfg = PjmBackfillConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 15),
        chunk_days=31,
    )
    chunks = plan_backfill_chunks(cfg)
    assert len(chunks) >= 3
    assert chunks[0][0] == date(2024, 1, 1)
    assert chunks[-1][1] == date(2024, 3, 15)


def test_validate_backfill_completeness_flags_duplicates_and_missing() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP", "AEP"],
            "ds": pd.to_datetime(
                [
                    "2024-01-01 00:00:00-05:00",
                    "2024-01-01 00:00:00-05:00",
                    "2024-01-01 02:00:00-05:00",
                ]
            ),
            "y": [10.0, 11.0, 12.0],
            "market": ["DA", "DA", "DA"],
            "location_type": ["ZONE", "ZONE", "ZONE"],
            "source": ["pjm", "pjm", "pjm"],
            "pulled_at": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"], utc=True),
        }
    )
    out = validate_backfill_completeness(df, zone="AEP")
    assert out["duplicate_timestamp_count"] == 1
    assert out["missing_hour_count"] >= 1


def test_validate_backfill_requires_required_columns() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_backfill_completeness(pd.DataFrame({"x": [1]}), zone="AEP")
