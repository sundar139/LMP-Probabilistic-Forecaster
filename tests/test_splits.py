"""Tests for time split utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from lmp_forecaster.data.splits import TimeSplitConfig, split_single_series_panel


def _panel(rows: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": [float(i) for i in range(rows)],
        }
    )


def test_split_no_overlap_and_ordering() -> None:
    train, val, test = split_single_series_panel(
        _panel(300),
        TimeSplitConfig(val_size=72, test_size=72),
    )

    assert train["ds"].max() < val["ds"].min() < test["ds"].min()
    assert set(train["ds"]) & set(val["ds"]) == set()
    assert set(train["ds"]) & set(test["ds"]) == set()
    assert set(val["ds"]) & set(test["ds"]) == set()


def test_insufficient_history_raises() -> None:
    with pytest.raises(ValueError, match="Insufficient history"):
        split_single_series_panel(_panel(100), TimeSplitConfig(val_size=72, test_size=72))
