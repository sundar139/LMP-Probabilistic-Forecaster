"""Tests for synthetic panel generator."""

from __future__ import annotations

import pandas as pd
from pandas import DatetimeTZDtype

from lmp_forecaster.data.synthetic_panel import SyntheticPanelConfig, make_synthetic_panel

REQUIRED_COLUMNS = {
    "unique_id",
    "ds",
    "y",
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_2m",
    "load_forecast",
    "lmp_lag_1",
    "lmp_lag_24",
    "lmp_lag_168",
}


def test_synthetic_panel_schema_and_non_nulls() -> None:
    frame = make_synthetic_panel(SyntheticPanelConfig(periods=220))

    assert REQUIRED_COLUMNS.issubset(set(frame.columns))
    assert frame["unique_id"].notna().all()
    assert frame["ds"].notna().all()
    assert frame["y"].notna().all()


def test_lag_1_matches_previous_observation_after_warmup() -> None:
    frame = make_synthetic_panel(SyntheticPanelConfig(periods=220, zones=("AEP", "ATSI")))

    for _, grp in frame.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        shifted = grp["y"].shift(1)
        mask = grp.index >= 1
        left = grp.loc[mask, "lmp_lag_1"].reset_index(drop=True)
        right = shifted.loc[mask].reset_index(drop=True)
        pd.testing.assert_series_equal(left, right, check_names=False)


def test_dst_sensitive_range_timezone_awareness() -> None:
    frame = make_synthetic_panel(
        SyntheticPanelConfig(
            start="2024-03-09 00:00:00",
            periods=72,
            timezone="America/New_York",
        )
    )

    assert isinstance(frame["ds"].dtype, DatetimeTZDtype)
    assert str(frame["ds"].dt.tz) == "America/New_York"
