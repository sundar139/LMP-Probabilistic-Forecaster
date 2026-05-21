"""Tests for lag and rolling LMP features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lmp_forecaster.features.lags import (
    add_lag_features,
    add_rolling_features,
    validate_no_lag_leakage,
)


def _base_panel(rows: int = 220) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": np.arange(rows, dtype=float),
        }
    )


def test_lag_features_shift_correctly() -> None:
    panel = _base_panel()
    out = add_lag_features(panel)

    assert out.loc[10, "lmp_lag_1"] == out.loc[9, "y"]
    assert out.loc[10, "lmp_lag_2"] == out.loc[8, "y"]
    assert out.loc[48, "lmp_lag_48"] == out.loc[0, "y"]

    validate_no_lag_leakage(out)


def test_rolling_features_use_shifted_history_only() -> None:
    panel = _base_panel()
    out = add_lag_features(panel)
    out = add_rolling_features(out)

    expected_mean_24 = panel.loc[:23, "y"].mean()
    assert out.loc[24, "lmp_rolling_mean_24"] == pytest.approx(expected_mean_24)


def test_validate_no_lag_leakage_raises_on_bad_lag() -> None:
    panel = _base_panel(30)
    out = add_lag_features(panel)
    out.loc[10, "lmp_lag_1"] = out.loc[10, "y"]

    with pytest.raises(ValueError):
        validate_no_lag_leakage(out)
