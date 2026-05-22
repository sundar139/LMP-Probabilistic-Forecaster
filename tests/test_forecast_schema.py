"""Tests for forecast schema normalization and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from lmp_forecaster.models.forecast_schema import (
    normalize_neuralforecast_output,
    validate_quantile_forecast,
)


def test_normalize_forecast_columns_to_p10_p50_p90() -> None:
    fcst = pd.DataFrame(
        {
            "unique_id": ["AEP", "AEP"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="h", tz="America/New_York"),
            "TFT-lo-80": [9.0, 10.0],
            "TFT-median": [10.0, 11.0],
            "TFT-hi-80": [11.0, 12.0],
        }
    )

    norm = normalize_neuralforecast_output(fcst, model="TFT")
    assert {"p10", "p50", "p90", "model", "generated_at"}.issubset(norm.columns)


def test_quantile_validation_catches_bad_ordering() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["AEP"],
            "ds": [pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York")],
            "p10": [11.0],
            "p50": [10.0],
            "p90": [12.0],
        }
    )

    with pytest.raises(ValueError, match="Invalid quantile ordering"):
        validate_quantile_forecast(frame)
