"""Tests for probabilistic metrics."""

from __future__ import annotations

import pandas as pd

from lmp_forecaster.eval.metrics import evaluate_probabilistic_forecast


def test_metrics_expected_values() -> None:
    frame = pd.DataFrame(
        {
            "y": [10.0, 12.0],
            "p10": [9.0, 11.0],
            "p50": [10.0, 13.0],
            "p90": [11.0, 14.0],
        }
    )

    out = evaluate_probabilistic_forecast(
        frame,
        model="TFT",
        zone="AEP",
        data_source_label="synthetic",
    )

    assert out["row_count"] == 2
    assert out["mae"] == 0.5
    assert out["coverage_80"] == 1.0
    assert out["interval_width_mean"] == 2.5
