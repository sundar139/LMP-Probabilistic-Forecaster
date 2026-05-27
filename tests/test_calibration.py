"""Unit tests for calibration diagnostics utilities."""

from __future__ import annotations

import pandas as pd

from lmp_forecaster.eval.calibration import (
    CalibrationDiagnosticConfig,
    classify_calibration_status,
    compute_interval_coverage_by_fold,
    compute_interval_coverage_by_horizon,
    compute_median_bias,
    compute_quantile_crossing_rate,
)


def _fake_forecasts() -> pd.DataFrame:
    rows = [
        # TFT, fold 1 (both covered)
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York"),
            "y": 10.0,
            "p10": 9.0,
            "p50": 10.0,
            "p90": 11.0,
            "model": "TFT",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 1,
        },
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-01 01:00:00", tz="America/New_York"),
            "y": 12.0,
            "p10": 11.0,
            "p50": 12.0,
            "p90": 13.0,
            "model": "TFT",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 1,
        },
        # TFT, fold 2 (one covered, one missed)
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-02 00:00:00", tz="America/New_York"),
            "y": 14.0,
            "p10": 13.0,
            "p50": 14.0,
            "p90": 15.0,
            "model": "TFT",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 2,
        },
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-02 01:00:00", tz="America/New_York"),
            "y": 16.0,
            "p10": 17.0,
            "p50": 18.0,
            "p90": 19.0,
            "model": "TFT",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 2,
        },
        # DeepAR, fold 1 (crossing + not covered)
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York"),
            "y": 30.0,
            "p10": 33.0,
            "p50": 32.0,
            "p90": 34.0,
            "model": "DeepAR",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 1,
        },
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-01 01:00:00", tz="America/New_York"),
            "y": 31.0,
            "p10": 31.0,
            "p50": 35.0,
            "p90": 34.0,
            "model": "DeepAR",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 1,
        },
        # DeepAR, fold 2 (one covered, one missed)
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-02 00:00:00", tz="America/New_York"),
            "y": 32.0,
            "p10": 32.0,
            "p50": 33.0,
            "p90": 33.0,
            "model": "DeepAR",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 2,
        },
        {
            "unique_id": "AEP",
            "ds": pd.Timestamp("2024-01-02 01:00:00", tz="America/New_York"),
            "y": 33.0,
            "p10": 34.0,
            "p50": 35.0,
            "p90": 36.0,
            "model": "DeepAR",
            "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
            "data_source_label": "real",
            "zone": "AEP",
            "fold_id": 2,
        },
    ]
    return pd.DataFrame(rows)


def test_coverage_by_fold_computes_expected_values() -> None:
    frame = _fake_forecasts()
    out = compute_interval_coverage_by_fold(frame)

    tft_fold_1 = out[(out["model"] == "TFT") & (out["fold_id"] == 1)].iloc[0]
    tft_fold_2 = out[(out["model"] == "TFT") & (out["fold_id"] == 2)].iloc[0]
    assert float(tft_fold_1["coverage_80"]) == 1.0
    assert float(tft_fold_2["coverage_80"]) == 0.5


def test_coverage_by_horizon_computes_expected_values() -> None:
    frame = _fake_forecasts()
    out = compute_interval_coverage_by_horizon(frame)

    tft_h1 = out[(out["model"] == "TFT") & (out["horizon_hour"] == 1)].iloc[0]
    tft_h2 = out[(out["model"] == "TFT") & (out["horizon_hour"] == 2)].iloc[0]
    deepar_h2 = out[(out["model"] == "DeepAR") & (out["horizon_hour"] == 2)].iloc[0]

    assert float(tft_h1["coverage_80"]) == 1.0
    assert float(tft_h2["coverage_80"]) == 0.5
    assert float(deepar_h2["coverage_80"]) == 0.5


def test_quantile_crossing_rate_detects_crossing() -> None:
    frame = _fake_forecasts()
    out = compute_quantile_crossing_rate(frame)

    deepar = out[out["model"] == "DeepAR"].iloc[0]
    tft = out[out["model"] == "TFT"].iloc[0]

    assert float(deepar["crossing_rate"]) > 0.0
    assert float(tft["crossing_rate"]) == 0.0


def test_median_bias_computes_expected_value() -> None:
    frame = _fake_forecasts()
    out = compute_median_bias(frame)

    tft = out[out["model"] == "TFT"].iloc[0]
    deepar = out[out["model"] == "DeepAR"].iloc[0]

    assert float(tft["median_bias_mean"]) == 0.5
    assert float(deepar["median_bias_mean"]) == 2.25


def test_classification_marks_under_coverage_for_tft_like_metrics() -> None:
    cfg = CalibrationDiagnosticConfig()
    summary = pd.DataFrame(
        [
            {"model": "TFT", "coverage_80": 0.5833, "interval_width_mean": 13.4},
            {"model": "DeepAR", "coverage_80": 0.2, "interval_width_mean": 5.0},
        ]
    )
    out = classify_calibration_status(summary, cfg)
    tft = out[out["model"] == "TFT"].iloc[0]
    assert str(tft["calibration_status"]) == "under-coverage"


def test_classification_marks_zero_coverage_as_collapse_warning() -> None:
    cfg = CalibrationDiagnosticConfig(collapse_coverage_threshold=0.05)
    summary = pd.DataFrame(
        [
            {"model": "TFT", "coverage_80": 0.60, "interval_width_mean": 12.0},
            {"model": "DeepAR", "coverage_80": 0.0, "interval_width_mean": 3.0},
        ]
    )
    out = classify_calibration_status(summary, cfg)
    deepar = out[out["model"] == "DeepAR"].iloc[0]
    assert bool(deepar["interval_collapse_warning"]) is True
    assert "collapse" in str(deepar["classification_note"])
