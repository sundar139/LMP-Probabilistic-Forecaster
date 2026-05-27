"""Tests for backtest metrics reporting and interpretation behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lmp_forecaster.eval.backtest import BacktestFold
from lmp_forecaster.eval.backtest_runner import (
    BacktestRunConfig,
    RollingBacktestResult,
    write_backtest_results,
)


def _folds() -> list[BacktestFold]:
    return [
        BacktestFold(
            fold_id=1,
            train_start=pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York"),
            train_end=pd.Timestamp("2024-01-08 23:00:00", tz="America/New_York"),
            origin=pd.Timestamp("2024-01-09 00:00:00", tz="America/New_York"),
            test_start=pd.Timestamp("2024-01-09 00:00:00", tz="America/New_York"),
            test_end=pd.Timestamp("2024-01-09 23:00:00", tz="America/New_York"),
            train_rows=192,
            test_rows=24,
            horizon_hours=24,
            leakage_check_passed=True,
            overlap_check_passed=True,
        ),
        BacktestFold(
            fold_id=2,
            train_start=pd.Timestamp("2024-01-01 00:00:00", tz="America/New_York"),
            train_end=pd.Timestamp("2024-01-09 23:00:00", tz="America/New_York"),
            origin=pd.Timestamp("2024-01-10 00:00:00", tz="America/New_York"),
            test_start=pd.Timestamp("2024-01-10 00:00:00", tz="America/New_York"),
            test_end=pd.Timestamp("2024-01-10 23:00:00", tz="America/New_York"),
            train_rows=216,
            test_rows=24,
            horizon_hours=24,
            leakage_check_passed=True,
            overlap_check_passed=True,
        ),
    ]


def test_low_coverage_is_preserved_and_reported(tmp_path: Path) -> None:
    cfg = BacktestRunConfig(output_root=tmp_path / "data/cache/backtests")

    forecasts = pd.DataFrame(
        {
            "unique_id": ["AEP"] * 2,
            "ds": [
                pd.Timestamp("2024-01-09 00:00:00", tz="America/New_York"),
                pd.Timestamp("2024-01-10 00:00:00", tz="America/New_York"),
            ],
            "y": [100.0, 101.0],
            "p10": [99.0, 100.0],
            "p50": [100.0, 101.0],
            "p90": [101.0, 102.0],
            "model": ["DeepAR", "DeepAR"],
            "generated_at": [pd.Timestamp("2026-05-22T00:00:00Z")] * 2,
            "data_source_label": ["real"] * 2,
            "zone": ["AEP"] * 2,
            "fold_id": [1, 2],
        }
    )
    fold_metrics = pd.DataFrame(
        [
            {
                "model": "DeepAR",
                "fold_id": 1,
                "zone": "AEP",
                "row_count": 24,
                "test_start": "2024-01-09 00:00:00-05:00",
                "test_end": "2024-01-09 23:00:00-05:00",
                "MAE": 10.0,
                "RMSE": 12.0,
                "pinball_p10": 2.0,
                "pinball_p50": 3.0,
                "pinball_p90": 4.0,
                "mean_pinball_loss": 3.0,
                "coverage_80": 0.0,
                "interval_width_mean": 1.0,
                "data_source_label": "real",
            },
            {
                "model": "DeepAR",
                "fold_id": 2,
                "zone": "AEP",
                "row_count": 24,
                "test_start": "2024-01-10 00:00:00-05:00",
                "test_end": "2024-01-10 23:00:00-05:00",
                "MAE": 11.0,
                "RMSE": 13.0,
                "pinball_p10": 2.1,
                "pinball_p50": 3.1,
                "pinball_p90": 4.1,
                "mean_pinball_loss": 3.1,
                "coverage_80": 0.0,
                "interval_width_mean": 1.0,
                "data_source_label": "real",
            },
        ]
    )
    aggregate = pd.DataFrame(
        [
            {
                "model": "DeepAR",
                "zone": "AEP",
                "folds_completed": 2,
                "total_test_rows": 48,
                "MAE_mean": 10.5,
                "MAE_std": 0.5,
                "RMSE_mean": 12.5,
                "RMSE_std": 0.5,
                "mean_pinball_loss_mean": 3.05,
                "mean_pinball_loss_std": 0.05,
                "coverage_80_mean": 0.0,
                "coverage_80_std": 0.0,
                "interval_width_mean": 1.0,
                "interval_width_std": 0.0,
                "best_fold_mae": 10.0,
                "worst_fold_mae": 11.0,
                "data_source_label": "real",
            }
        ]
    )

    result = RollingBacktestResult(
        config=cfg,
        folds=_folds(),
        forecasts=forecasts,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        accelerator="cpu",
        device_name="CPU",
        data_source_label="real",
    )

    paths = write_backtest_results(result, output_root=tmp_path / "data/cache/backtests")

    summary = Path(paths["summary_json"]).read_text(encoding="utf-8")
    assert "under-coverage" in summary
    assert '"coverage_80_mean": 0.0' in summary
