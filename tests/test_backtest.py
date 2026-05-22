"""Tests for rolling-origin backtest fold planning scaffold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lmp_forecaster.eval.backtest import (
    BacktestConfig,
    BacktestFold,
    make_rolling_origin_folds,
    summarize_backtest_plan,
    validate_backtest_folds,
    write_backtest_plan,
)


def _panel(rows: int = 8616) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-08 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": [float(i % 200) for i in range(rows)],
        }
    )


def test_make_rolling_origin_folds_count_matches_request() -> None:
    panel = _panel()
    cfg = BacktestConfig(zone="AEP", folds=3, horizon_hours=24, min_train_hours=2160)
    folds = make_rolling_origin_folds(panel, cfg)
    assert len(folds) == 3


def test_validate_backtest_folds_catches_leakage() -> None:
    ts = pd.Timestamp("2024-12-01 00:00:00", tz="America/New_York")
    bad = [
        BacktestFold(
            fold_id=1,
            train_start=ts - pd.Timedelta(hours=100),
            train_end=ts - pd.Timedelta(hours=1),
            origin=ts,
            test_start=ts,
            test_end=ts + pd.Timedelta(hours=23),
            train_rows=100,
            test_rows=24,
            horizon_hours=24,
            leakage_check_passed=False,
            overlap_check_passed=True,
        )
    ]
    with pytest.raises(ValueError, match="leakage validation failed"):
        validate_backtest_folds(bad)


def test_validate_backtest_folds_catches_overlap() -> None:
    ts = pd.Timestamp("2024-12-01 00:00:00", tz="America/New_York")
    bad = [
        BacktestFold(
            fold_id=1,
            train_start=ts - pd.Timedelta(hours=300),
            train_end=ts - pd.Timedelta(hours=1),
            origin=ts,
            test_start=ts,
            test_end=ts + pd.Timedelta(hours=23),
            train_rows=300,
            test_rows=24,
            horizon_hours=24,
            leakage_check_passed=True,
            overlap_check_passed=True,
        ),
        BacktestFold(
            fold_id=2,
            train_start=ts - pd.Timedelta(hours=276),
            train_end=ts + pd.Timedelta(hours=1),
            origin=ts + pd.Timedelta(hours=12),
            test_start=ts + pd.Timedelta(hours=12),
            test_end=ts + pd.Timedelta(hours=35),
            train_rows=300,
            test_rows=24,
            horizon_hours=24,
            leakage_check_passed=True,
            overlap_check_passed=False,
        ),
    ]
    with pytest.raises(ValueError, match="overlap validation failed"):
        validate_backtest_folds(bad)


def test_backtest_summary_includes_required_fields(tmp_path: Path) -> None:
    panel = _panel()
    cfg = BacktestConfig(zone="AEP", folds=3, horizon_hours=24, min_train_hours=2160)
    folds = make_rolling_origin_folds(panel, cfg)
    validate_backtest_folds(folds)

    summary = summarize_backtest_plan(
        panel,
        cfg,
        folds,
        tmp_path / "backtest_plan_AEP_<timestamp>.json",
    )
    assert summary["zone"] == "AEP"
    assert summary["horizon_hours"] == 24
    assert len(summary["folds"]) == 3

    json_path, md_path = write_backtest_plan(summary, output_dir=tmp_path)
    assert json_path.exists()
    assert md_path.exists()
