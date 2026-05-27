"""Tests for rolling-origin backtest execution utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from lmp_forecaster.eval.backtest import BacktestConfig, make_rolling_origin_folds
from lmp_forecaster.eval.backtest_runner import (
    BacktestRunConfig,
    aggregate_backtest_metrics,
    planned_output_paths,
    run_single_fold_backtest,
)
from lmp_forecaster.models.baselines import BaselineTrainingConfig, detect_accelerator


def _panel(rows: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-01 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": [float(i % 100) for i in range(rows)],
            "source_label": ["real"] * rows,
        }
    )


def test_run_single_fold_uses_only_pre_origin_train_rows() -> None:
    panel = _panel()
    cfg = BacktestConfig(zone="AEP", folds=2, horizon_hours=24, min_train_hours=168)
    folds = make_rolling_origin_folds(panel, cfg)
    fold = folds[0]

    observed: dict[str, Any] = {}

    def fake_runner(
        model: str,
        train_df: pd.DataFrame,
        history_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_cfg: BaselineTrainingConfig,
        accelerator: Any,
        data_source_label: str,
    ) -> pd.DataFrame:
        observed["model"] = model
        observed["train_max"] = train_df["ds"].max()
        observed["test_min"] = test_df["ds"].min()
        observed["history_rows"] = len(history_df)
        observed["source_label"] = data_source_label

        out = test_df.copy()
        out["p10"] = out["y"] - 1.0
        out["p50"] = out["y"]
        out["p90"] = out["y"] + 1.0
        out["model"] = model
        out["generated_at"] = pd.Timestamp("2026-05-22T00:00:00Z")
        out["data_source_label"] = data_source_label
        out["zone"] = "AEP"
        return out[
            [
                "unique_id",
                "ds",
                "y",
                "p10",
                "p50",
                "p90",
                "model",
                "generated_at",
                "data_source_label",
                "zone",
            ]
        ]

    result = run_single_fold_backtest(
        panel,
        fold,
        model="TFT",
        zone="AEP",
        data_source_label="real",
        train_cfg=BaselineTrainingConfig(),
        accelerator=detect_accelerator("cpu"),
        model_runner=fake_runner,
    )

    assert observed["train_max"] < fold.origin
    assert observed["train_max"] < observed["test_min"]
    assert observed["history_rows"] > 0
    assert observed["source_label"] == "real"
    assert result.metrics["model"] == "TFT"


def test_run_single_fold_appends_fold_id_to_forecasts() -> None:
    panel = _panel()
    cfg = BacktestConfig(zone="AEP", folds=2, horizon_hours=24, min_train_hours=168)
    fold = make_rolling_origin_folds(panel, cfg)[0]

    def fake_runner(
        model: str,
        train_df: pd.DataFrame,
        history_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_cfg: BaselineTrainingConfig,
        accelerator: Any,
        data_source_label: str,
    ) -> pd.DataFrame:
        out = test_df.copy()
        out["p10"] = out["y"] - 2.0
        out["p50"] = out["y"]
        out["p90"] = out["y"] + 2.0
        out["model"] = model
        out["generated_at"] = pd.Timestamp("2026-05-22T00:00:00Z")
        out["data_source_label"] = data_source_label
        out["zone"] = "AEP"
        return out

    fold_result = run_single_fold_backtest(
        panel,
        fold,
        model="DeepAR",
        zone="AEP",
        data_source_label="real",
        train_cfg=BaselineTrainingConfig(),
        accelerator=detect_accelerator("cpu"),
        model_runner=fake_runner,
    )

    assert "fold_id" in fold_result.forecast.columns
    assert fold_result.forecast["fold_id"].nunique() == 1
    assert int(fold_result.forecast["fold_id"].iloc[0]) == fold.fold_id


def test_aggregate_backtest_metrics_includes_required_fields() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "model": "TFT",
                "fold_id": 1,
                "zone": "AEP",
                "row_count": 24,
                "test_start": "2024-01-10 00:00:00-05:00",
                "test_end": "2024-01-10 23:00:00-05:00",
                "MAE": 10.0,
                "RMSE": 12.0,
                "pinball_p10": 2.0,
                "pinball_p50": 4.0,
                "pinball_p90": 6.0,
                "mean_pinball_loss": 4.0,
                "coverage_80": 0.5,
                "interval_width_mean": 8.0,
                "data_source_label": "real",
            },
            {
                "model": "TFT",
                "fold_id": 2,
                "zone": "AEP",
                "row_count": 24,
                "test_start": "2024-01-11 00:00:00-05:00",
                "test_end": "2024-01-11 23:00:00-05:00",
                "MAE": 12.0,
                "RMSE": 14.0,
                "pinball_p10": 2.2,
                "pinball_p50": 4.2,
                "pinball_p90": 6.2,
                "mean_pinball_loss": 4.2,
                "coverage_80": 0.4,
                "interval_width_mean": 9.0,
                "data_source_label": "real",
            },
        ]
    )

    aggregate = aggregate_backtest_metrics(fold_metrics)
    required = {
        "model",
        "zone",
        "folds_completed",
        "total_test_rows",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "mean_pinball_loss_mean",
        "mean_pinball_loss_std",
        "coverage_80_mean",
        "coverage_80_std",
        "interval_width_mean",
        "interval_width_std",
        "best_fold_mae",
        "worst_fold_mae",
        "data_source_label",
    }
    assert required.issubset(set(aggregate.columns))


def test_skip_flags_and_at_least_one_model_validation() -> None:
    cfg_tft_only = BacktestRunConfig(skip_deepar=True)
    cfg_tft_only.validate()
    assert cfg_tft_only.enabled_models == ["TFT"]

    cfg_deepar_only = BacktestRunConfig(skip_tft=True)
    cfg_deepar_only.validate()
    assert cfg_deepar_only.enabled_models == ["DeepAR"]

    cfg_none = BacktestRunConfig(skip_tft=True, skip_deepar=True)
    with pytest.raises(ValueError, match="At least one model must be enabled"):
        cfg_none.validate()


def test_generated_report_paths_are_under_ignored_locations() -> None:
    paths = planned_output_paths(BacktestRunConfig())
    rendered = {k: str(v).replace("\\", "/") for k, v in paths.items()}

    assert "/data/cache/backtests/" in rendered["forecasts"]
    assert "/data/cache/backtests/" in rendered["fold_metrics"]
    assert "/data/cache/backtests/" in rendered["aggregate_metrics"]
    assert "/data/cache/reports/" in rendered["summary_json"]
    assert "/data/cache/reports/" in rendered["summary_markdown"]
    assert "/artifacts/backtests/" in rendered["manifest"]
