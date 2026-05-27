"""Unit tests for focused tuning runner orchestration and resource safety."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lmp_forecaster.tuning.promotion import BaselineMetrics, PromotionDecision
from lmp_forecaster.tuning.tuning_runner import (
    TrialResult,
    TuningRunConfig,
    planned_tuning_output_paths,
    run_focused_tuning,
)


def _search_design_payload() -> dict[str, object]:
    return {
        "zone": "AEP",
        "spaces": {
            "TFT": {
                "parameters": [
                    {"name": "dropout", "type": "float", "min": 0.05, "max": 0.10},
                    {"name": "max_steps", "type": "int", "min": 2, "max": 3},
                ]
            },
            "DeepAR": {
                "parameters": [
                    {
                        "name": "distribution_loss",
                        "type": "categorical",
                        "values": ["StudentT", "Normal"],
                    },
                    {"name": "max_steps", "type": "int", "min": 2, "max": 3},
                ]
            },
        },
    }


def _baseline_map() -> dict[str, BaselineMetrics]:
    return {
        "TFT": BaselineMetrics(
            model="TFT",
            MAE_mean=5.5506,
            RMSE_mean=6.2262,
            mean_pinball_loss_mean=1.8597,
            coverage_80_mean=0.5833,
            interval_width_mean=13.4750,
            data_source_label="real",
        ),
        "DeepAR": BaselineMetrics(
            model="DeepAR",
            MAE_mean=21.9064,
            RMSE_mean=22.4298,
            mean_pinball_loss_mean=10.2479,
            coverage_80_mean=0.0,
            interval_width_mean=5.0474,
            data_source_label="real",
        ),
    }


def _successful_trial_result(cfg: TuningRunConfig, trial_id: str, model: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        model=model,
        zone=cfg.zone,
        folds_completed=cfg.folds,
        total_test_rows=cfg.folds * cfg.horizon_hours,
        config={"max_steps": 3},
        MAE_mean=5.3,
        RMSE_mean=6.0,
        mean_pinball_loss_mean=1.6,
        coverage_80_mean=0.79,
        interval_width_mean=13.0,
        median_bias_mean=0.0,
        quantile_crossing_rate=0.0,
        interval_collapse_warning=False,
        objective_score=1.61,
        promotion_status="smoke_candidate_requires_full_validation",
        rejection_reason="single-fold smoke tuning result requires multi-fold validation",
        runtime_seconds=0.2,
        data_source_label="real",
        coverage_gate_passed=True,
        mae_regression_gate_passed=True,
        interval_collapse_gate_passed=True,
    )


def _successful_decision(model: str, trial_id: str) -> PromotionDecision:
    return PromotionDecision(
        model=model,
        trial_id=trial_id,
        promoted=True,
        promotion_status="promoted",
        rejection_reason=None,
        objective_score=1.61,
        mae_regression_fraction=0.0,
        coverage_gate_passed=True,
        mae_regression_gate_passed=True,
        interval_collapse_gate_passed=True,
        quantile_crossing_gate_passed=True,
        pinball_improved=True,
        rmse_not_worse=True,
        mae_not_worse=True,
    )


def test_cleanup_after_trial_hook_called(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cleanup_calls: list[str] = []

    def fake_run_single_tuning_trial(trial, run_cfg, baseline_map):  # type: ignore[no-untyped-def]
        return _successful_trial_result(run_cfg, trial.trial_id, trial.model), _successful_decision(
            trial.model,
            trial.trial_id,
        )

    def fake_cleanup_after_trial(run_cfg, trial_id):  # type: ignore[no-untyped-def]
        cleanup_calls.append(trial_id)

    monkeypatch.setattr(
        "lmp_forecaster.tuning.tuning_runner.run_single_tuning_trial",
        fake_run_single_tuning_trial,
    )
    monkeypatch.setattr(
        "lmp_forecaster.tuning.tuning_runner.cleanup_after_trial",
        fake_cleanup_after_trial,
    )

    cfg = TuningRunConfig(
        zone="AEP",
        models=("TFT",),
        max_trials=2,
        folds=1,
        max_steps_cap=3,
        resource_profile="local_safe",
    )
    summary = run_focused_tuning(
        cfg,
        search_design=_search_design_payload(),
        baseline_map=_baseline_map(),
    )

    assert len(summary.trial_results) == 2
    assert len(cleanup_calls) == 2


def test_cuda_oom_is_captured_as_failed_resource_limited(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_run_single_tuning_trial(trial, run_cfg, baseline_map):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA out of memory while allocating tensor")

    monkeypatch.setattr(
        "lmp_forecaster.tuning.tuning_runner.run_single_tuning_trial",
        fake_run_single_tuning_trial,
    )

    cfg = TuningRunConfig(
        zone="AEP",
        models=("TFT",),
        max_trials=2,
        folds=1,
        max_steps_cap=3,
        resource_profile="local_safe",
    )
    summary = run_focused_tuning(
        cfg,
        search_design=_search_design_payload(),
        baseline_map=_baseline_map(),
    )

    assert summary.trial_results
    assert all(row.promotion_status == "failed_resource_limited" for row in summary.trial_results)
    assert summary.promotion_summary["overall_status"] == "failed_resource_limited"


def test_generated_report_paths_are_under_ignored_locations() -> None:
    paths = planned_tuning_output_paths(TuningRunConfig())
    rendered = {k: str(v).replace("\\", "/") for k, v in paths.items()}
    assert "/data/cache/tuning/" in rendered["trials"]
    assert "/data/cache/tuning/" in rendered["ranked"]
    assert "/data/cache/reports/" in rendered["summary_json"]
    assert "/data/cache/reports/" in rendered["summary_markdown"]
    assert "/artifacts/tuning/" in rendered["manifest"]


def test_no_optuna_db_is_created_by_default_local_safe(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    def fake_run_single_tuning_trial(trial, run_cfg, baseline_map):  # type: ignore[no-untyped-def]
        return _successful_trial_result(run_cfg, trial.trial_id, trial.model), _successful_decision(
            trial.model,
            trial.trial_id,
        )

    monkeypatch.setattr(
        "lmp_forecaster.tuning.tuning_runner.run_single_tuning_trial",
        fake_run_single_tuning_trial,
    )

    optuna_db = tmp_path / "data" / "cache" / "tuning" / "optuna_smoke.sqlite3"
    cfg = TuningRunConfig(
        zone="AEP",
        models=("TFT",),
        max_trials=2,
        folds=1,
        max_steps_cap=3,
        resource_profile="local_safe",
        use_optuna=False,
        optuna_storage_path=optuna_db,
    )
    _ = run_focused_tuning(
        cfg,
        search_design=_search_design_payload(),
        baseline_map=_baseline_map(),
    )

    assert not optuna_db.exists()


def test_tracking_disabled_status_stable() -> None:
    from lmp_forecaster.tuning.tuning_runner import log_tuning_tracking

    summary = type(
        "Summary",
        (),
        {
            "config": TuningRunConfig(enable_tracking=False),
            "promotion_summary": {},
            "ranked_candidates": pd.DataFrame(),
        },
    )()
    status = log_tuning_tracking(summary, {})
    assert status["enabled"] is False
