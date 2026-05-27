"""Unit tests for focused tuning promotion gates and decisions."""

from __future__ import annotations

from lmp_forecaster.tuning.promotion import (
    BaselineMetrics,
    CandidateMetrics,
    PromotionGate,
    evaluate_candidate_against_baseline,
)
from lmp_forecaster.tuning.tuning_runner import TrialResult, decide_promotion


def _baseline(model: str = "TFT") -> BaselineMetrics:
    return BaselineMetrics(
        model=model,
        MAE_mean=10.0,
        RMSE_mean=12.0,
        mean_pinball_loss_mean=3.0,
        coverage_80_mean=0.58,
        interval_width_mean=8.0,
        data_source_label="real",
    )


def _candidate(**overrides: object) -> CandidateMetrics:
    base = CandidateMetrics(
        model="TFT",
        trial_id="tft_001",
        MAE_mean=9.5,
        RMSE_mean=11.8,
        mean_pinball_loss_mean=2.7,
        coverage_80_mean=0.80,
        interval_width_mean=9.2,
        median_bias_mean=0.1,
        quantile_crossing_rate=0.0,
        interval_collapse_warning=False,
    )
    payload = {**base.__dict__, **overrides}
    return CandidateMetrics(**payload)


def test_promotion_rejects_under_coverage_below_gate() -> None:
    decision = evaluate_candidate_against_baseline(
        _candidate(coverage_80_mean=0.65),
        _baseline(),
        PromotionGate(coverage_min=0.70, coverage_max=0.90, mae_regression_limit=0.15),
    )
    assert decision.promoted is False
    assert decision.promotion_status == "rejected"
    assert decision.rejection_reason is not None
    assert "coverage below gate" in decision.rejection_reason
    assert decision.coverage_gate_passed is False


def test_promotion_rejects_over_coverage_above_gate() -> None:
    decision = evaluate_candidate_against_baseline(
        _candidate(coverage_80_mean=0.95),
        _baseline(),
        PromotionGate(coverage_min=0.70, coverage_max=0.90, mae_regression_limit=0.15),
    )
    assert decision.promoted is False
    assert decision.rejection_reason is not None
    assert "coverage above gate" in decision.rejection_reason


def test_promotion_rejects_mae_regression_above_limit() -> None:
    decision = evaluate_candidate_against_baseline(
        _candidate(MAE_mean=12.0, mean_pinball_loss_mean=3.1, RMSE_mean=12.6),
        _baseline(),
        PromotionGate(coverage_min=0.70, coverage_max=0.90, mae_regression_limit=0.15),
    )
    assert decision.promoted is False
    assert decision.rejection_reason is not None
    assert "MAE regression above limit" in decision.rejection_reason
    assert decision.mae_regression_gate_passed is False


def test_promotion_rejects_interval_collapse_warning() -> None:
    decision = evaluate_candidate_against_baseline(
        _candidate(interval_collapse_warning=True),
        _baseline(),
        PromotionGate(coverage_min=0.70, coverage_max=0.90, mae_regression_limit=0.15),
    )
    assert decision.promoted is False
    assert decision.rejection_reason is not None
    assert "interval collapse warning" in decision.rejection_reason
    assert decision.interval_collapse_gate_passed is False


def test_promotion_accepts_candidate_with_improved_pinball_and_valid_coverage() -> None:
    decision = evaluate_candidate_against_baseline(
        _candidate(),
        _baseline(),
        PromotionGate(coverage_min=0.70, coverage_max=0.90, mae_regression_limit=0.15),
    )
    assert decision.promoted is True
    assert decision.promotion_status == "promoted"
    assert decision.rejection_reason is None
    assert decision.coverage_gate_passed is True
    assert decision.mae_regression_gate_passed is True


def test_decide_promotion_marks_smoke_candidate_as_not_promoted() -> None:
    trial = TrialResult(
        trial_id="tft_001",
        model="TFT",
        zone="AEP",
        folds_completed=1,
        total_test_rows=24,
        config={"max_steps": 3},
        MAE_mean=9.0,
        RMSE_mean=10.0,
        mean_pinball_loss_mean=2.5,
        coverage_80_mean=0.78,
        interval_width_mean=10.0,
        median_bias_mean=0.0,
        quantile_crossing_rate=0.0,
        interval_collapse_warning=False,
        objective_score=2.52,
        promotion_status="smoke_candidate_requires_full_validation",
        rejection_reason="single-fold smoke tuning result requires multi-fold validation",
        runtime_seconds=1.0,
        data_source_label="real",
        coverage_gate_passed=True,
        mae_regression_gate_passed=True,
        interval_collapse_gate_passed=True,
    )
    summary = decide_promotion([trial], [])
    assert summary["overall_status"] == "smoke_candidate_requires_full_validation"
    best = summary["best_candidate"]
    assert isinstance(best, dict)
    assert best["promotion_status"] == "smoke_candidate_requires_full_validation"
