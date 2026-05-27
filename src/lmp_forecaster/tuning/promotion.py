"""Promotion decision utilities for focused tuning candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineMetrics:
    model: str
    MAE_mean: float
    RMSE_mean: float
    mean_pinball_loss_mean: float
    coverage_80_mean: float
    interval_width_mean: float
    data_source_label: str = "real"


@dataclass(frozen=True)
class CandidateMetrics:
    model: str
    trial_id: str
    MAE_mean: float
    RMSE_mean: float
    mean_pinball_loss_mean: float
    coverage_80_mean: float
    interval_width_mean: float
    median_bias_mean: float | None = None
    quantile_crossing_rate: float | None = None
    interval_collapse_warning: bool = False


@dataclass(frozen=True)
class PromotionGate:
    coverage_min: float = 0.70
    coverage_max: float = 0.90
    mae_regression_limit: float = 0.15
    require_no_quantile_crossing: bool = True
    require_no_interval_collapse: bool = True

    def validate(self) -> None:
        if not (0.0 < self.coverage_min < 1.0):
            raise ValueError("coverage_min must be in (0, 1)")
        if not (0.0 < self.coverage_max < 1.0):
            raise ValueError("coverage_max must be in (0, 1)")
        if self.coverage_min >= self.coverage_max:
            raise ValueError("coverage_min must be < coverage_max")
        if self.mae_regression_limit < 0:
            raise ValueError("mae_regression_limit must be >= 0")


@dataclass(frozen=True)
class PromotionDecision:
    model: str
    trial_id: str
    promoted: bool
    promotion_status: str
    rejection_reason: str | None
    objective_score: float
    mae_regression_fraction: float
    coverage_gate_passed: bool
    mae_regression_gate_passed: bool
    interval_collapse_gate_passed: bool
    quantile_crossing_gate_passed: bool
    pinball_improved: bool
    rmse_not_worse: bool
    mae_not_worse: bool


def evaluate_candidate_against_baseline(
    candidate: CandidateMetrics,
    baseline: BaselineMetrics,
    gate: PromotionGate,
) -> PromotionDecision:
    gate.validate()

    reasons: list[str] = []

    coverage_gate_passed = gate.coverage_min <= candidate.coverage_80_mean <= gate.coverage_max
    if candidate.coverage_80_mean < gate.coverage_min:
        reasons.append(
            f"coverage below gate ({candidate.coverage_80_mean:.4f} < {gate.coverage_min:.2f})"
        )
    if candidate.coverage_80_mean > gate.coverage_max:
        reasons.append(
            f"coverage above gate ({candidate.coverage_80_mean:.4f} > {gate.coverage_max:.2f})"
        )

    crossing = candidate.quantile_crossing_rate or 0.0
    quantile_crossing_gate_passed = (not gate.require_no_quantile_crossing) or crossing <= 0
    if gate.require_no_quantile_crossing and crossing > 0:
        reasons.append(f"quantile crossing detected (rate={crossing:.4f})")

    interval_collapse_gate_passed = (
        (not gate.require_no_interval_collapse) or (not candidate.interval_collapse_warning)
    )
    if gate.require_no_interval_collapse and candidate.interval_collapse_warning:
        reasons.append("interval collapse warning present")

    mae_regression_fraction = 0.0
    if baseline.MAE_mean > 0:
        mae_regression_fraction = (candidate.MAE_mean - baseline.MAE_mean) / baseline.MAE_mean
    mae_regression_gate_passed = mae_regression_fraction <= gate.mae_regression_limit
    if not mae_regression_gate_passed:
        reasons.append(
            "MAE regression above limit "
            f"({mae_regression_fraction:.4f} > {gate.mae_regression_limit:.4f})"
        )

    pinball_improved = candidate.mean_pinball_loss_mean < baseline.mean_pinball_loss_mean
    rmse_not_worse = candidate.RMSE_mean <= baseline.RMSE_mean
    mae_not_worse = candidate.MAE_mean <= baseline.MAE_mean * (1 + gate.mae_regression_limit)

    if not pinball_improved and not (mae_not_worse and rmse_not_worse):
        reasons.append(
            "objective not improved and error metrics regressed beyond acceptable fallback"
        )

    objective_score = float(
        candidate.mean_pinball_loss_mean
        + abs(candidate.coverage_80_mean - 0.80)
        + max(0.0, mae_regression_fraction)
    )

    promoted = len(reasons) == 0
    return PromotionDecision(
        model=candidate.model,
        trial_id=candidate.trial_id,
        promoted=promoted,
        promotion_status="promoted" if promoted else "rejected",
        rejection_reason=None if promoted else "; ".join(reasons),
        objective_score=objective_score,
        mae_regression_fraction=float(mae_regression_fraction),
        coverage_gate_passed=coverage_gate_passed,
        mae_regression_gate_passed=mae_regression_gate_passed,
        interval_collapse_gate_passed=interval_collapse_gate_passed,
        quantile_crossing_gate_passed=quantile_crossing_gate_passed,
        pinball_improved=pinball_improved,
        rmse_not_worse=rmse_not_worse,
        mae_not_worse=mae_not_worse,
    )


def summarize_promotion_decisions(decisions: list[PromotionDecision]) -> dict[str, object]:
    if not decisions:
        return {
            "best_promoted_trial": None,
            "best_rejected_trial": None,
            "promoted_count": 0,
            "rejected_count": 0,
            "overall_status": "no_promotion",
            "summary_reason": "No trials were evaluated.",
        }

    promoted = [d for d in decisions if d.promoted]
    rejected = [d for d in decisions if not d.promoted]

    best_promoted = min(promoted, key=lambda d: d.objective_score) if promoted else None
    best_rejected = min(rejected, key=lambda d: d.objective_score) if rejected else None

    if best_promoted is not None:
        overall = "promoted"
        reason = (
            f"Promoted {best_promoted.model} trial {best_promoted.trial_id} "
            f"with objective_score={best_promoted.objective_score:.6f}."
        )
    elif best_rejected is not None:
        overall = "no_promotion"
        reason = (
            "No candidate passed promotion gates. "
            f"Best rejected trial: {best_rejected.model} {best_rejected.trial_id} "
            f"({best_rejected.rejection_reason})."
        )
    else:
        overall = "no_promotion"
        reason = "No candidate passed promotion gates."

    return {
        "best_promoted_trial": None
        if best_promoted is None
        else {
            "model": best_promoted.model,
            "trial_id": best_promoted.trial_id,
            "objective_score": best_promoted.objective_score,
            "promotion_status": best_promoted.promotion_status,
        },
        "best_rejected_trial": None
        if best_rejected is None
        else {
            "model": best_rejected.model,
            "trial_id": best_rejected.trial_id,
            "objective_score": best_rejected.objective_score,
            "promotion_status": best_rejected.promotion_status,
            "rejection_reason": best_rejected.rejection_reason,
        },
        "promoted_count": len(promoted),
        "rejected_count": len(rejected),
        "overall_status": overall,
        "summary_reason": reason,
    }
