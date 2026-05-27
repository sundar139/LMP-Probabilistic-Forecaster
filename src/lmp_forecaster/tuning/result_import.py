"""Import and validate externally generated focused-tuning results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.tuning.promotion import (
    BaselineMetrics,
    CandidateMetrics,
    PromotionGate,
    evaluate_candidate_against_baseline,
)

REQUIRED_CANDIDATE_FIELDS = {
    "model",
    "trial_id",
    "MAE_mean",
    "RMSE_mean",
    "mean_pinball_loss_mean",
    "coverage_80_mean",
    "interval_width_mean",
}


@dataclass(frozen=True)
class ImportedCandidate:
    model: str
    trial_id: str
    MAE_mean: float
    RMSE_mean: float
    mean_pinball_loss_mean: float
    coverage_80_mean: float
    interval_width_mean: float
    quantile_crossing_rate: float | None
    interval_collapse_warning: bool
    imported_promotion_status: str | None
    recomputed_promotion_status: str
    recomputed_rejection_reason: str | None
    status_match: bool


@dataclass(frozen=True)
class ImportedTuningResult:
    zone: str
    ranked_candidates_path: Path
    summary_path: Path | None
    baseline_metrics_path: Path
    imported_candidates: list[ImportedCandidate]
    recomputed_overall_status: str
    mismatch_count: int
    rejection_count: int
    accepted_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "ranked_candidates_path": self.ranked_candidates_path.as_posix(),
            "summary_path": None if self.summary_path is None else self.summary_path.as_posix(),
            "baseline_metrics_path": self.baseline_metrics_path.as_posix(),
            "recomputed_overall_status": self.recomputed_overall_status,
            "mismatch_count": self.mismatch_count,
            "rejection_count": self.rejection_count,
            "accepted_count": self.accepted_count,
            "imported_candidates": [asdict(item) for item in self.imported_candidates],
        }


@dataclass(frozen=True)
class ImportedEvaluation:
    candidates: list[ImportedCandidate]
    recomputed_overall_status: str
    mismatch_count: int
    rejection_count: int
    accepted_count: int


def validate_imported_result_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_CANDIDATE_FIELDS.difference(frame.columns))
    if missing:
        raise ValueError(f"Imported ranked candidates missing required fields: {missing}")


def _load_baseline_metrics(path: Path) -> dict[str, BaselineMetrics]:
    baseline = pd.read_csv(path)
    required = {
        "model",
        "MAE_mean",
        "RMSE_mean",
        "mean_pinball_loss_mean",
        "coverage_80_mean",
        "interval_width_mean",
    }
    missing = sorted(required.difference(baseline.columns))
    if missing:
        raise ValueError(f"Baseline metrics file missing columns: {missing}")

    out: dict[str, BaselineMetrics] = {}
    for _, row in baseline.iterrows():
        model = str(row["model"])
        out[model] = BaselineMetrics(
            model=model,
            MAE_mean=float(row["MAE_mean"]),
            RMSE_mean=float(row["RMSE_mean"]),
            mean_pinball_loss_mean=float(row["mean_pinball_loss_mean"]),
            coverage_80_mean=float(row["coverage_80_mean"]),
            interval_width_mean=float(row["interval_width_mean"]),
            data_source_label=str(row.get("data_source_label", "real")),
        )
    return out


def evaluate_imported_candidates(
    ranked_frame: pd.DataFrame,
    baseline_by_model: dict[str, BaselineMetrics],
    gate: PromotionGate,
) -> ImportedEvaluation:
    records: list[ImportedCandidate] = []
    mismatch_count = 0
    accepted_count = 0

    for _, row in ranked_frame.iterrows():
        model = str(row["model"])
        if model not in baseline_by_model:
            raise ValueError(f"Missing baseline metrics for imported model: {model}")

        crossing_raw = row.get("quantile_crossing_rate")
        crossing = (
            float(crossing_raw)
            if crossing_raw is not None and not pd.isna(crossing_raw)
            else None
        )
        collapse_raw = row.get("interval_collapse_warning", False)
        collapse = bool(collapse_raw)

        candidate = CandidateMetrics(
            model=model,
            trial_id=str(row["trial_id"]),
            MAE_mean=float(row["MAE_mean"]),
            RMSE_mean=float(row["RMSE_mean"]),
            mean_pinball_loss_mean=float(row["mean_pinball_loss_mean"]),
            coverage_80_mean=float(row["coverage_80_mean"]),
            interval_width_mean=float(row["interval_width_mean"]),
            quantile_crossing_rate=crossing,
            interval_collapse_warning=collapse,
        )

        decision = evaluate_candidate_against_baseline(candidate, baseline_by_model[model], gate)
        if decision.promoted:
            accepted_count += 1

        recomputed_status = "promoted" if decision.promoted else "rejected"
        imported_status_raw = row.get("promotion_status")
        imported_status = (
            None
            if imported_status_raw is None or pd.isna(imported_status_raw)
            else str(imported_status_raw)
        )
        status_match = imported_status is None or imported_status == recomputed_status
        if not status_match:
            mismatch_count += 1

        records.append(
            ImportedCandidate(
                model=model,
                trial_id=str(row["trial_id"]),
                MAE_mean=float(row["MAE_mean"]),
                RMSE_mean=float(row["RMSE_mean"]),
                mean_pinball_loss_mean=float(row["mean_pinball_loss_mean"]),
                coverage_80_mean=float(row["coverage_80_mean"]),
                interval_width_mean=float(row["interval_width_mean"]),
                quantile_crossing_rate=crossing,
                interval_collapse_warning=collapse,
                imported_promotion_status=imported_status,
                recomputed_promotion_status=recomputed_status,
                recomputed_rejection_reason=decision.rejection_reason,
                status_match=status_match,
            )
        )

    rejection_count = len(records) - accepted_count
    if accepted_count > 0:
        overall = "promoted"
    elif rejection_count > 0:
        overall = "no_promotion"
    else:
        overall = "failed_resource_limited"

    return ImportedEvaluation(
        candidates=records,
        recomputed_overall_status=overall,
        mismatch_count=mismatch_count,
        rejection_count=rejection_count,
        accepted_count=accepted_count,
    )


def import_tuning_results(
    zone: str,
    ranked_candidates_path: Path,
    baseline_metrics_path: Path,
    summary_path: Path | None = None,
    gate: PromotionGate | None = None,
) -> ImportedTuningResult:
    ranked = pd.read_csv(ranked_candidates_path)
    validate_imported_result_schema(ranked)

    baseline_by_model = _load_baseline_metrics(baseline_metrics_path)
    effective_gate = gate or PromotionGate(
        coverage_min=0.70,
        coverage_max=0.90,
        mae_regression_limit=0.15,
        require_no_quantile_crossing=True,
        require_no_interval_collapse=True,
    )
    evaluation = evaluate_imported_candidates(ranked, baseline_by_model, effective_gate)

    if summary_path is not None and summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Imported summary JSON must be an object")

    return ImportedTuningResult(
        zone=zone,
        ranked_candidates_path=ranked_candidates_path,
        summary_path=summary_path,
        baseline_metrics_path=baseline_metrics_path,
        imported_candidates=evaluation.candidates,
        recomputed_overall_status=evaluation.recomputed_overall_status,
        mismatch_count=evaluation.mismatch_count,
        rejection_count=evaluation.rejection_count,
        accepted_count=evaluation.accepted_count,
    )


def write_import_validation_report(
    result: ImportedTuningResult,
    output_root: Path | None = None,
) -> dict[str, Path]:
    root = output_root or Path("data/cache/reports")
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    zone = result.zone.lower()
    json_path = root / f"{zone}_import_validation_{stamp}.json"
    md_path = root / f"{zone}_import_validation_{stamp}.md"

    payload = result.to_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Imported tuning validation — {result.zone}",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Ranked candidates: {result.ranked_candidates_path.as_posix()}",
        f"Summary path: "
        f"{result.summary_path.as_posix() if result.summary_path else '(not provided)'}",
        f"Baseline metrics: {result.baseline_metrics_path.as_posix()}",
        f"Recomputed overall status: {result.recomputed_overall_status}",
        f"Accepted candidates: {result.accepted_count}",
        f"Rejected candidates: {result.rejection_count}",
        f"Status mismatches: {result.mismatch_count}",
        "",
        "## Candidate checks",
    ]

    if result.imported_candidates:
        for item in result.imported_candidates:
            lines.append(
                "- "
                f"{item.model}/{item.trial_id}: "
                f"imported={item.imported_promotion_status or '(missing)'}; "
                f"recomputed={item.recomputed_promotion_status}; "
                f"match={item.status_match}; "
                f"reason={item.recomputed_rejection_reason or 'n/a'}"
            )
    else:
        lines.append("- No candidates found")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
