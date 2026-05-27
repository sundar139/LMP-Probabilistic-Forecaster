"""Imported tuning result schema and gate validation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lmp_forecaster.tuning.result_import import import_tuning_results


def _write_baseline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "TFT",
                "MAE_mean": 5.5506,
                "RMSE_mean": 6.2262,
                "mean_pinball_loss_mean": 1.8597,
                "coverage_80_mean": 0.5833,
                "interval_width_mean": 13.4750,
                "data_source_label": "real",
            }
        ]
    ).to_csv(path, index=False)


def _candidate_row(
    *,
    trial_id: str,
    mae: float,
    rmse: float,
    pinball: float,
    coverage: float,
    width: float,
    crossing: float = 0.0,
    collapse: bool = False,
    promotion_status: str = "rejected",
) -> dict[str, object]:
    return {
        "model": "TFT",
        "trial_id": trial_id,
        "MAE_mean": mae,
        "RMSE_mean": rmse,
        "mean_pinball_loss_mean": pinball,
        "coverage_80_mean": coverage,
        "interval_width_mean": width,
        "quantile_crossing_rate": crossing,
        "interval_collapse_warning": collapse,
        "promotion_status": promotion_status,
    }


def test_import_validates_required_candidate_fields(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame([{"model": "TFT", "trial_id": "tft_001"}]).to_csv(ranked_path, index=False)

    with pytest.raises(ValueError, match="missing required fields"):
        import_tuning_results(
            zone="AEP",
            ranked_candidates_path=ranked_path,
            baseline_metrics_path=baseline_path,
        )


def test_import_recomputes_promotion_decision_locally(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame(
        [
            _candidate_row(
                trial_id="tft_001",
                mae=5.0,
                rmse=6.0,
                pinball=1.5,
                coverage=0.8,
                width=12.0,
                promotion_status="rejected",
            )
        ]
    ).to_csv(ranked_path, index=False)

    result = import_tuning_results(
        zone="AEP",
        ranked_candidates_path=ranked_path,
        baseline_metrics_path=baseline_path,
    )

    assert result.accepted_count == 1
    assert result.recomputed_overall_status == "promoted"
    assert result.imported_candidates[0].recomputed_promotion_status == "promoted"


def test_import_detects_promotion_label_mismatch(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame(
        [
            _candidate_row(
                trial_id="tft_001",
                mae=5.0,
                rmse=6.0,
                pinball=1.5,
                coverage=0.8,
                width=12.0,
                promotion_status="rejected",
            )
        ]
    ).to_csv(ranked_path, index=False)

    result = import_tuning_results(
        zone="AEP",
        ranked_candidates_path=ranked_path,
        baseline_metrics_path=baseline_path,
    )

    assert result.mismatch_count == 1
    assert result.imported_candidates[0].status_match is False


def test_import_rejects_under_covered_candidate(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame(
        [
            _candidate_row(
                trial_id="tft_002",
                mae=5.0,
                rmse=6.0,
                pinball=1.5,
                coverage=0.4,
                width=12.0,
                promotion_status="promoted",
            )
        ]
    ).to_csv(ranked_path, index=False)

    result = import_tuning_results(
        zone="AEP",
        ranked_candidates_path=ranked_path,
        baseline_metrics_path=baseline_path,
    )

    candidate = result.imported_candidates[0]
    assert candidate.recomputed_promotion_status == "rejected"
    assert candidate.recomputed_rejection_reason is not None
    assert "coverage below gate" in candidate.recomputed_rejection_reason


def test_import_rejects_interval_collapse_candidate(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame(
        [
            _candidate_row(
                trial_id="tft_003",
                mae=5.0,
                rmse=6.0,
                pinball=1.5,
                coverage=0.8,
                width=12.0,
                collapse=True,
                promotion_status="promoted",
            )
        ]
    ).to_csv(ranked_path, index=False)

    result = import_tuning_results(
        zone="AEP",
        ranked_candidates_path=ranked_path,
        baseline_metrics_path=baseline_path,
    )

    candidate = result.imported_candidates[0]
    assert candidate.recomputed_promotion_status == "rejected"
    assert candidate.recomputed_rejection_reason is not None
    assert "interval collapse warning present" in candidate.recomputed_rejection_reason


def test_import_accepts_valid_candidate_that_passes_gates(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.csv"
    baseline_path = tmp_path / "baseline.csv"
    _write_baseline(baseline_path)

    pd.DataFrame(
        [
            _candidate_row(
                trial_id="tft_004",
                mae=5.0,
                rmse=6.0,
                pinball=1.4,
                coverage=0.8,
                width=12.0,
                crossing=0.0,
                collapse=False,
                promotion_status="promoted",
            )
        ]
    ).to_csv(ranked_path, index=False)

    result = import_tuning_results(
        zone="AEP",
        ranked_candidates_path=ranked_path,
        baseline_metrics_path=baseline_path,
    )

    assert result.accepted_count == 1
    assert result.rejection_count == 0
    assert result.recomputed_overall_status == "promoted"
    assert result.imported_candidates[0].status_match is True
