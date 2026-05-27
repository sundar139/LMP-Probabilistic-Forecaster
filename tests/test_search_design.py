"""Unit tests for focused search design utilities."""

from __future__ import annotations

from lmp_forecaster.tuning.search_design import (
    SearchDesignConfig,
    build_deepar_search_space,
    build_tft_search_space,
    recommend_search_strategy,
)


def _diagnostics_payload() -> dict[str, object]:
    return {
        "zone": "AEP",
        "target_coverage": 0.80,
        "models": ["TFT", "DeepAR"],
        "model_summary": [
            {
                "model": "TFT",
                "coverage_80": 0.5833,
                "interval_width_mean": 13.475,
                "calibration_status": "under-coverage",
                "classification_note": "under-coverage",
                "interval_collapse_warning": False,
            },
            {
                "model": "DeepAR",
                "coverage_80": 0.0,
                "interval_width_mean": 5.0474,
                "calibration_status": "under-coverage",
                "classification_note": "interval collapse warning",
                "interval_collapse_warning": True,
            },
        ],
    }


def test_search_design_includes_tft_and_deepar_spaces() -> None:
    cfg = SearchDesignConfig()
    strategy = recommend_search_strategy(_diagnostics_payload(), cfg)

    spaces = strategy["spaces"]
    assert "TFT" in spaces
    assert "DeepAR" in spaces


def test_tft_space_reason_mentions_under_coverage() -> None:
    cfg = SearchDesignConfig()
    space = build_tft_search_space(_diagnostics_payload(), cfg)
    assert "under-coverage" in space.objective_focus


def test_deepar_space_reason_mentions_interval_collapse() -> None:
    cfg = SearchDesignConfig()
    space = build_deepar_search_space(_diagnostics_payload(), cfg)
    text = space.objective_focus.lower()
    assert "collapse" in text or "under-coverage" in text
