"""Focused tuning config validation tests."""

from __future__ import annotations

import pytest

from lmp_forecaster.tuning.tuning_runner import TuningRunConfig, load_tuning_config


def test_tuning_config_accepts_valid_values() -> None:
    cfg = TuningRunConfig(
        max_trials=12,
        folds=2,
        horizon_hours=24,
        coverage_min=0.70,
        target_coverage=0.80,
        coverage_max=0.90,
        mae_regression_limit=0.15,
    )
    cfg.validate()


def test_tuning_config_rejects_bad_coverage_gates() -> None:
    cfg = TuningRunConfig(
        coverage_min=0.82,
        target_coverage=0.80,
        coverage_max=0.90,
    )
    with pytest.raises(ValueError, match="coverage_min < target_coverage < coverage_max"):
        cfg.validate()


def test_local_safe_profile_defaults_loaded_from_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    conf = tmp_path / "conf"
    conf.mkdir(parents=True, exist_ok=True)
    path = conf / "tuning.yaml"
    path.write_text(
        "tuning:\n"
        "  zone: AEP\n"
        "  models: [TFT, DeepAR]\n"
        "  max_trials_first_pass: 12\n"
        "  folds_for_full_first_pass: 2\n"
        "  horizon_hours: 24\n"
        "  target_coverage: 0.80\n"
        "  coverage_min: 0.70\n"
        "  coverage_max: 0.90\n"
        "  mae_regression_limit: 0.15\n"
        "resource_profile:\n"
        "  name: local_8gb_vram_16gb_ram\n"
        "  max_trials_safe: 2\n"
        "  folds_safe: 1\n"
        "  max_steps_cap_safe: 3\n"
        "  batch_size_safe: 4\n"
        "  num_workers: 0\n",
        encoding="utf-8",
    )
    cfg = load_tuning_config(path)
    assert cfg.profile.max_trials_safe <= 2
    assert cfg.profile.folds_safe <= 1


def test_local_safe_rejects_heavy_trials_without_allow_flag() -> None:
    cfg = TuningRunConfig(
        resource_profile="local_safe",
        max_trials=3,
        folds=1,
        max_steps_cap=3,
        batch_size=4,
        allow_heavy_run=False,
    )
    with pytest.raises(ValueError, match="refused heavy run"):
        cfg.validate()


def test_local_safe_rejects_heavy_folds_without_allow_flag() -> None:
    cfg = TuningRunConfig(
        resource_profile="local_safe",
        max_trials=2,
        folds=2,
        max_steps_cap=3,
        batch_size=4,
        allow_heavy_run=False,
    )
    with pytest.raises(ValueError, match="refused heavy run"):
        cfg.validate()


def test_local_safe_allows_heavy_only_with_explicit_flag() -> None:
    cfg = TuningRunConfig(
        resource_profile="local_safe",
        max_trials=4,
        folds=2,
        max_steps_cap=6,
        batch_size=8,
        allow_heavy_run=True,
    )
    cfg.validate()
