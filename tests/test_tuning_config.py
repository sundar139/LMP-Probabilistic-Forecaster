"""Focused tuning config validation tests."""

from __future__ import annotations

from pathlib import Path

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


def _write_tuning_conf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "resource_profiles:\n"
        "  local_safe:\n"
        "    description: Local laptop-safe profile for 8GB VRAM / 16GB RAM\n"
        "    max_trials: 2\n"
        "    folds: 1\n"
        "    max_steps_cap: 3\n"
        "    batch_size: 4\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: false\n"
        "  cloud_16gb:\n"
        "    description: Moderate cloud GPU profile, intended for 16GB VRAM\n"
        "    max_trials: 12\n"
        "    folds: 2\n"
        "    max_steps_cap: 50\n"
        "    batch_size: 8\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n"
        "  cloud_24gb:\n"
        "    description: Larger cloud GPU profile, intended for 24GB+ VRAM\n"
        "    max_trials: 30\n"
        "    folds: 3\n"
        "    max_steps_cap: 100\n"
        "    batch_size: 16\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n",
        encoding="utf-8",
    )


def test_local_safe_and_cloud_profiles_load_from_config(tmp_path: Path) -> None:
    conf_path = tmp_path / "conf/tuning.yaml"
    _write_tuning_conf(conf_path)

    cfg = load_tuning_config(conf_path)

    assert cfg.resource_profiles["local_safe"].max_trials == 2
    assert cfg.resource_profiles["local_safe"].folds == 1
    assert cfg.resource_profiles["cloud_16gb"].max_trials == 12
    assert cfg.resource_profiles["cloud_16gb"].folds == 2
    assert cfg.resource_profiles["cloud_24gb"].max_trials == 30
    assert cfg.resource_profiles["cloud_24gb"].folds == 3


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
