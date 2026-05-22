"""Tests for baseline training config loading and validation."""

from __future__ import annotations

import pytest

from lmp_forecaster.models.baselines import (
    BaselineTrainingConfig,
    _validate_training_config,
    load_training_config,
)


def test_training_config_defaults_load() -> None:
    cfg = load_training_config("AEP")
    assert cfg.zone == "AEP"
    assert cfg.horizon == 24
    assert cfg.input_size == 168
    assert cfg.quantiles == (0.1, 0.5, 0.9)
    assert cfg.interval_level == 80
    assert cfg.val_size == 72
    assert cfg.test_size == 72
    assert cfg.max_steps_smoke == 30
    assert cfg.max_steps_real_candidate == 200


def test_training_config_rejects_bad_quantiles() -> None:
    cfg = BaselineTrainingConfig(quantiles=(0.9, 0.5, 0.1))
    with pytest.raises(ValueError, match="Quantiles must be sorted ascending"):
        _validate_training_config(cfg)


def test_training_config_rejects_bad_interval() -> None:
    cfg = BaselineTrainingConfig(quantiles=(0.1, 0.5, 0.9), interval_level=70)
    with pytest.raises(ValueError, match="interval_level must match quantile span"):
        _validate_training_config(cfg)


def test_training_config_rejects_impossible_split_sizes() -> None:
    cfg = BaselineTrainingConfig(val_size=0, test_size=72)
    with pytest.raises(ValueError, match="validation_hours and test_hours must be positive"):
        _validate_training_config(cfg)
