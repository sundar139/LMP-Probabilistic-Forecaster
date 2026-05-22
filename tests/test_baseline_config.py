"""Tests for baseline training config loading."""

from __future__ import annotations

from lmp_forecaster.models.baselines import load_training_config


def test_training_config_defaults_load() -> None:
    cfg = load_training_config("AEP")
    assert cfg.zone == "AEP"
    assert cfg.horizon == 24
    assert cfg.input_size == 168
    assert cfg.quantiles == (0.1, 0.5, 0.9)
    assert cfg.val_size == 72
    assert cfg.test_size == 72
    assert cfg.max_steps_smoke == 30
