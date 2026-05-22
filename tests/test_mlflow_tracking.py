"""Tests for optional MLflow tracking wrappers and safety guards."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from lmp_forecaster.tracking.mlflow_utils import (
    TrackingConfig,
    configure_mlflow,
    get_tracking_uri,
    safe_log_params,
    start_mlflow_run,
)


class _DummyMlflow(ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []
        self._tracking_uri = ""
        self._experiment_name = ""

    def set_tracking_uri(self, uri: str) -> None:
        self._tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self._experiment_name = name

    def start_run(self, run_name: str):
        return _DummyRun(run_name)

    def log_params(self, payload: dict[str, str]) -> None:
        self.params.update(payload)

    def log_metrics(self, payload: dict[str, float]) -> None:
        self.metrics.update(payload)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)


class _DummyRun:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def __enter__(self):
        return SimpleNamespace(info=SimpleNamespace(run_name=self.run_name))

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def test_tracking_disabled_by_default() -> None:
    cfg = TrackingConfig()
    ctx = configure_mlflow(cfg)
    assert ctx.enabled is False
    assert ctx.reason == "tracking_disabled"


def test_get_tracking_uri_default_when_blank() -> None:
    cfg = TrackingConfig(enabled=True, tracking_uri="")
    assert get_tracking_uri(cfg) == "file:./mlruns"


def test_tracking_enabled_uses_wrapper_without_real_server(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dummy = _DummyMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy)

    cfg = TrackingConfig(enabled=True, experiment_name="exp", tracking_uri="file:./mlruns")
    ctx = configure_mlflow(cfg)
    assert ctx.enabled is True

    with start_mlflow_run(ctx, run_name="unit") as run:
        assert run is not None
        safe = safe_log_params(
            run,
            {
                "zone": "AEP",
                "seed": 42,
                "api_key": "secret",
                "password": "secret",
            },
        )

    assert "zone" in safe
    assert "seed" in safe
    assert "api_key" not in safe
    assert "password" not in safe
    assert "api_key" not in dummy.params
    assert "password" not in dummy.params


def test_start_run_noop_when_disabled() -> None:
    ctx = configure_mlflow(TrackingConfig(enabled=False))
    with start_mlflow_run(ctx, run_name="noop") as run:
        assert run is None
