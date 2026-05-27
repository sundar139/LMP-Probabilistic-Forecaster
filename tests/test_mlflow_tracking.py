"""Tests for optional MLflow tracking wrappers and safety guards."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from lmp_forecaster.eval.backtest import BacktestFold
from lmp_forecaster.eval.backtest_runner import (
    BacktestRunConfig,
    RollingBacktestResult,
    log_backtest_tracking,
)
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
        return SimpleNamespace(
            info=SimpleNamespace(run_name=self.run_name, run_id="dummy-run-id")
        )

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


def test_mlflow_disabled_does_not_create_mlruns(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)

    cfg = BacktestRunConfig(enable_tracking=False)
    result = RollingBacktestResult(
        config=cfg,
        folds=[
            BacktestFold(
                fold_id=1,
                train_start=SimpleNamespace(),
                train_end=SimpleNamespace(),
                origin=SimpleNamespace(),
                test_start=SimpleNamespace(),
                test_end=SimpleNamespace(),
                train_rows=1,
                test_rows=1,
                horizon_hours=24,
                leakage_check_passed=True,
                overlap_check_passed=True,
            )
        ],
        forecasts=SimpleNamespace(),
        fold_metrics=SimpleNamespace(),
        aggregate_metrics=SimpleNamespace(to_dict=lambda orient: []),
        accelerator="cpu",
        device_name="CPU",
        data_source_label="real",
    )
    status = log_backtest_tracking(result, {})
    assert status["enabled"] is False
    assert not Path("mlruns").exists()


def test_mlflow_artifact_scratch_dir_is_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert ".mlflow_artifacts/" in gitignore


def test_mlflow_enabled_backtest_logging_filters_secret_params(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dummy = _DummyMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy)

    cfg = BacktestRunConfig(
        enable_tracking=True,
        tracking_uri="file:./mlruns",
        experiment_name="backtest_smoke",
    )

    aggregate_metrics = SimpleNamespace(
        to_dict=lambda orient: [
            {
                "model": "TFT",
                "MAE_mean": 10.0,
                "RMSE_mean": 12.0,
                "mean_pinball_loss_mean": 3.0,
                "coverage_80_mean": 0.5,
            }
        ]
    )

    result = RollingBacktestResult(
        config=cfg,
        folds=[],
        forecasts=SimpleNamespace(),
        fold_metrics=SimpleNamespace(),
        aggregate_metrics=aggregate_metrics,
        accelerator="cpu",
        device_name="CPU",
        data_source_label="real",
    )

    artifact_path = tmp_path / "data/cache/backtests" / "a.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("ok\n", encoding="utf-8")

    status = log_backtest_tracking(result, {"fold_metrics": artifact_path})
    assert status["enabled"] is True
    assert status["run_id"] == "dummy-run-id"

    # Ensure sensible params are logged and secret-like keys are absent.
    assert "zone" in dummy.params
    assert "models" in dummy.params
    assert all("key" not in key.lower() for key in dummy.params)
