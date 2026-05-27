"""Tests for rolling backtest execution CLI command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def test_cli_backtest_execution_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("conf").mkdir(parents=True, exist_ok=True)
        Path("conf/training.yaml").write_text(
            "training:\n"
            "  horizon_hours: 24\n"
            "  input_size_hours: 168\n"
            "  quantiles: [0.1, 0.5, 0.9]\n"
            "  interval_level: 80\n"
            "  validation_hours: 72\n"
            "  test_hours: 72\n"
            "  seed: 42\n"
            "  max_steps_smoke: 30\n"
            "  max_steps_real_candidate: 200\n"
            "  batch_size: 32\n"
            "  num_workers: 0\n"
            "  accelerator: auto\n",
            encoding="utf-8",
        )

        panel_path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        panel_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        pd.DataFrame(
            {
                "unique_id": ["AEP"] * 300,
                "ds": pd.date_range(
                    "2024-01-01 00:00:00",
                    periods=300,
                    freq="h",
                    tz="America/New_York",
                ),
                "y": [float(i % 50) for i in range(300)],
                "source_label": ["real"] * 300,
            }
        ).to_parquet(panel_path, index=False)

        result = runner.invoke(
            app,
            [
                "run-rolling-backtest",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel_path),
                "--folds",
                "2",
                "--horizon-hours",
                "24",
                "--min-train-hours",
                "168",
            ],
        )
        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        assert not Path("data/cache/backtests").exists()


def test_cli_backtest_execution_missing_panel_fails_clearly(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "run-rolling-backtest",
                "--zone",
                "AEP",
                "--panel-path",
                "data/processed/panel/single_zone/AEP_panel.parquet",
                "--folds",
                "2",
                "--horizon-hours",
                "24",
            ],
        )
        assert result.exit_code == 2
        assert "Real AEP panel is missing" in result.stdout


def test_cli_backtest_write_calls_runner_and_prints_paths(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()

    captured = {}

    def fake_run_rolling_backtest(cfg):  # type: ignore[no-untyped-def]
        captured["cfg"] = cfg
        return SimpleNamespace(accelerator="cpu", device_name="CPU", tracking=None)

    def fake_write_backtest_results(result):  # type: ignore[no-untyped-def]
        return {
            "forecasts": Path("data/cache/backtests/aep_rolling_backtest_forecasts_x.parquet"),
            "fold_metrics": Path("data/cache/backtests/aep_rolling_backtest_fold_metrics_x.csv"),
            "aggregate_metrics": Path(
                "data/cache/backtests/aep_rolling_backtest_aggregate_metrics_x.csv"
            ),
            "summary_json": Path("data/cache/reports/aep_rolling_backtest_summary_x.json"),
            "summary_markdown": Path("data/cache/reports/aep_rolling_backtest_summary_x.md"),
            "manifest": Path("artifacts/backtests/aep_rolling_backtest_manifest_x.json"),
        }

    def fake_log_backtest_tracking(result, paths):  # type: ignore[no-untyped-def]
        return {"enabled": False, "reason": "tracking_disabled", "tracking_uri": "file:./mlruns"}

    monkeypatch.setattr("lmp_forecaster.cli.run_rolling_backtest", fake_run_rolling_backtest)
    monkeypatch.setattr("lmp_forecaster.cli.write_backtest_results", fake_write_backtest_results)
    monkeypatch.setattr("lmp_forecaster.cli.log_backtest_tracking", fake_log_backtest_tracking)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("conf").mkdir(parents=True, exist_ok=True)
        Path("conf/training.yaml").write_text(
            "training:\n"
            "  horizon_hours: 24\n"
            "  input_size_hours: 168\n"
            "  quantiles: [0.1, 0.5, 0.9]\n"
            "  interval_level: 80\n"
            "  validation_hours: 72\n"
            "  test_hours: 72\n"
            "  seed: 42\n"
            "  max_steps_smoke: 30\n"
            "  max_steps_real_candidate: 200\n"
            "  batch_size: 32\n"
            "  num_workers: 0\n"
            "  accelerator: auto\n",
            encoding="utf-8",
        )

        panel_path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        panel_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        pd.DataFrame(
            {
                "unique_id": ["AEP"] * 300,
                "ds": pd.date_range(
                    "2024-01-01 00:00:00",
                    periods=300,
                    freq="h",
                    tz="America/New_York",
                ),
                "y": [float(i % 50) for i in range(300)],
                "source_label": ["real"] * 300,
            }
        ).to_parquet(panel_path, index=False)

        result = runner.invoke(
            app,
            [
                "run-rolling-backtest",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel_path),
                "--folds",
                "2",
                "--horizon-hours",
                "24",
                "--min-train-hours",
                "168",
                "--write",
            ],
        )

        assert result.exit_code == 0
        assert "Forecast outputs:" in result.stdout
        assert "Fold metrics CSV:" in result.stdout
        assert "Summary JSON:" in result.stdout
        assert captured["cfg"].zone == "AEP"
