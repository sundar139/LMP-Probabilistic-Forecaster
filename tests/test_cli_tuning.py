"""CLI tests for focused tuning command behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app


def _write_search_design(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "zone": "AEP",
        "spaces": {
            "TFT": {
                "parameters": [
                    {"name": "dropout", "type": "float", "min": 0.05, "max": 0.10},
                    {"name": "max_steps", "type": "int", "min": 2, "max": 3},
                ]
            },
            "DeepAR": {
                "parameters": [
                    {
                        "name": "distribution_loss",
                        "type": "categorical",
                        "values": ["StudentT", "Normal"],
                    },
                    {"name": "max_steps", "type": "int", "min": 2, "max": 3},
                ]
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_baseline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "model": "TFT",
                "zone": "AEP",
                "folds_completed": 3,
                "total_test_rows": 72,
                "MAE_mean": 5.5506,
                "RMSE_mean": 6.2262,
                "mean_pinball_loss_mean": 1.8597,
                "coverage_80_mean": 0.5833,
                "interval_width_mean": 13.4750,
                "data_source_label": "real",
            },
            {
                "model": "DeepAR",
                "zone": "AEP",
                "folds_completed": 3,
                "total_test_rows": 72,
                "MAE_mean": 21.9064,
                "RMSE_mean": 22.4298,
                "mean_pinball_loss_mean": 10.2479,
                "coverage_80_mean": 0.0,
                "interval_width_mean": 5.0474,
                "data_source_label": "real",
            },
        ]
    )
    frame.to_csv(path, index=False)


def _write_panel(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "unique_id": ["AEP"] * 300,
            "ds": pd.date_range(
                "2024-01-01 00:00:00", periods=300, freq="h", tz="America/New_York"
            ),
            "y": [float(i % 50) for i in range(300)],
            "source_label": ["real"] * 300,
        }
    ).to_parquet(path, index=False)


def _write_tuning_conf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tuning:\n"
        "  zone: AEP\n"
        "  models: [TFT, DeepAR]\n"
        "  max_trials_first_pass: 12\n"
        "  folds_for_full_first_pass: 2\n"
        "  horizon_hours: 24\n"
        "  max_steps_cap: 8\n"
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
        "  num_workers: 0\n"
        "  full_search_deferred: true\n",
        encoding="utf-8",
    )


def test_cli_tuning_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n", encoding="utf-8"
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        panel = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        design = Path("data/cache/reports/aep_focused_search_design_test.json")
        baseline = Path("data/cache/backtests/aep_rolling_backtest_aggregate_metrics_test.csv")
        _write_panel(panel)
        _write_search_design(design)
        _write_baseline(baseline)

        result = runner.invoke(
            app,
            [
                "run-focused-tuning",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel),
                "--search-design-path",
                str(design),
                "--baseline-metrics-path",
                str(baseline),
                "--resource-profile",
                "local_safe",
            ],
        )
        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        assert "resource_profile=local_safe" in result.stdout
        assert not list(Path("data/cache/tuning").glob("aep_focused_tuning_*"))


def test_cli_tuning_refuses_heavy_run_without_allow_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n", encoding="utf-8"
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        panel = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        design = Path("data/cache/reports/aep_focused_search_design_test.json")
        baseline = Path("data/cache/backtests/aep_rolling_backtest_aggregate_metrics_test.csv")
        _write_panel(panel)
        _write_search_design(design)
        _write_baseline(baseline)

        result = runner.invoke(
            app,
            [
                "run-focused-tuning",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel),
                "--search-design-path",
                str(design),
                "--baseline-metrics-path",
                str(baseline),
                "--resource-profile",
                "local_safe",
                "--max-trials",
                "3",
            ],
        )
        assert result.exit_code == 2
        assert "refused heavy run" in result.stdout


def test_cli_tuning_missing_panel_fails_clearly(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n", encoding="utf-8"
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))
        result = runner.invoke(app, ["run-focused-tuning", "--zone", "AEP"])
        assert result.exit_code == 2
        assert "Real AEP panel is missing" in result.stdout


def test_cli_tuning_write_calls_runner_and_writes_expected_paths(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()

    class _DummySummary:
        promotion_summary = {"overall_status": "no_promotion"}

    def fake_run_focused_tuning(cfg, **kwargs):  # type: ignore[no-untyped-def]
        return _DummySummary()

    def fake_write_tuning_results(summary):  # type: ignore[no-untyped-def]
        return {
            "trials": Path("data/cache/tuning/aep_focused_tuning_trials_x.csv"),
            "ranked": Path("data/cache/tuning/aep_focused_tuning_ranked_x.csv"),
            "summary_json": Path("data/cache/reports/aep_focused_tuning_summary_x.json"),
            "summary_markdown": Path("data/cache/reports/aep_focused_tuning_summary_x.md"),
            "manifest": Path("artifacts/tuning/aep_focused_tuning_manifest_x.json"),
        }

    def fake_log_tuning_tracking(summary, output_paths):  # type: ignore[no-untyped-def]
        return {"enabled": False, "reason": "tracking_disabled", "tracking_uri": "file:./mlruns"}

    monkeypatch.setattr("lmp_forecaster.cli.run_focused_tuning", fake_run_focused_tuning)
    monkeypatch.setattr("lmp_forecaster.cli.write_tuning_results", fake_write_tuning_results)
    monkeypatch.setattr("lmp_forecaster.cli.log_tuning_tracking", fake_log_tuning_tracking)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n", encoding="utf-8"
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        panel = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        design = Path("data/cache/reports/aep_focused_search_design_test.json")
        baseline = Path("data/cache/backtests/aep_rolling_backtest_aggregate_metrics_test.csv")
        _write_panel(panel)
        _write_search_design(design)
        _write_baseline(baseline)

        result = runner.invoke(
            app,
            [
                "run-focused-tuning",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel),
                "--search-design-path",
                str(design),
                "--baseline-metrics-path",
                str(baseline),
                "--resource-profile",
                "local_safe",
                "--max-trials",
                "2",
                "--folds",
                "1",
                "--max-steps-cap",
                "3",
                "--skip-deepar",
                "--write",
                "--no-mlflow",
            ],
        )

        assert result.exit_code == 0
        assert "Trial metrics CSV:" in result.stdout
        assert "Summary JSON:" in result.stdout
        assert "Tracking status:" in result.stdout
