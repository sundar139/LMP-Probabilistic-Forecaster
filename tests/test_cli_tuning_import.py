"""CLI tests for import-tuning-results command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app


def _write_tuning_conf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tuning:\n"
        "  zone: AEP\n"
        "  models: [TFT, DeepAR]\n"
        "  max_trials_first_pass: 12\n"
        "  folds_for_full_first_pass: 2\n"
        "  horizon_hours: 24\n"
        "  max_steps_cap: 60\n"
        "  target_coverage: 0.80\n"
        "  coverage_min: 0.70\n"
        "  coverage_max: 0.90\n"
        "  mae_regression_limit: 0.15\n"
        "  allow_deepar_if_interval_collapse: false\n"
        "resource_profiles:\n"
        "  local_safe:\n"
        "    description: Local laptop-safe profile for 8GB VRAM / 16GB RAM\n"
        "    max_trials: 2\n"
        "    folds: 1\n"
        "    max_steps_cap: 3\n"
        "    batch_size: 4\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: false\n",
        encoding="utf-8",
    )


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


def _write_ranked(path: Path, *, coverage: float, collapse: bool, promotion_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": "TFT",
                "trial_id": "tft_001",
                "MAE_mean": 5.0,
                "RMSE_mean": 6.0,
                "mean_pinball_loss_mean": 1.5,
                "coverage_80_mean": coverage,
                "interval_width_mean": 12.0,
                "quantile_crossing_rate": 0.0,
                "interval_collapse_warning": collapse,
                "promotion_status": promotion_status,
            }
        ]
    ).to_csv(path, index=False)


def test_import_tuning_results_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        ranked = Path("tmp/ranked.csv")
        baseline = Path("tmp/baseline.csv")
        _write_ranked(ranked, coverage=0.8, collapse=False, promotion_status="promoted")
        _write_baseline(baseline)

        result = runner.invoke(
            app,
            [
                "import-tuning-results",
                "--zone",
                "AEP",
                "--ranked-candidates-path",
                str(ranked),
                "--baseline-metrics-path",
                str(baseline),
            ],
        )

        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        assert not list(Path("data/cache/reports").glob("aep_import_validation_*"))


def test_import_tuning_results_write_writes_only_ignored_report_path(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        ranked = Path("tmp/ranked.csv")
        baseline = Path("tmp/baseline.csv")
        summary = Path("tmp/summary.json")
        _write_ranked(ranked, coverage=0.8, collapse=False, promotion_status="promoted")
        _write_baseline(baseline)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"status": "external"}), encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "import-tuning-results",
                "--zone",
                "AEP",
                "--ranked-candidates-path",
                str(ranked),
                "--summary-path",
                str(summary),
                "--baseline-metrics-path",
                str(baseline),
                "--write",
            ],
        )

        assert result.exit_code == 0
        assert "Import validation JSON:" in result.stdout
        assert "Import validation Markdown:" in result.stdout

        outputs = sorted(Path("data/cache/reports").glob("aep_import_validation_*"))
        assert outputs
        for path in outputs:
            rendered = str(path).replace("\\", "/")
            assert rendered.startswith("data/cache/reports/")


def test_import_tuning_results_reports_mismatch_detection(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        _write_tuning_conf(Path("conf/tuning.yaml"))

        ranked = Path("tmp/ranked.csv")
        baseline = Path("tmp/baseline.csv")
        _write_ranked(ranked, coverage=0.8, collapse=False, promotion_status="rejected")
        _write_baseline(baseline)

        result = runner.invoke(
            app,
            [
                "import-tuning-results",
                "--zone",
                "AEP",
                "--ranked-candidates-path",
                str(ranked),
                "--baseline-metrics-path",
                str(baseline),
            ],
        )

        assert result.exit_code == 0
        assert "mismatch_count=1" in result.stdout
