"""CLI tests for calibration diagnostics command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app


def _write_fake_forecasts(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ["TFT", "DeepAR"]:
        for fold_id in [1, 2]:
            for hour in [0, 1]:
                y = 10.0 + hour if model == "TFT" else 30.0 + hour
                p10 = y - 1.0 if model == "TFT" else y + 1.0
                p50 = y
                p90 = y + 1.0 if model == "TFT" else y + 2.0
                rows.append(
                    {
                        "unique_id": "AEP",
                        "ds": pd.Timestamp(
                            f"2024-01-0{fold_id} {hour:02d}:00:00", tz="America/New_York"
                        ),
                        "y": y,
                        "p10": p10,
                        "p50": p50,
                        "p90": p90,
                        "model": model,
                        "generated_at": pd.Timestamp("2026-05-27T00:00:00Z"),
                        "data_source_label": "real",
                        "zone": "AEP",
                        "fold_id": fold_id,
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_cli_analyze_calibration_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        forecasts = Path("data/cache/backtests/aep_rolling_backtest_forecasts_test.parquet")
        _write_fake_forecasts(forecasts)

        result = runner.invoke(
            app,
            [
                "analyze-calibration",
                "--zone",
                "AEP",
                "--forecasts-path",
                str(forecasts),
            ],
        )
        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        reports = Path("data/cache/reports")
        assert not reports.exists()


def test_cli_analyze_calibration_write_writes_ignored_report_path(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        forecasts = Path("data/cache/backtests/aep_rolling_backtest_forecasts_test.parquet")
        _write_fake_forecasts(forecasts)

        result = runner.invoke(
            app,
            [
                "analyze-calibration",
                "--zone",
                "AEP",
                "--forecasts-path",
                str(forecasts),
                "--write",
            ],
        )
        assert result.exit_code == 0
        reports = Path("data/cache/reports")
        json_files = list(reports.glob("aep_calibration_diagnostics_*.json"))
        md_files = list(reports.glob("aep_calibration_diagnostics_*.md"))
        assert json_files
        assert md_files
        assert "data/cache/reports" in str(json_files[0]).replace("\\", "/")
