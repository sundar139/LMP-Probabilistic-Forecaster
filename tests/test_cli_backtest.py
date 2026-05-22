"""Tests for rolling backtest planning CLI command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from lmp_forecaster.cli import app


def _panel(rows: int = 8616) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AEP"] * rows,
            "ds": pd.date_range(
                "2024-01-08 00:00:00",
                periods=rows,
                freq="h",
                tz="America/New_York",
            ),
            "y": [float(i % 200) for i in range(rows)],
            "source_label": ["real"] * rows,
        }
    )


def test_cli_plan_rolling_backtest_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        panel_path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        _panel().to_parquet(panel_path, index=False)

        result = runner.invoke(
            app,
            [
                "plan-rolling-backtest",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel_path),
                "--folds",
                "3",
                "--horizon-hours",
                "24",
            ],
        )
        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        reports = Path("data/cache/reports")
        assert not reports.exists() or not any(reports.glob("backtest_plan_AEP_*.json"))


def test_cli_plan_rolling_backtest_write_creates_reports_under_cache(tmp_path: Path) -> None:
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        panel_path = Path("data/processed/panel/single_zone/AEP_panel.parquet")
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        _panel().to_parquet(panel_path, index=False)

        result = runner.invoke(
            app,
            [
                "plan-rolling-backtest",
                "--zone",
                "AEP",
                "--panel-path",
                str(panel_path),
                "--folds",
                "3",
                "--horizon-hours",
                "24",
                "--write",
            ],
        )

        assert result.exit_code == 0
        reports = Path("data/cache/reports")
        jsons = list(reports.glob("backtest_plan_AEP_*.json"))
        mds = list(reports.glob("backtest_plan_AEP_*.md"))
        assert jsons
        assert mds
