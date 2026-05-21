"""Tests for panel builder CLI command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def test_cli_panel_dry_run_does_not_write() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-single-zone-panel",
            "--zone",
            "AEP",
            "--allow-synthetic-lmp",
            "--allow-synthetic-weather",
        ],
    )

    assert result.exit_code == 0
    assert "Dry-run only" in result.stdout


def test_cli_panel_write_and_summary(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("conf").mkdir(parents=True, exist_ok=True)
        Path("conf/zones.yaml").write_text(
            "zones:\n"
            "  - zone: AEP\n"
            "    display_name: AEP\n"
            "    latitude: 39.96\n"
            "    longitude: -82.99\n"
            "    region_cluster: west\n"
            "    type: ZONE\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "build-single-zone-panel",
                "--zone",
                "AEP",
                "--allow-synthetic-lmp",
                "--allow-synthetic-weather",
                "--write",
                "--summary",
            ],
        )
        assert result.exit_code == 0

        processed = list(Path("data/processed/panel/single_zone").rglob("*.parquet"))
        reports = list(Path("data/cache/reports").rglob("*.json"))
        assert len(processed) == 1
        assert len(reports) == 1
