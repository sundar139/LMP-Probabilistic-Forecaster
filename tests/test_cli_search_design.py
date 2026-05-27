"""CLI tests for focused search design command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lmp_forecaster.cli import app


def _write_diag_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "zone": "AEP",
        "target_coverage": 0.80,
        "models": ["TFT", "DeepAR"],
        "model_summary": [
            {
                "model": "TFT",
                "coverage_80": 0.5833,
                "interval_width_mean": 13.475,
                "calibration_status": "under-coverage",
                "classification_note": "under-coverage",
                "interval_collapse_warning": False,
            },
            {
                "model": "DeepAR",
                "coverage_80": 0.0,
                "interval_width_mean": 5.047,
                "calibration_status": "under-coverage",
                "classification_note": "interval collapse warning",
                "interval_collapse_warning": True,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cli_design_focused_search_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        diag = Path("data/cache/reports/aep_calibration_diagnostics_test.json")
        _write_diag_json(diag)

        result = runner.invoke(
            app,
            [
                "design-focused-search",
                "--zone",
                "AEP",
                "--diagnostics-path",
                str(diag),
            ],
        )
        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        files = list(Path("data/cache/reports").glob("aep_focused_search_design_*"))
        assert not files


def test_cli_design_focused_search_write_writes_ignored_report_path(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        diag = Path("data/cache/reports/aep_calibration_diagnostics_test.json")
        _write_diag_json(diag)

        result = runner.invoke(
            app,
            [
                "design-focused-search",
                "--zone",
                "AEP",
                "--diagnostics-path",
                str(diag),
                "--write",
            ],
        )
        assert result.exit_code == 0
        json_files = list(Path("data/cache/reports").glob("aep_focused_search_design_*.json"))
        md_files = list(Path("data/cache/reports").glob("aep_focused_search_design_*.md"))
        assert json_files
        assert md_files
        assert "data/cache/reports" in str(json_files[0]).replace("\\", "/")
