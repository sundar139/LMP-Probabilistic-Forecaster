"""CLI tests for export-tuning-package command."""

from __future__ import annotations

import json
from pathlib import Path

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
        "resource_profiles:\n"
        "  local_safe:\n"
        "    description: Local laptop-safe profile for 8GB VRAM / 16GB RAM\n"
        "    max_trials: 2\n"
        "    folds: 1\n"
        "    max_steps_cap: 3\n"
        "    batch_size: 4\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: false\n"
        "  cloud_16gb:\n"
        "    description: Moderate cloud GPU profile, intended for 16GB VRAM\n"
        "    max_trials: 12\n"
        "    folds: 2\n"
        "    max_steps_cap: 50\n"
        "    batch_size: 8\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n"
        "  cloud_24gb:\n"
        "    description: Larger cloud GPU profile, intended for 24GB+ VRAM\n"
        "    max_trials: 30\n"
        "    folds: 3\n"
        "    max_steps_cap: 100\n"
        "    batch_size: 16\n"
        "    num_workers: 0\n"
        "    allow_heavy_run: true\n",
        encoding="utf-8",
    )


def test_export_tuning_package_dry_run_writes_nothing(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("README.md").write_text("# tmp\n", encoding="utf-8")
        _write_tuning_conf(Path("conf/tuning.yaml"))

        result = runner.invoke(
            app,
            [
                "export-tuning-package",
                "--zone",
                "AEP",
                "--resource-profile",
                "cloud_16gb",
                "--models",
                "TFT,DeepAR",
                "--max-trials",
                "12",
                "--folds",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "Dry-run only" in result.stdout
        assert "resource_profile=cloud_16gb" in result.stdout
        assert not (Path("data/cache/tuning_packages").exists())


def test_export_tuning_package_write_writes_only_ignored_output_paths(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        Path("pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.0'\n",
            encoding="utf-8",
        )
        Path("README.md").write_text("# tmp\n", encoding="utf-8")
        Path(".env.example").write_text("PJM_API_KEY=\n", encoding="utf-8")
        _write_tuning_conf(Path("conf/tuning.yaml"))

        result = runner.invoke(
            app,
            [
                "export-tuning-package",
                "--zone",
                "AEP",
                "--resource-profile",
                "cloud_16gb",
                "--models",
                "TFT,DeepAR",
                "--max-trials",
                "12",
                "--folds",
                "2",
                "--write",
            ],
        )

        assert result.exit_code == 0
        assert "Package manifest JSON:" in result.stdout
        assert "Package manifest Markdown:" in result.stdout

        outputs = sorted(Path("data/cache/tuning_packages").glob("aep_tuning_package_*"))
        assert outputs
        for path in outputs:
            rendered = str(path).replace("\\", "/")
            assert rendered.startswith("data/cache/tuning_packages/")

        json_outputs = [p for p in outputs if p.suffix == ".json"]
        assert json_outputs
        payload = json.loads(json_outputs[0].read_text(encoding="utf-8"))
        assert payload["resource_profile"] == "cloud_16gb"
        assert "uv_commands" in payload
