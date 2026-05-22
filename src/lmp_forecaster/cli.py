"""CLI utilities for local project operations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.config.settings import get_settings
from lmp_forecaster.data.build_panel import (
    PanelBuildConfig,
    build_single_zone_panel,
    summarize_panel,
    write_panel,
)
from lmp_forecaster.data.http_client import HttpRequestError
from lmp_forecaster.data.openmeteo_weather import (
    OpenMeteoRequestConfig,
    pull_historical_forecast_smoke,
    pull_historical_weather_smoke,
)
from lmp_forecaster.data.pjm_lmp import PjmLmpRequestConfig, pull_pjm_lmp_smoke
from lmp_forecaster.data.source_registry import load_source_registry
from lmp_forecaster.data.synthetic_panel import SyntheticPanelConfig, make_synthetic_panel
from lmp_forecaster.data.zones import get_zone, load_zone_registry
from lmp_forecaster.eval.panel_report import build_panel_summary, write_panel_summary
from lmp_forecaster.models.baselines import (
    BaselineTrainingConfig,
    load_training_config,
    train_single_zone_baselines,
)

app = typer.Typer(help="LMP forecaster CLI")


def _today_utc() -> date:
    return date.today()


def _resolve_window(days_back: int) -> tuple[date, date]:
    end = _today_utc() - timedelta(days=1)
    start = end - timedelta(days=days_back - 1)
    return start, end


def _parse_iso_date_option(value: str | None, *, option_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid {option_name} date '{value}'. Expected YYYY-MM-DD."
        ) from exc


@app.command("show-config")
def show_config() -> None:
    """Print resolved runtime settings."""
    settings = get_settings()
    paths = get_project_paths()

    print("[bold]Runtime configuration[/bold]")
    print(f"Environment: {settings.environment}")
    print(f"Timezone: {settings.timezone}")
    print(f"Forecast horizon: {settings.forecast_horizon}")
    print(f"Input size: {settings.input_size}")
    print(f"Quantiles: {settings.quantiles}")
    print(f"Default zone: {settings.default_zone}")
    print(f"Project root: {paths.root}")


@app.command("list-sources")
def list_sources() -> None:
    """Print configured source registry."""
    registry = load_source_registry()
    print("[bold]Configured sources[/bold]")
    for source in registry.sources:
        print(
            f"- {source.name} | provider={source.provider} | "
            f"frequency={source.expected_frequency}"
        )


@app.command("list-zones")
def list_zones() -> None:
    """Print configured zone metadata."""
    registry = load_zone_registry()
    print("[bold]Configured zones[/bold]")
    for zone in registry.zones:
        print(
            f"- {zone.zone} ({zone.type}) | {zone.display_name} | "
            f"lat={zone.latitude:.2f}, lon={zone.longitude:.2f} | "
            f"cluster={zone.region_cluster}"
        )


@app.command("pull-pjm-smoke")
def pull_pjm_smoke_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    max_rows: Annotated[
        int,
        typer.Option(help="Maximum rows for smoke pull.", min=1, max=5000),
    ] = 300,
    write: Annotated[
        bool,
        typer.Option(help="Write output parquet under data/cache when set."),
    ] = False,
) -> None:
    """PJM Day-Ahead Hourly LMP smoke pull."""
    zone_meta = get_zone(zone)
    default_start, default_end = _resolve_window(2)
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    cfg = PjmLmpRequestConfig(
        start_date=parsed_start or default_start,
        end_date=parsed_end or default_end,
        locations=[zone_meta.zone],
        max_rows=max_rows,
    )

    try:
        result = pull_pjm_lmp_smoke(cfg, write=write)
    except HttpRequestError as exc:
        print("PJM smoke pull failed.")
        print(f"Error: {exc}")
        print("Action: verify Data Miner params/feed format in pjm_lmp.py request builder.")
        raise typer.Exit(code=2) from exc
    print("[bold]PJM smoke pull[/bold]")
    print(f"Request URL: {result.request_url}")
    print(f"Request params: {result.request_params}")
    print(f"Output path: {result.output_path}")
    if write:
        print(f"Wrote rows: {len(result.normalized)}")
    else:
        print("Dry-run only. Re-run with --write to persist parquet output.")


@app.command("pull-weather-smoke")
def pull_weather_smoke_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Write output parquet under data/cache when set."),
    ] = False,
) -> None:
    """Open-Meteo historical weather smoke pull."""
    zone_meta = get_zone(zone)
    default_start, default_end = _resolve_window(7)
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    cfg = OpenMeteoRequestConfig(
        latitude=zone_meta.latitude,
        longitude=zone_meta.longitude,
        start_date=parsed_start or default_start,
        end_date=parsed_end or default_end,
    )

    result = pull_historical_weather_smoke(cfg, write=write)
    print("[bold]Open-Meteo historical weather smoke pull[/bold]")
    print(f"Request URL: {result.request_url}")
    print(f"Request params: {result.request_params}")
    print(f"Output path: {result.output_path}")
    if write:
        print(f"Wrote rows: {len(result.normalized)}")
    else:
        print("Dry-run only. Re-run with --write to persist parquet output.")


@app.command("pull-historical-forecast-smoke")
def pull_historical_forecast_smoke_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Write output parquet under data/cache when set."),
    ] = False,
) -> None:
    """Open-Meteo historical forecast API smoke pull foundation."""
    zone_meta = get_zone(zone)
    default_start, default_end = _resolve_window(2)
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    cfg = OpenMeteoRequestConfig(
        latitude=zone_meta.latitude,
        longitude=zone_meta.longitude,
        start_date=parsed_start or default_start,
        end_date=parsed_end or default_end,
    )

    result = pull_historical_forecast_smoke(cfg, write=write)
    print("[bold]Open-Meteo historical forecast smoke pull[/bold]")
    print(f"Request URL: {result.request_url}")
    print(f"Request params: {result.request_params}")
    print(f"Output path: {result.output_path}")
    if write:
        print(f"Wrote rows: {len(result.normalized)}")
    else:
        print("Dry-run only. Re-run with --write to persist parquet output.")


@app.command("make-synthetic-panel")
def make_synthetic_panel_command(
    output: Annotated[
        Path,
        typer.Option(help="Parquet output path (default under ignored cache directory)."),
    ] = Path("data/cache/synthetic_panel.parquet"),
    periods: Annotated[int, typer.Option(min=24, help="Number of hourly periods.")] = 240,
    write: Annotated[bool, typer.Option(help="Persist output parquet file when set.")] = False,
) -> None:
    """Generate a deterministic synthetic panel for smoke checks."""
    cfg = SyntheticPanelConfig(periods=periods)
    panel = make_synthetic_panel(cfg)

    print(f"Rows: {len(panel)}")
    print(f"Columns: {list(panel.columns)}")
    print(f"Time span: {panel['ds'].min()} -> {panel['ds'].max()}")

    if write:
        paths = get_project_paths()
        resolved_output = (paths.root / output).resolve() if not output.is_absolute() else output
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(resolved_output, index=False)
        print(f"Wrote synthetic panel to: {resolved_output}")
    else:
        print("No file written. Use --write to persist parquet output.")


@app.command("build-single-zone-panel")
def build_single_zone_panel_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    lmp_path: Annotated[
        Path | None,
        typer.Option(help="Optional parquet path for normalized LMP input."),
    ] = None,
    weather_path: Annotated[
        Path | None,
        typer.Option(help="Optional parquet path for normalized weather input."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional parquet output path."),
    ] = None,
    allow_synthetic_lmp: Annotated[
        bool,
        typer.Option(help="Allow deterministic synthetic LMP fallback."),
    ] = False,
    allow_synthetic_weather: Annotated[
        bool,
        typer.Option(help="Allow deterministic synthetic weather fallback."),
    ] = False,
    write: Annotated[
        bool,
        typer.Option(help="Write panel parquet under processed path when set."),
    ] = False,
    summary: Annotated[
        bool,
        typer.Option(help="Write panel summary JSON under cache reports when set."),
    ] = False,
) -> None:
    """Build a cleaned single-zone panel with leakage-safe features."""
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")

    cfg = PanelBuildConfig(
        zone=zone.upper(),
        input_lmp_path=lmp_path,
        input_weather_path=weather_path,
        output_path=output,
        start_date=parsed_start,
        end_date=parsed_end,
        allow_synthetic_lmp=allow_synthetic_lmp,
        allow_synthetic_weather=allow_synthetic_weather,
    )

    paths = get_project_paths()
    planned_output = cfg.output_path or (
        paths.root / "data" / "processed" / "panel" / "single_zone" / f"{cfg.zone}_panel.parquet"
    )

    print("[bold]Single-zone panel build[/bold]")
    print(f"Zone: {cfg.zone}")
    print(f"LMP input: {cfg.input_lmp_path or '(auto-cache lookup)'}")
    print(f"Weather input: {cfg.input_weather_path or '(auto-cache lookup)'}")
    print(f"Start date: {cfg.start_date}")
    print(f"End date: {cfg.end_date}")
    print(f"Output path: {planned_output}")
    print(f"allow_synthetic_lmp={cfg.allow_synthetic_lmp}")
    print(f"allow_synthetic_weather={cfg.allow_synthetic_weather}")

    if cfg.allow_synthetic_lmp or cfg.allow_synthetic_weather:
        print(
            "WARNING: Synthetic fallback enabled. Output is for pipeline smoke testing "
            "and is not a real PJM training dataset."
        )

    if not write:
        print("Dry-run only. Re-run with --write to build and persist the panel.")
        return

    panel = build_single_zone_panel(cfg)
    written = write_panel(panel, cfg)
    stats = summarize_panel(panel)

    print(f"Wrote panel: {written}")
    print(f"Rows: {stats['rows']}")
    print(f"Range: {stats['start_ds']} -> {stats['end_ds']}")

    if summary:
        report = build_panel_summary(panel, zone=cfg.zone)
        report_path = write_panel_summary(report)
        print(f"Wrote summary: {report_path}")


@app.command("train-single-zone-baselines")
def train_single_zone_baselines_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    panel_path: Annotated[
        Path | None,
        typer.Option(help="Optional panel parquet path."),
    ] = None,
    allow_synthetic_panel: Annotated[
        bool,
        typer.Option(help="Allow synthetic panel fallback."),
    ] = False,
    build_panel_if_missing: Annotated[
        bool,
        typer.Option(help="Build panel if missing using panel builder."),
    ] = False,
    max_steps_smoke: Annotated[
        int | None,
        typer.Option(help="Override smoke max training steps."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional baseline artifact output directory."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Run training and write artifacts when set."),
    ] = False,
    skip_tft: Annotated[
        bool,
        typer.Option(help="Skip TFT model training."),
    ] = False,
    skip_deepar: Annotated[
        bool,
        typer.Option(help="Skip DeepAR model training."),
    ] = False,
) -> None:
    """Train single-zone TFT/DeepAR baselines with probabilistic outputs."""
    cfg = load_training_config(zone.upper(), max_steps_smoke=max_steps_smoke)
    cfg = BaselineTrainingConfig(
        **{
            **cfg.__dict__,
            "zone": zone.upper(),
            "panel_path": panel_path,
            "allow_synthetic_panel": allow_synthetic_panel,
            "build_panel_if_missing": build_panel_if_missing,
            "output_dir": output_dir,
            "skip_tft": skip_tft,
            "skip_deepar": skip_deepar,
        }
    )

    paths = get_project_paths()
    resolved_panel = cfg.panel_path or (
        paths.root / "data" / "processed" / "panel" / "single_zone" / f"{cfg.zone}_panel.parquet"
    )

    print("[bold]Single-zone baseline training[/bold]")
    print(f"Zone: {cfg.zone}")
    print(f"Panel path: {resolved_panel}")
    print(f"allow_synthetic_panel={cfg.allow_synthetic_panel}")
    print(f"build_panel_if_missing={cfg.build_panel_if_missing}")
    print(f"horizon={cfg.horizon}, input_size={cfg.input_size}, quantiles={cfg.quantiles}")
    print(
        f"val_size={cfg.val_size}, test_size={cfg.test_size}, "
        f"max_steps_smoke={cfg.max_steps_smoke}"
    )
    print(f"skip_tft={cfg.skip_tft}, skip_deepar={cfg.skip_deepar}")
    print(f"Forecast cache path: {paths.root / 'data' / 'cache' / 'forecasts'}")
    print(f"Report cache path: {paths.root / 'data' / 'cache' / 'reports'}")
    print(f"Artifact path: {cfg.output_dir or (paths.root / 'artifacts' / 'baselines')}")

    if cfg.allow_synthetic_panel:
        print(
            "Synthetic panel option enabled. If synthetic data is used, metrics are smoke-test "
            "metrics only and must not be presented as PJM performance."
        )

    if not write:
        print("Dry-run only. Re-run with --write to train and persist outputs.")
        return

    result = train_single_zone_baselines(cfg)

    if result.get("data_source_label") in {"synthetic", "mixed"}:
        print(
            "Synthetic panel detected. Metrics are smoke-test metrics only and must not "
            "be presented as PJM performance."
        )

    print(f"Accelerator: {result['accelerator']} ({result['device_name']})")
    print(f"Forecast outputs: {result['forecasts']}")
    print(f"Metrics JSON: {result['metrics_json']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    print(f"Training report: {result['training_report_json']}")


if __name__ == "__main__":
    app()
