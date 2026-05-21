"""CLI utilities for local project operations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.config.settings import get_settings
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


if __name__ == "__main__":
    app()
