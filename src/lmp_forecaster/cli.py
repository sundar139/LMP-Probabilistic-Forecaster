"""CLI utilities for local project operations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
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
from lmp_forecaster.data.pjm_api import (
    PjmApiError,
    build_day_ahead_lmp_params,
    build_pjm_api_headers,
    effective_max_connections_per_minute,
    effective_pjm_throttle_seconds,
    fetch_pjm_json_page,
    normalize_da_lmp_response,
    redact_headers,
    resolve_pjm_api_config,
)
from lmp_forecaster.data.pjm_backfill import (
    PjmBackfillConfig,
    plan_backfill_chunks,
    run_da_lmp_backfill,
)
from lmp_forecaster.data.pjm_lmp import PjmLmpRequestConfig, pull_pjm_lmp_smoke
from lmp_forecaster.data.real_cache_discovery import (
    locate_latest_lmp_cache,
    locate_latest_lmp_quality_report,
    locate_latest_weather_cache,
    locate_latest_weather_quality_report,
)
from lmp_forecaster.data.source_registry import load_source_registry
from lmp_forecaster.data.synthetic_panel import SyntheticPanelConfig, make_synthetic_panel
from lmp_forecaster.data.weather_backfill import pull_real_weather
from lmp_forecaster.data.zones import get_zone, load_zone_registry
from lmp_forecaster.eval.backtest import (
    BacktestConfig,
    make_rolling_origin_folds,
    summarize_backtest_plan,
    validate_backtest_folds,
    write_backtest_plan,
)
from lmp_forecaster.eval.backtest_runner import (
    BacktestRunConfig,
    load_backtest_panel,
    log_backtest_tracking,
    planned_output_paths,
    run_rolling_backtest,
    write_backtest_results,
)
from lmp_forecaster.eval.calibration import (
    discover_latest_backtest_outputs,
    load_calibration_config,
    summarize_calibration_diagnostics,
    write_calibration_report,
)
from lmp_forecaster.eval.data_quality import (
    build_weather_quality_report,
    write_weather_quality_report,
)
from lmp_forecaster.eval.panel_report import build_panel_summary, write_panel_summary
from lmp_forecaster.models.baselines import (
    BaselineTrainingConfig,
    load_training_config,
    train_single_zone_baselines,
)
from lmp_forecaster.tuning.package import (
    TuningPackageConfig,
    collect_required_command_plan,
    create_tuning_package,
    validate_tuning_package,
    write_tuning_package_manifest,
)
from lmp_forecaster.tuning.promotion import PromotionGate
from lmp_forecaster.tuning.result_import import (
    import_tuning_results,
    write_import_validation_report,
)
from lmp_forecaster.tuning.search_design import (
    discover_latest_calibration_report,
    load_search_design_config,
    recommend_search_strategy,
    write_search_design,
)
from lmp_forecaster.tuning.tuning_runner import (
    TuningRunConfig,
    load_tuning_config,
    log_tuning_tracking,
    planned_tuning_output_paths,
    run_focused_tuning,
    write_tuning_results,
)
from lmp_forecaster.tuning.tuning_runner import (
    load_baseline_metrics as load_tuning_baseline_metrics,
)
from lmp_forecaster.tuning.tuning_runner import (
    load_search_design as load_tuning_search_design,
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


def _resolve_real_pjm_window(
    one_year: bool,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    if start_date and end_date:
        return start_date, end_date
    if one_year:
        return date(2024, 1, 1), date(2024, 12, 31)
    return date(2024, 1, 1), date(2024, 1, 31)


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
            f"- {source.name} | provider={source.provider} | frequency={source.expected_frequency}"
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


@app.command("pull-real-weather")
def pull_real_weather_command(
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
        typer.Option(help="Write normalized weather cache and quality report when set."),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite existing normalized weather cache path when set."),
    ] = False,
) -> None:
    """Pull real Open-Meteo hourly weather for a zone/date window."""
    zone_meta = get_zone(zone)
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    start = parsed_start or date(2024, 1, 1)
    end = parsed_end or date(2024, 12, 31)

    print("[bold]Real weather pull[/bold]")
    print(f"Zone: {zone_meta.zone}")
    print(f"Date window: {start} -> {end}")
    print(f"Coordinates: lat={zone_meta.latitude}, lon={zone_meta.longitude}")

    if not write:
        planned = locate_latest_weather_cache(
            zone=zone_meta.zone,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        if planned is not None:
            print(f"Existing cache candidate: {planned.path}")
            print(f"Existing cache rows: {planned.row_count}")
        else:
            print("No existing full-window weather cache found.")
        print("Dry-run only. Re-run with --write to persist weather cache/report.")
        return

    result = pull_real_weather(
        zone=zone_meta.zone,
        latitude=zone_meta.latitude,
        longitude=zone_meta.longitude,
        start_date=start,
        end_date=end,
        write=True,
        overwrite=overwrite,
    )
    quality_path = result.quality_report_path
    if quality_path is None:
        quality = build_weather_quality_report(result.normalized, zone=zone_meta.zone)
        quality_path = write_weather_quality_report(quality)
    else:
        quality = build_weather_quality_report(result.normalized, zone=zone_meta.zone)

    print(f"Weather cache path: {result.output_path}")
    print(f"Rows: {len(result.normalized)}")
    print(f"Weather quality report: {quality_path}")
    print(f"Data source label: {quality['data_source_label']}")


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
        require_real_sources=not (allow_synthetic_lmp or allow_synthetic_weather),
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

    if cfg.start_date is not None and cfg.end_date is not None:
        discovered_lmp = locate_latest_lmp_cache(
            zone=cfg.zone,
            start_date=cfg.start_date.isoformat(),
            end_date=cfg.end_date.isoformat(),
        )
        print(f"Discovered LMP chunks: {len(discovered_lmp)}")
        if discovered_lmp:
            print(f"First LMP chunk: {discovered_lmp[0]}")
            print(f"Last LMP chunk: {discovered_lmp[-1]}")
            lmp_report = locate_latest_lmp_quality_report(
                zone=cfg.zone,
                start_date=cfg.start_date.isoformat(),
                end_date=cfg.end_date.isoformat(),
            )
            if lmp_report is not None:
                print(f"LMP quality report: {lmp_report}")

        discovered_weather = locate_latest_weather_cache(
            zone=cfg.zone,
            start_date=cfg.start_date.isoformat(),
            end_date=cfg.end_date.isoformat(),
        )
        if discovered_weather is not None:
            print(f"Discovered weather cache: {discovered_weather.path}")
            print(f"Discovered weather rows: {discovered_weather.row_count}")
            weather_report = locate_latest_weather_quality_report(
                zone=cfg.zone,
                start_date=cfg.start_date.isoformat(),
                end_date=cfg.end_date.isoformat(),
            )
            if weather_report is not None:
                print(f"Weather quality report: {weather_report}")

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
    print(f"Source labels: {sorted(panel['source_label'].astype(str).unique().tolist())}")

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
    enable_tracking: Annotated[
        bool,
        typer.Option(help="Enable MLflow tracking for this training run."),
    ] = False,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="Optional MLflow tracking URI override."),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow experiment name override."),
    ] = None,
    run_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow run name override."),
    ] = None,
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
            "tracking_enabled": enable_tracking,
            "tracking_uri": tracking_uri,
            "tracking_experiment_name": experiment_name,
            "tracking_run_name": run_name,
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
        f"val_size={cfg.val_size}, test_size={cfg.test_size}, max_steps_smoke={cfg.max_steps_smoke}"
    )
    print(f"skip_tft={cfg.skip_tft}, skip_deepar={cfg.skip_deepar}")
    print(
        "tracking_enabled="
        f"{cfg.tracking_enabled}, "
        f"tracking_uri={cfg.tracking_uri or '(config default)'}, "
        f"experiment_name={cfg.tracking_experiment_name or '(config default)'}"
    )
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

    print(f"Data source label: {result['data_source_label']}")
    print(f"Accelerator: {result['accelerator']} ({result['device_name']})")
    print(f"Forecast outputs: {result['forecasts']}")
    print(f"Metrics JSON: {result['metrics_json']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    print(f"Training report: {result['training_report_json']}")
    if "tracking" in result:
        print(f"Tracking status: {result['tracking']}")


@app.command("plan-rolling-backtest")
def plan_rolling_backtest_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    panel_path: Annotated[
        Path,
        typer.Option(help="Panel parquet path for fold planning."),
    ] = Path("data/processed/panel/single_zone/AEP_panel.parquet"),
    folds: Annotated[int, typer.Option(min=1, help="Number of folds.")] = 3,
    horizon_hours: Annotated[int, typer.Option(min=1, help="Test horizon per fold in hours.")] = 24,
    min_train_hours: Annotated[
        int,
        typer.Option(min=24, help="Minimum train history per fold in hours."),
    ] = 2160,
    window_mode: Annotated[
        str,
        typer.Option(help="Train window mode: expanding or rolling."),
    ] = "expanding",
    write: Annotated[
        bool,
        typer.Option(help="Write backtest plan reports under data/cache/reports when set."),
    ] = False,
) -> None:
    """Plan rolling-origin backtest folds without running expensive model training."""
    paths = get_project_paths()
    resolved_panel = panel_path if panel_path.is_absolute() else (paths.root / panel_path)

    print("[bold]Rolling-origin backtest planning[/bold]")
    print(f"Zone: {zone.upper()}")
    print(f"Panel path: {resolved_panel}")
    print(f"folds={folds}, horizon_hours={horizon_hours}, min_train_hours={min_train_hours}")
    print(f"window_mode={window_mode}")

    if not resolved_panel.exists():
        raise typer.BadParameter(f"Panel path does not exist: {resolved_panel}")

    panel = pd.read_parquet(resolved_panel)
    cfg = BacktestConfig(
        zone=zone.upper(),
        horizon_hours=horizon_hours,
        folds=folds,
        min_train_hours=min_train_hours,
        window_mode=window_mode,
    )
    fold_plan = make_rolling_origin_folds(panel, cfg)
    validate_backtest_folds(fold_plan)

    reports_root = paths.root / "data" / "cache" / "reports"
    planned_output = reports_root / f"backtest_plan_{zone.upper()}_<timestamp>.json"
    summary = summarize_backtest_plan(panel, cfg, fold_plan, planned_output)

    print(f"Detected panel rows: {summary['panel_row_count']}")
    print(f"Panel ds range: {summary['panel_min_ds']} -> {summary['panel_max_ds']}")
    print("Leakage validation result: passed")
    print("Overlap validation result: passed")
    print(f"Intended output path: {planned_output}")

    for fold in summary["folds"]:
        row: dict[str, object] = fold if isinstance(fold, dict) else {}
        print(
            "Fold "
            f"{row.get('fold_id')}: "
            f"train[{row.get('train_start')} -> {row.get('train_end')}] "
            f"origin={row.get('origin')} "
            f"test[{row.get('test_start')} -> {row.get('test_end')}]"
        )

    if not write:
        print("Dry-run only. Re-run with --write to persist backtest plan reports.")
        return

    json_path, md_path = write_backtest_plan(summary, output_dir=reports_root)
    print(f"Backtest plan JSON: {json_path}")
    print(f"Backtest plan Markdown: {md_path}")


@app.command("run-rolling-backtest")
def run_rolling_backtest_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    panel_path: Annotated[
        Path,
        typer.Option(help="Panel parquet path for rolling-origin backtest."),
    ] = Path("data/processed/panel/single_zone/AEP_panel.parquet"),
    folds: Annotated[int, typer.Option(min=1, help="Number of folds.")] = 3,
    horizon_hours: Annotated[int, typer.Option(min=1, help="Test horizon per fold in hours.")] = 24,
    min_train_hours: Annotated[
        int,
        typer.Option(min=1, help="Minimum train history per fold in hours."),
    ] = 2160,
    window_mode: Annotated[
        str,
        typer.Option(help="Train window mode: expanding or rolling."),
    ] = "expanding",
    write: Annotated[
        bool,
        typer.Option(help="Train/evaluate folds and write outputs when set."),
    ] = False,
    skip_tft: Annotated[bool, typer.Option(help="Skip TFT model backtest.")] = False,
    skip_deepar: Annotated[bool, typer.Option(help="Skip DeepAR model backtest.")] = False,
    enable_tracking: Annotated[
        bool,
        typer.Option(help="Enable MLflow tracking for this run."),
    ] = False,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="Optional MLflow tracking URI override."),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow experiment name override."),
    ] = None,
    run_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow run name override."),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(help="Override max training steps per fold/model."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional override for backtest output root directory."),
    ] = None,
) -> None:
    """Execute rolling-origin backtest over real AEP panel with TFT/DeepAR baselines."""
    default_cfg = BacktestRunConfig()
    cfg = BacktestRunConfig(
        zone=zone.upper(),
        panel_path=panel_path,
        folds=folds,
        horizon_hours=horizon_hours,
        min_train_hours=min_train_hours,
        input_size_hours=default_cfg.input_size_hours,
        window_mode=window_mode,
        models=default_cfg.models,
        max_steps=max_steps or default_cfg.max_steps,
        seed=default_cfg.seed,
        skip_tft=skip_tft,
        skip_deepar=skip_deepar,
        require_real_data=True,
        enable_tracking=enable_tracking,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        output_root=output_dir or default_cfg.output_root,
        report_root=default_cfg.report_root,
        artifact_root=default_cfg.artifact_root,
    )

    try:
        cfg.validate()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    print("[bold]Rolling-origin backtest execution[/bold]")
    print(f"Zone: {cfg.zone}")
    print(f"Panel path: {panel_path}")
    print(
        f"folds={cfg.folds}, horizon_hours={cfg.horizon_hours}, "
        f"min_train_hours={cfg.min_train_hours}, window_mode={cfg.window_mode}"
    )
    print(f"models_enabled={cfg.enabled_models}")
    print(f"max_steps={cfg.max_steps}")
    print(
        "tracking_enabled="
        f"{cfg.enable_tracking}, "
        f"tracking_uri={cfg.tracking_uri or '(default file:./mlruns)'}, "
        f"experiment_name={cfg.experiment_name or '(default lmp_rolling_backtest)'}"
    )

    try:
        panel, source_label = load_backtest_panel(
            cfg.panel_path,
            zone=cfg.zone,
            require_real_data=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        raise typer.Exit(code=2) from exc

    plan_cfg = BacktestConfig(
        zone=cfg.zone,
        horizon_hours=cfg.horizon_hours,
        folds=cfg.folds,
        min_train_hours=cfg.min_train_hours,
        window_mode=cfg.window_mode,
    )
    fold_plan = make_rolling_origin_folds(panel, plan_cfg)
    validate_backtest_folds(fold_plan)

    print(f"Data source label: {source_label}")
    print(f"Detected panel rows: {len(panel)}")
    print(f"Panel ds range: {panel['ds'].min()} -> {panel['ds'].max()}")
    print("Leakage validation result: passed")
    print("Overlap validation result: passed")
    print("Planned folds:")
    for fold in fold_plan:
        print(
            "- fold_id="
            f"{fold.fold_id} "
            f"train[{fold.train_start} -> {fold.train_end}] "
            f"test[{fold.test_start} -> {fold.test_end}] "
            f"train_rows={fold.train_rows} test_rows={fold.test_rows}"
        )

    expected = planned_output_paths(cfg)
    print("Expected output paths:")
    for key, value in expected.items():
        print(f"- {key}: {value}")

    if not write:
        print("Dry-run only. Re-run with --write to execute fold training/evaluation.")
        return

    result = run_rolling_backtest(cfg)
    if result.accelerator == "gpu":
        print(f"Accelerator selected: GPU ({result.device_name})")
    else:
        print(f"Accelerator selected: CPU fallback ({result.device_name})")

    paths = write_backtest_results(result)
    tracking_status = log_backtest_tracking(result, paths)
    result.tracking = tracking_status

    print(f"Forecast outputs: {paths['forecasts']}")
    print(f"Fold metrics CSV: {paths['fold_metrics']}")
    print(f"Aggregate metrics CSV: {paths['aggregate_metrics']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Summary Markdown: {paths['summary_markdown']}")
    print(f"Manifest: {paths['manifest']}")
    print(f"Tracking status: {tracking_status}")


@app.command("analyze-calibration")
def analyze_calibration_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    forecasts_path: Annotated[
        Path | None,
        typer.Option(help="Optional rolling backtest forecasts parquet path."),
    ] = None,
    fold_metrics_path: Annotated[
        Path | None,
        typer.Option(help="Optional rolling fold metrics CSV path."),
    ] = None,
    aggregate_metrics_path: Annotated[
        Path | None,
        typer.Option(help="Optional rolling aggregate metrics CSV path."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Write calibration diagnostics reports when set."),
    ] = False,
) -> None:
    """Analyze interval calibration diagnostics from rolling-origin forecast outputs."""
    cfg = load_calibration_config()
    zone_u = zone.upper()

    discovered = discover_latest_backtest_outputs(zone_u)
    resolved_forecasts = forecasts_path or discovered.get("forecasts")
    resolved_fold_metrics = fold_metrics_path or discovered.get("fold_metrics")
    resolved_aggregate_metrics = aggregate_metrics_path or discovered.get("aggregate_metrics")

    if resolved_forecasts is None:
        print(
            "No rolling backtest forecasts found. Run run-rolling-backtest --write before "
            "analyze-calibration."
        )
        raise typer.Exit(code=2)

    try:
        forecasts = pd.read_parquet(resolved_forecasts)
    except Exception as exc:
        print(f"Failed to read forecasts parquet: {resolved_forecasts}")
        print(str(exc))
        raise typer.Exit(code=2) from exc

    summary = summarize_calibration_diagnostics(forecasts, cfg=cfg, zone=zone_u)

    print("[bold]Calibration diagnostics[/bold]")
    print(f"Zone: {zone_u}")
    print(f"Forecasts path: {resolved_forecasts}")
    if resolved_fold_metrics is not None:
        print(f"Fold metrics path: {resolved_fold_metrics}")
    if resolved_aggregate_metrics is not None:
        print(f"Aggregate metrics path: {resolved_aggregate_metrics}")
    print(f"Rows analyzed: {summary['rows_analyzed']}")
    print(f"Target coverage: {summary['target_coverage']:.2f} ± {summary['tolerance']:.2f}")

    for row in summary.get("model_summary", []):
        model = row.get("model")
        coverage = float(row.get("coverage_80", 0.0))
        width = float(row.get("interval_width_mean", 0.0))
        crossing = float(row.get("crossing_rate", 0.0))
        median_bias = float(row.get("median_bias_mean", 0.0))
        status = row.get("calibration_status")
        note = row.get("classification_note")
        print(
            f"- {model}: coverage_80={coverage:.4f}, interval_width_mean={width:.4f}, "
            f"crossing_rate={crossing:.4f}, median_bias={median_bias:.4f}, "
            f"status={status}, note={note}"
        )

    if not write:
        print("Dry-run only. Re-run with --write to persist calibration diagnostics reports.")
        return

    json_path, md_path = write_calibration_report(summary, zone=zone_u)
    print(f"Calibration diagnostics JSON: {json_path}")
    print(f"Calibration diagnostics Markdown: {md_path}")


@app.command("design-focused-search")
def design_focused_search_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    diagnostics_path: Annotated[
        Path | None,
        typer.Option(help="Optional calibration diagnostics JSON path."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Write focused search design reports when set."),
    ] = False,
) -> None:
    """Design focused TFT/DeepAR search spaces from calibration diagnostics."""
    zone_u = zone.upper()
    search_cfg = load_search_design_config()

    resolved_diag = diagnostics_path or discover_latest_calibration_report(zone_u)
    diagnostics: dict[str, object]

    if resolved_diag is not None:
        try:
            import json

            loaded = json.loads(Path(resolved_diag).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("Diagnostics JSON must parse to object mapping")
            diagnostics = loaded
        except Exception as exc:
            print(f"Failed to load diagnostics JSON: {resolved_diag}")
            print(str(exc))
            raise typer.Exit(code=2) from exc
    else:
        discovered = discover_latest_backtest_outputs(zone_u)
        forecasts_path = discovered.get("forecasts")
        if forecasts_path is None:
            print(
                "No calibration diagnostics or rolling backtest forecasts found. Run "
                "run-rolling-backtest --write and analyze-calibration first."
            )
            raise typer.Exit(code=2)
        try:
            forecasts = pd.read_parquet(forecasts_path)
        except Exception as exc:
            print(f"Failed to read forecasts parquet: {forecasts_path}")
            print(str(exc))
            raise typer.Exit(code=2) from exc
        diagnostics = summarize_calibration_diagnostics(
            forecasts,
            cfg=load_calibration_config(),
            zone=zone_u,
        )
        resolved_diag = forecasts_path

    strategy = recommend_search_strategy(diagnostics, search_cfg)

    print("[bold]Focused search design[/bold]")
    print(f"Zone: {zone_u}")
    print(f"Diagnostics source: {resolved_diag}")
    print(
        "Strategy: "
        f"first_pass_trials={strategy['strategy']['first_pass_trials']}, "
        f"second_pass_trials={strategy['strategy']['second_pass_trials']}, "
        f"primary_metric={strategy['strategy']['primary_metric']}, "
        f"secondary_metric={strategy['strategy']['secondary_metric']}"
    )

    for model in ["TFT", "DeepAR"]:
        space = strategy.get("spaces", {}).get(model, {})
        print(f"- {model}: {space.get('objective_focus', '')}")
        print(f"  recommended_first_search_size={space.get('recommended_first_search_size')}")

    if not write:
        print("Dry-run only. Re-run with --write to persist focused search design reports.")
        return

    json_path, md_path = write_search_design(strategy, zone=zone_u)
    print(f"Focused search design JSON: {json_path}")
    print(f"Focused search design Markdown: {md_path}")


@app.command("run-focused-tuning")
def run_focused_tuning_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    panel_path: Annotated[
        Path,
        typer.Option(help="Panel parquet path for focused tuning trials."),
    ] = Path("data/processed/panel/single_zone/AEP_panel.parquet"),
    search_design_path: Annotated[
        Path | None,
        typer.Option(help="Optional focused search design JSON path."),
    ] = None,
    baseline_metrics_path: Annotated[
        Path | None,
        typer.Option(help="Optional rolling backtest aggregate metrics CSV path."),
    ] = None,
    resource_profile: Annotated[
        str,
        typer.Option(help="Resource profile name: local_safe or default."),
    ] = "default",
    models: Annotated[
        str,
        typer.Option(help="Comma-separated models (TFT,DeepAR)."),
    ] = "TFT,DeepAR",
    max_trials: Annotated[
        int | None,
        typer.Option(min=1, help="Optional total trial budget override."),
    ] = None,
    folds: Annotated[
        int | None,
        typer.Option(min=1, help="Optional rolling folds per trial override."),
    ] = None,
    horizon_hours: Annotated[int, typer.Option(min=1, help="Forecast horizon per fold.")] = 24,
    max_steps_cap: Annotated[
        int | None,
        typer.Option(min=1, help="Optional cap for per-trial training steps."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(min=1, help="Optional batch size override."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Execute tuning trials and write reports when set."),
    ] = False,
    skip_tft: Annotated[bool, typer.Option(help="Skip TFT model trials.")] = False,
    skip_deepar: Annotated[bool, typer.Option(help="Skip DeepAR model trials.")] = False,
    cleanup_after_trial: Annotated[
        bool,
        typer.Option(help="Cleanup trial temporary artifacts and release memory after each trial."),
    ] = True,
    cpu_only: Annotated[
        bool,
        typer.Option(help="Force CPU-only execution for stability."),
    ] = False,
    allow_heavy_run: Annotated[
        bool,
        typer.Option(help="Explicitly allow budgets above local_safe limits."),
    ] = False,
    enable_tracking: Annotated[
        bool,
        typer.Option(help="Enable MLflow tracking for this tuning run."),
    ] = False,
    no_mlflow: Annotated[
        bool,
        typer.Option("--no-mlflow", help="Disable MLflow for this run."),
    ] = False,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="Optional MLflow tracking URI override."),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow experiment name override."),
    ] = None,
    run_name: Annotated[
        str | None,
        typer.Option(help="Optional MLflow run name override."),
    ] = None,
    timeout_minutes: Annotated[
        int | None,
        typer.Option(help="Optional wall-clock timeout (minutes) for trial loop."),
    ] = None,
    dry_run_plan_only: Annotated[
        bool,
        typer.Option(help="Force plan-only behavior even if --write is provided."),
    ] = False,
) -> None:
    """Run a bounded focused tuning pass with resource-safe controls."""
    zone_u = zone.upper()
    default_cfg = load_tuning_config()

    parsed_models = [m.strip() for m in models.split(",") if m.strip()]
    if not parsed_models:
        raise typer.BadParameter("--models must include at least one model name")

    profile_key = resource_profile.strip().lower()
    if profile_key in {"", "default"}:
        selected_profile_key = "local_safe"
    elif profile_key in {"local_safe", "local-safe"}:
        selected_profile_key = "local_safe"
    elif profile_key in default_cfg.resource_profiles:
        selected_profile_key = profile_key
    else:
        options = ", ".join(sorted(default_cfg.resource_profiles))
        raise typer.BadParameter(
            f"Unknown --resource-profile '{resource_profile}'. Expected one of: {options}"
        )

    selected_profile = default_cfg.resource_profiles[selected_profile_key]

    if selected_profile_key.startswith("cloud_") and write and not allow_heavy_run:
        print(
            "Cloud resource profiles are intended for stronger hardware. "
            "Refusing local execution without --allow-heavy-run."
        )
        raise typer.Exit(code=2)

    applied_max_trials = (
        max_trials if max_trials is not None else int(selected_profile.max_trials)
    )
    applied_folds = folds if folds is not None else int(selected_profile.folds)
    applied_max_steps_cap = (
        max_steps_cap if max_steps_cap is not None else int(selected_profile.max_steps_cap)
    )
    applied_batch_size = batch_size if batch_size is not None else int(selected_profile.batch_size)
    applied_num_workers = int(selected_profile.num_workers)

    tracking_enabled = (
        selected_profile.enable_mlflow_by_default or enable_tracking
    ) and not no_mlflow

    cfg = TuningRunConfig(
        zone=zone_u,
        panel_path=panel_path,
        search_design_path=search_design_path,
        baseline_metrics_path=baseline_metrics_path,
        models=tuple(parsed_models),
        max_trials=applied_max_trials,
        folds=applied_folds,
        horizon_hours=horizon_hours,
        min_train_hours=default_cfg.min_train_hours,
        window_mode=default_cfg.window_mode,
        primary_metric=default_cfg.primary_metric,
        target_coverage=default_cfg.target_coverage,
        coverage_min=default_cfg.coverage_min,
        coverage_max=default_cfg.coverage_max,
        mae_regression_limit=default_cfg.mae_regression_limit,
        allow_deepar_if_interval_collapse=default_cfg.allow_deepar_if_interval_collapse,
        skip_tft=skip_tft,
        skip_deepar=skip_deepar,
        seed=default_cfg.seed,
        max_steps_cap=applied_max_steps_cap,
        batch_size=applied_batch_size,
        num_workers=applied_num_workers,
        cpu_only=cpu_only,
        cleanup_after_trial=cleanup_after_trial,
        resource_profile=selected_profile_key,
        allow_heavy_run=allow_heavy_run,
        enable_tracking=tracking_enabled,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        timeout_minutes=timeout_minutes,
        output_root=default_cfg.output_root,
        report_root=default_cfg.report_root,
        artifact_root=default_cfg.artifact_root,
        profile=selected_profile,
        resource_profiles=default_cfg.resource_profiles,
    )

    try:
        cfg.validate()
    except ValueError as exc:
        if cfg.resource_profile == "local_safe" and "refused heavy run" in str(exc):
            print(str(exc))
            print("Use --allow-heavy-run only when you intentionally accept higher local risk.")
            raise typer.Exit(code=2) from exc
        raise typer.BadParameter(str(exc)) from exc

    paths = get_project_paths()
    resolved_panel = panel_path if panel_path.is_absolute() else (paths.root / panel_path)
    if not resolved_panel.exists():
        print("Real AEP panel is missing. Run build-single-zone-panel before focused tuning.")
        raise typer.Exit(code=2)

    try:
        _, resolved_search_design = load_tuning_search_design(zone_u, search_design_path)
    except FileNotFoundError as exc:
        print(str(exc))
        raise typer.Exit(code=2) from exc

    try:
        _, resolved_baseline = load_tuning_baseline_metrics(zone_u, baseline_metrics_path)
    except FileNotFoundError as exc:
        print(str(exc))
        raise typer.Exit(code=2) from exc

    expected = planned_tuning_output_paths(cfg)

    estimated_trial_fold_units = cfg.max_trials * cfg.folds

    print("[bold]Focused tuning execution[/bold]")
    print(f"Zone: {zone_u}")
    print(f"Panel path: {resolved_panel}")
    print(f"Search design path: {resolved_search_design}")
    print(f"Baseline metrics path: {resolved_baseline}")
    print(f"resource_profile={cfg.resource_profile} ({cfg.profile.name})")
    print(f"models_enabled={cfg.enabled_models}")
    print(
        f"max_trials={cfg.max_trials}, folds={cfg.folds}, horizon_hours={cfg.horizon_hours}, "
        f"max_steps_cap={cfg.max_steps_cap}, batch_size={cfg.effective_batch_size}, "
        f"num_workers={cfg.num_workers}, objective={cfg.primary_metric}"
    )
    print(
        "estimated_workload="
        f"model_count({len(cfg.enabled_models)}) x "
        f"trial_fold_units({estimated_trial_fold_units})"
    )
    print(
        f"promotion_gate: coverage=[{cfg.coverage_min:.2f},{cfg.coverage_max:.2f}], "
        f"mae_regression_limit={cfg.mae_regression_limit:.2f}, "
        f"allow_deepar_if_interval_collapse={cfg.allow_deepar_if_interval_collapse}"
    )
    print(
        "tracking_enabled="
        f"{cfg.enable_tracking}, "
        f"tracking_uri={cfg.tracking_uri or '(default file:./mlruns)'}, "
        f"experiment_name={cfg.experiment_name or '(default lmp_focused_tuning)'}"
    )

    if cfg.resource_profile == "local_safe":
        print(
            "local_safe warning: full search remains deferred on this machine; "
            "run only bounded smoke scope unless --allow-heavy-run is explicitly set."
        )

    print("Expected output paths:")
    for key, value in expected.items():
        print(f"- {key}: {value}")

    if not write or dry_run_plan_only:
        print("Dry-run only. Re-run with --write to execute focused tuning trials.")
        return

    summary = run_focused_tuning(
        cfg,
        search_design_path=search_design_path,
        baseline_metrics_path=baseline_metrics_path,
    )
    output_paths = write_tuning_results(summary)
    tracking_status = log_tuning_tracking(summary, output_paths)

    print(f"Trial metrics CSV: {output_paths['trials']}")
    print(f"Ranked candidates CSV: {output_paths['ranked']}")
    print(f"Summary JSON: {output_paths['summary_json']}")
    print(f"Summary Markdown: {output_paths['summary_markdown']}")
    print(f"Manifest: {output_paths['manifest']}")
    print(f"Promotion summary: {summary.promotion_summary}")
    print(f"Tracking status: {tracking_status}")


@app.command("export-tuning-package")
def export_tuning_package_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    resource_profile: Annotated[
        str,
        typer.Option(help="Resource profile: local_safe, cloud_16gb, cloud_24gb."),
    ] = "cloud_16gb",
    models: Annotated[
        str,
        typer.Option(help="Comma-separated models list, e.g. TFT,DeepAR."),
    ] = "TFT,DeepAR",
    max_trials: Annotated[
        int | None,
        typer.Option(min=1, help="Optional trial count override."),
    ] = None,
    folds: Annotated[
        int | None,
        typer.Option(min=1, help="Optional fold count override."),
    ] = None,
    max_steps_cap: Annotated[
        int | None,
        typer.Option(min=1, help="Optional max steps cap override."),
    ] = None,
    panel_path: Annotated[
        Path,
        typer.Option(help="Expected panel path (relative preferred)."),
    ] = Path("data/processed/panel/single_zone/AEP_panel.parquet"),
    baseline_metrics_path: Annotated[
        Path,
        typer.Option(help="Expected baseline metrics CSV path."),
    ] = Path("data/cache/backtests/aep_rolling_backtest_aggregate_metrics_latest.csv"),
    search_design_path: Annotated[
        Path,
        typer.Option(help="Expected focused search design JSON path."),
    ] = Path("data/cache/reports/aep_focused_search_design_latest.json"),
    write: Annotated[
        bool,
        typer.Option(help="Write package manifest under data/cache/tuning_packages/."),
    ] = False,
) -> None:
    """Plan or write a portable package manifest for external focused tuning."""
    zone_u = zone.upper()
    base_cfg = load_tuning_config()

    profile_key = resource_profile.strip().lower()
    if profile_key not in base_cfg.resource_profiles:
        options = ", ".join(sorted(base_cfg.resource_profiles))
        raise typer.BadParameter(
            f"Unknown --resource-profile '{resource_profile}'. Expected one of: {options}"
        )

    profile = base_cfg.resource_profiles[profile_key]
    model_tuple = tuple(item.strip() for item in models.split(",") if item.strip())
    if not model_tuple:
        raise typer.BadParameter("--models must include at least one model")

    package_cfg = TuningPackageConfig(
        zone=zone_u,
        resource_profile=profile_key,
        models=model_tuple,
        max_trials=max_trials if max_trials is not None else profile.max_trials,
        folds=folds if folds is not None else profile.folds,
        max_steps_cap=max_steps_cap if max_steps_cap is not None else profile.max_steps_cap,
        panel_path=panel_path,
        baseline_metrics_path=baseline_metrics_path,
        search_design_path=search_design_path,
    )

    manifest = create_tuning_package(package_cfg)
    issues = validate_tuning_package(manifest)
    if issues:
        for issue in issues:
            print(f"Validation issue: {issue}")
        raise typer.Exit(code=2)

    print("Portable tuning package plan")
    print(f"zone={manifest.zone}")
    print(f"resource_profile={manifest.resource_profile}")
    print(f"models={','.join(package_cfg.models)}")
    print(
        f"max_trials={package_cfg.max_trials}, folds={package_cfg.folds}, "
        f"max_steps_cap={package_cfg.max_steps_cap}"
    )
    print(f"repo_commit={manifest.repo_commit}")
    print(f"repo_branch={manifest.repo_branch}")
    print("Cloud command plan:")
    for cmd in collect_required_command_plan(package_cfg):
        print(f"- {cmd}")
    print(f"External run command: {manifest.tuning_command}")
    print(f"Import command template: {manifest.import_command_template}")

    if not write:
        print("Dry-run only. Re-run with --write to persist package manifest.")
        return

    output_paths = write_tuning_package_manifest(manifest)
    print(f"Package manifest JSON: {output_paths['json']}")
    print(f"Package manifest Markdown: {output_paths['markdown']}")


@app.command("import-tuning-results")
def import_tuning_results_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    ranked_candidates_path: Annotated[
        Path | None,
        typer.Option(help="Path to ranked candidates CSV generated externally."),
    ] = None,
    summary_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to tuning summary JSON."),
    ] = None,
    baseline_metrics_path: Annotated[
        Path | None,
        typer.Option(help="Optional local baseline metrics CSV path."),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(help="Write import validation report under data/cache/reports/."),
    ] = False,
) -> None:
    """Import external ranked results and recompute promotion decisions locally."""
    zone_u = zone.upper()

    if ranked_candidates_path is None or not ranked_candidates_path.exists():
        raise typer.BadParameter(
            "Ranked candidates file not found. Provide --ranked-candidates-path."
        )

    if summary_path is not None and not summary_path.exists():
        raise typer.BadParameter(f"Summary file not found: {summary_path}")

    try:
        base_cfg = load_tuning_config()
        gate = PromotionGate(
            coverage_min=base_cfg.coverage_min,
            coverage_max=base_cfg.coverage_max,
            mae_regression_limit=base_cfg.mae_regression_limit,
            require_no_quantile_crossing=True,
            require_no_interval_collapse=not base_cfg.allow_deepar_if_interval_collapse,
        )
    except Exception:
        gate = PromotionGate(
            coverage_min=0.70,
            coverage_max=0.90,
            mae_regression_limit=0.15,
            require_no_quantile_crossing=True,
            require_no_interval_collapse=True,
        )

    baseline_path = baseline_metrics_path
    if baseline_path is None:
        try:
            _, resolved = load_tuning_baseline_metrics(zone_u, None)
            baseline_path = resolved
        except FileNotFoundError as exc:
            raise typer.BadParameter(
                "Could not auto-discover baseline metrics. Provide --baseline-metrics-path."
            ) from exc

    result = import_tuning_results(
        zone=zone_u,
        ranked_candidates_path=ranked_candidates_path,
        summary_path=summary_path,
        baseline_metrics_path=baseline_path,
        gate=gate,
    )

    print("Imported tuning result validation")
    print(f"zone={result.zone}")
    print(f"ranked_candidates_path={result.ranked_candidates_path}")
    print(f"summary_path={result.summary_path or '(not provided)'}")
    print(f"baseline_metrics_path={result.baseline_metrics_path}")
    print(f"recomputed_overall_status={result.recomputed_overall_status}")
    print(f"accepted_count={result.accepted_count}")
    print(f"rejection_count={result.rejection_count}")
    print(f"mismatch_count={result.mismatch_count}")
    for item in result.imported_candidates[:10]:
        print(
            "- "
            f"{item.model}/{item.trial_id}: "
            f"imported={item.imported_promotion_status or '(missing)'}; "
            f"recomputed={item.recomputed_promotion_status}; "
            f"match={item.status_match}; "
            f"reason={item.recomputed_rejection_reason or 'n/a'}"
        )

    if not write:
        print("Dry-run only. Re-run with --write to persist import validation report.")
        return

    output_paths = write_import_validation_report(result)
    print(f"Import validation JSON: {output_paths['json']}")
    print(f"Import validation Markdown: {output_paths['markdown']}")


@app.command("inspect-pjm-api")
def inspect_pjm_api_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    row_count: Annotated[int, typer.Option(min=1, max=1000)] = 10,
    require_key: Annotated[
        bool,
        typer.Option(help="Exit non-zero when PJM_API_KEY is missing."),
    ] = False,
) -> None:
    """Probe PJM API endpoint/params and print redacted request diagnostics."""
    cfg = resolve_pjm_api_config()
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    start, end = _resolve_real_pjm_window(False, parsed_start, parsed_end)

    print("[bold]PJM API inspect[/bold]")
    print(f"Endpoint: {cfg.api_base_url.rstrip('/')}/da_hrl_lmps")
    print(f"Zone: {zone.upper()}")

    if not cfg.api_key:
        print("PJM_API_KEY not configured. Add PJM_API_KEY to .env for live API probes.")
        if require_key:
            raise typer.Exit(code=2)
        return

    params = build_day_ahead_lmp_params(
        start=start,
        end=end,
        zone=zone.upper(),
        start_row=1,
        row_count=row_count,
    )
    headers = redact_headers(build_pjm_api_headers(cfg))
    print(f"Headers: {headers}")
    print(f"Params: {params}")

    try:
        payload = fetch_pjm_json_page(config=cfg, endpoint="da_hrl_lmps", params=params)
        normalized = normalize_da_lmp_response(payload, zone=zone.upper(), timezone=cfg.timezone)
    except PjmApiError as exc:
        print(f"PJM API inspect failed: {exc}")
        raise typer.Exit(code=2) from exc

    if isinstance(payload, dict):
        print(f"Response keys: {sorted(payload.keys())}")
    else:
        print("Response keys: <list payload>")
    print(f"Normalized columns: {list(normalized.columns)}")
    print(f"Normalized rows: {len(normalized)}")


@app.command("pull-real-pjm-lmp")
def pull_real_pjm_lmp_command(
    zone: Annotated[str, typer.Option(help="Zone code, defaults to AEP.")] = "AEP",
    start_date: Annotated[
        str | None,
        typer.Option(help="Inclusive start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive end date (YYYY-MM-DD)."),
    ] = None,
    chunk_days: Annotated[int, typer.Option(min=1, max=90)] = 7,
    row_count: Annotated[int, typer.Option(min=100, max=50000)] = 50000,
    write: Annotated[bool, typer.Option(help="Execute pull and write outputs.")] = False,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing local outputs.")] = False,
    one_year: Annotated[bool, typer.Option(help="Use default full-year date window.")] = False,
    allow_partial_completion: Annotated[
        bool,
        typer.Option(
            "--allow-partial-completion",
            help="Continue and record failed chunks instead of failing fast.",
        ),
    ] = False,
) -> None:
    """Backfill real PJM DA LMP for a single zone into ignored local cache paths."""
    parsed_start = _parse_iso_date_option(start_date, option_name="--start-date")
    parsed_end = _parse_iso_date_option(end_date, option_name="--end-date")
    start, end = _resolve_real_pjm_window(one_year, parsed_start, parsed_end)

    api_cfg = resolve_pjm_api_config()
    effective_connections = effective_max_connections_per_minute(api_cfg)
    throttle_seconds = effective_pjm_throttle_seconds(api_cfg)

    backfill_cfg = PjmBackfillConfig(
        zone=zone.upper(),
        start_date=start,
        end_date=end,
        chunk_days=chunk_days,
        row_count=row_count,
        overwrite=overwrite,
        dry_run=not write,
        allow_partial_completion=allow_partial_completion,
    )
    chunks = plan_backfill_chunks(backfill_cfg)

    print("[bold]Real PJM AEP pull[/bold]")
    print(f"Zone: {backfill_cfg.zone}")
    print(f"Date window: {start} -> {end}")
    print(f"Chunks planned: {len(chunks)}")
    print(f"Chunk days: {chunk_days}")
    print(f"row_count: {row_count}")
    print(f"Effective PJM connection limit (/min): {effective_connections}")
    print(f"Throttle seconds between requests: {throttle_seconds:.1f}")

    if not write:
        print("Dry-run only. Re-run with --write to pull and persist real data.")
        return

    if not api_cfg.api_key:
        print(
            "PJM_API_KEY is required for automated Data Miner API ingestion. "
            "Add it to .env or use dry-run mode."
        )
        raise typer.Exit(code=2)

    result = run_da_lmp_backfill(backfill_cfg)
    print(f"Raw outputs: {[str(p) for p in result.raw_paths]}")
    print(f"Normalized outputs: {[str(p) for p in result.normalized_paths]}")
    print(f"Quality report: {result.quality_report_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Rows pulled: {result.combined_rows}")


if __name__ == "__main__":
    app()
