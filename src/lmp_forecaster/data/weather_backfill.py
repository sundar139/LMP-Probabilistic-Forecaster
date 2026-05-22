"""Open-Meteo real-weather pull helpers for zone/date windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.http_client import HttpClientConfig, get_json
from lmp_forecaster.data.openmeteo_weather import (
    OpenMeteoRequestConfig,
    build_historical_weather_request,
    normalize_openmeteo_hourly,
)
from lmp_forecaster.data.validation import validate_weather_frame
from lmp_forecaster.eval.data_quality import (
    build_weather_quality_report,
    write_weather_quality_report,
)


@dataclass(frozen=True)
class WeatherPullResult:
    output_path: Path
    quality_report_path: Path | None
    normalized: pd.DataFrame
    wrote: bool


def weather_cache_output_path(*, zone: str, start: date, end: date) -> Path:
    root = get_project_paths().root
    return (
        root
        / "data"
        / "cache"
        / "weather"
        / "openmeteo"
        / f"openmeteo_{zone.upper()}_{start.isoformat()}_{end.isoformat()}.parquet"
    )


def pull_real_weather(
    *,
    zone: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    write: bool,
    overwrite: bool = False,
    timezone: str = "America/New_York",
    http_config: HttpClientConfig | None = None,
) -> WeatherPullResult:
    cfg = OpenMeteoRequestConfig(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
    )
    output_path = weather_cache_output_path(zone=zone, start=start_date, end=end_date)

    if not write:
        return WeatherPullResult(
            output_path=output_path,
            quality_report_path=None,
            normalized=pd.DataFrame(),
            wrote=False,
        )

    if output_path.exists() and not overwrite:
        cached = pd.read_parquet(output_path)
        validate_weather_frame(cached)
        report = build_weather_quality_report(cached, zone=zone.upper())
        report_path = write_weather_quality_report(report)
        return WeatherPullResult(
            output_path=output_path,
            quality_report_path=report_path,
            normalized=cached,
            wrote=False,
        )

    request_url, request_params = build_historical_weather_request(cfg)
    payload = get_json(
        url=request_url,
        params=request_params,
        config=http_config,
    )
    normalized = normalize_openmeteo_hourly(payload, source="openmeteo_historical_weather")
    validate_weather_frame(normalized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    report = build_weather_quality_report(normalized, zone=zone.upper())
    report_path = write_weather_quality_report(report)

    return WeatherPullResult(
        output_path=output_path,
        quality_report_path=report_path,
        normalized=normalized,
        wrote=True,
    )
