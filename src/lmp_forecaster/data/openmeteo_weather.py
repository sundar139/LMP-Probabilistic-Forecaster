"""Open-Meteo weather adapters for historical and historical-forecast smoke pulls.

References:
- Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.http_client import HttpClientConfig, get_json

OPENMETEO_HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

DEFAULT_HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
)


class OpenMeteoRequestConfig(BaseModel):
    """Typed Open-Meteo request configuration."""

    latitude: float
    longitude: float
    start_date: date
    end_date: date
    timezone: str = "America/New_York"
    hourly: tuple[str, ...] = DEFAULT_HOURLY_VARIABLES


@dataclass(frozen=True)
class OpenMeteoSmokeResult:
    """Result metadata for Open-Meteo smoke pulls."""

    output_path: Path
    normalized: pd.DataFrame
    request_url: str
    request_params: dict[str, str | float]


def _base_params(config: OpenMeteoRequestConfig) -> dict[str, str | float]:
    return {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "timezone": config.timezone,
        "hourly": ",".join(config.hourly),
    }


def build_historical_weather_request(
    config: OpenMeteoRequestConfig,
) -> tuple[str, dict[str, str | float]]:
    """Build historical weather API request."""
    return OPENMETEO_HISTORICAL_WEATHER_URL, _base_params(config)


def build_historical_forecast_request(
    config: OpenMeteoRequestConfig,
) -> tuple[str, dict[str, str | float]]:
    """Build historical forecast request foundation.

    This endpoint evolves over time. Keep request construction centralized so
    leakage-safe forecast snapshot parameters can be tightened in later steps.
    """
    params = _base_params(config)
    params["models"] = "best_match"
    return OPENMETEO_HISTORICAL_FORECAST_URL, params


def normalize_openmeteo_hourly(payload: dict[str, Any], *, source: str) -> pd.DataFrame:
    """Normalize hourly Open-Meteo payload to internal weather schema."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo payload missing 'hourly' object.")

    time_values = hourly.get("time")
    if not isinstance(time_values, list):
        raise ValueError("Open-Meteo payload missing hourly.time array.")

    frame = pd.DataFrame({"ds": pd.to_datetime(time_values, errors="coerce", utc=False)})
    if frame["ds"].isna().any():
        raise ValueError("Open-Meteo payload contains invalid hourly.time values.")

    for variable in DEFAULT_HOURLY_VARIABLES:
        values = hourly.get(variable)
        if isinstance(values, list) and len(values) == len(frame):
            frame[variable] = pd.to_numeric(values, errors="coerce")
        else:
            frame[variable] = pd.NA

    timezone = payload.get("timezone", "UTC")
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    pulled_at = pd.Timestamp(datetime.now(UTC))

    if isinstance(timezone, str):
        if frame["ds"].dt.tz is None:
            try:
                frame["ds"] = frame["ds"].dt.tz_localize(
                    timezone,
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
            except Exception:
                frame["ds"] = frame["ds"].dt.tz_localize(
                    timezone,
                    ambiguous=False,
                    nonexistent="shift_forward",
                )
        else:
            frame["ds"] = frame["ds"].dt.tz_convert(timezone)

        utc_deltas = frame["ds"].dt.tz_convert("UTC").diff().dropna()
        if not utc_deltas.empty and (utc_deltas != pd.Timedelta(hours=1)).any():
            frame["ds"] = pd.Series(
                pd.date_range(start=frame["ds"].iloc[0], periods=len(frame), freq="h"),
                index=frame.index,
            )

    if frame["ds"].isna().any():
        raise ValueError("Open-Meteo payload normalization produced null ds values.")

    frame["latitude"] = float(latitude) if latitude is not None else pd.NA
    frame["longitude"] = float(longitude) if longitude is not None else pd.NA
    frame["timezone"] = timezone
    frame["source"] = source
    frame["pulled_at"] = pulled_at

    ordered = [
        "ds",
        "latitude",
        "longitude",
        "timezone",
        "source",
        *DEFAULT_HOURLY_VARIABLES,
        "pulled_at",
    ]
    return frame[ordered]


def _weather_output_path(config: OpenMeteoRequestConfig, stem: str) -> Path:
    paths = get_project_paths()
    zone_token = f"lat{config.latitude:.2f}_lon{config.longitude:.2f}".replace("-", "m")
    filename = (
        f"{stem}_{config.start_date.isoformat()}_"
        f"{config.end_date.isoformat()}_{zone_token}.parquet"
    )
    return paths.root / "data" / "cache" / "weather" / "openmeteo" / filename


def pull_historical_weather_smoke(
    config: OpenMeteoRequestConfig,
    *,
    write: bool,
    http_config: HttpClientConfig | None = None,
) -> OpenMeteoSmokeResult:
    """Pull historical weather smoke payload and optionally persist local cache."""
    request_url, params = build_historical_weather_request(config)
    output_path = _weather_output_path(config, "historical_weather")

    if not write:
        return OpenMeteoSmokeResult(
            output_path=output_path,
            normalized=pd.DataFrame(),
            request_url=request_url,
            request_params=params,
        )

    payload = get_json(url=request_url, params=params, config=http_config)
    normalized = normalize_openmeteo_hourly(payload, source="openmeteo_historical_weather")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    return OpenMeteoSmokeResult(
        output_path=output_path,
        normalized=normalized,
        request_url=request_url,
        request_params=params,
    )


def pull_historical_forecast_smoke(
    config: OpenMeteoRequestConfig,
    *,
    write: bool,
    http_config: HttpClientConfig | None = None,
) -> OpenMeteoSmokeResult:
    """Pull historical forecast smoke payload and optionally persist local cache."""
    request_url, params = build_historical_forecast_request(config)
    output_path = _weather_output_path(config, "historical_forecast")

    if not write:
        return OpenMeteoSmokeResult(
            output_path=output_path,
            normalized=pd.DataFrame(),
            request_url=request_url,
            request_params=params,
        )

    payload = get_json(url=request_url, params=params, config=http_config)
    normalized = normalize_openmeteo_hourly(payload, source="openmeteo_historical_forecast")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    return OpenMeteoSmokeResult(
        output_path=output_path,
        normalized=normalized,
        request_url=request_url,
        request_params=params,
    )
