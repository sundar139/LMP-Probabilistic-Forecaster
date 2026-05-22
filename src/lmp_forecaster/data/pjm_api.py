"""Real PJM API ingestion utilities for Day-Ahead Hourly LMP.

References:
- Feed definition: https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
- API base: https://api.pjm.com/api/v1/
- Data Miner guide:
  https://www.pjm.com/-/media/DotCom/etools/data-miner-2/data-miner-2-getting-started-guide.pdf
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.http_client import HttpClientConfig, HttpRequestError, request_with_retries


@dataclass(frozen=True)
class PjmApiConfig:
    api_base_url: str = "https://api.pjm.com/api/v1"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    row_count: int = 50000
    max_connections_per_minute: int = 5
    timezone: str = "America/New_York"


class PjmApiError(RuntimeError):
    """Raised for PJM API ingestion/normalization failures."""


def resolve_pjm_api_config() -> PjmApiConfig:
    api_key = os.getenv("PJM_API_KEY") or os.getenv("LMP_PJM_API_KEY")
    api_base = os.getenv("PJM_API_BASE_URL") or os.getenv("LMP_PJM_API_BASE_URL")
    return PjmApiConfig(api_key=api_key or None, api_base_url=api_base or "https://api.pjm.com/api/v1")


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = dict(headers)
    if "Ocp-Apim-Subscription-Key" in redacted:
        redacted["Ocp-Apim-Subscription-Key"] = "***REDACTED***"
    return redacted


def build_pjm_api_headers(config: PjmApiConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["Ocp-Apim-Subscription-Key"] = config.api_key
    return headers


def _to_ept_range(start: date, end: date) -> str:
    left = f"{start.month:02d}/{start.day:02d}/{start.year} 00:00"
    right = f"{end.month:02d}/{end.day:02d}/{end.year} 23:59"
    return f"{left}to{right}"


def build_day_ahead_lmp_params(
    *,
    start: date,
    end: date,
    zone: str,
    start_row: int,
    row_count: int,
) -> dict[str, str | int]:
    return {
        "startRow": start_row,
        "rowCount": row_count,
        "datetime_beginning_ept": _to_ept_range(start, end),
        "fields": (
            "datetime_beginning_utc,datetime_beginning_ept,pnode_id,pnode_name,pnode_type,"
            "type,zone,system_energy_price_da,total_lmp_da,congestion_price_da,"
            "marginal_loss_price_da"
        ),
        # best-effort server-side filter; we still filter defensively client-side
        "pnode_name": zone.upper(),
    }


def fetch_pjm_json_page(
    *,
    config: PjmApiConfig,
    endpoint: str,
    params: dict[str, str | int],
) -> dict[str, Any] | list[Any]:
    if not config.api_key:
        raise PjmApiError(
            "PJM_API_KEY is required for automated Data Miner API ingestion. "
            "Add it to .env or use dry-run mode."
        )

    url = f"{config.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    http_cfg = HttpClientConfig(timeout_seconds=config.timeout_seconds)
    try:
        response = request_with_retries(
            method="GET",
            url=url,
            params=params,
            headers=build_pjm_api_headers(config),
            config=http_cfg,
        )
    except HttpRequestError as exc:
        raise PjmApiError(
            f"PJM API request failed ({exc.status_code}) at {url}. "
            f"Response preview: {(exc.response_text or '')[:300]}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise PjmApiError(
            f"PJM API returned non-JSON response at {url}. "
            "Verify endpoint, key, and query parameter format."
        ) from exc

    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return payload
    raise PjmApiError("PJM API JSON payload must be an object or list.")


def _payload_items(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        return []

    for k in ("items", "data", "results"):
        v = payload.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def fetch_pjm_paginated(
    *,
    config: PjmApiConfig,
    endpoint: str,
    base_params: dict[str, str | int],
) -> list[dict[str, Any]]:
    start_row = int(base_params.get("startRow", 1))
    row_count = int(base_params.get("rowCount", config.row_count))
    throttle_seconds = 60.0 / max(1, config.max_connections_per_minute)

    out: list[dict[str, Any]] = []
    while True:
        params = dict(base_params)
        params["startRow"] = start_row
        params["rowCount"] = row_count
        payload = fetch_pjm_json_page(config=config, endpoint=endpoint, params=params)
        rows = _payload_items(payload)
        out.extend(rows)

        if len(rows) < row_count:
            break

        start_row += row_count
        time.sleep(throttle_seconds)

    return out


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _parse_ds(value: Any, timezone: str) -> pd.Timestamp:
    if value is None:
        raise PjmApiError("Missing timestamp value in PJM row.")

    text = str(value).strip().replace("EPT", "").replace("EST", "").replace("EDT", "").strip()
    ts = pd.to_datetime(text, errors="coerce", utc=False)
    if pd.isna(ts):
        raise PjmApiError(f"Could not parse PJM timestamp: {value}")

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return stamp.tz_convert(timezone)


def normalize_da_lmp_response(
    payload: dict[str, Any] | list[Any],
    *,
    zone: str,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    rows = _payload_items(payload)
    if not rows:
        return pd.DataFrame(
            columns=[
                "unique_id",
                "ds",
                "y",
                "pnode_name",
                "pnode_type",
                "source",
                "pulled_at",
            ]
        )

    frame = pd.DataFrame.from_records(rows)
    columns = list(frame.columns)

    ds_col = _find_column(columns, ["datetime_beginning_ept", "datetime_beginning_utc", "ds"])
    y_col = _find_column(
        columns,
        ["total_lmp_da", "lmp", "da_lmp", "price", "system_energy_price_da"],
    )
    pnode_col = _find_column(columns, ["pnode_name", "zone", "location"])
    pnode_type_col = _find_column(columns, ["pnode_type", "type"]) or "type"

    if ds_col is None:
        raise PjmApiError("PJM payload missing timestamp column.")
    if y_col is None:
        raise PjmApiError(
            "PJM payload missing LMP/price column (expected total_lmp_da/lmp/da_lmp/price)."
        )

    out = pd.DataFrame(
        {
            "unique_id": zone.upper(),
            "ds": frame[ds_col].map(lambda v: _parse_ds(v, timezone)),
            "y": pd.to_numeric(frame[y_col], errors="coerce"),
            "pnode_name": frame[pnode_col].astype(str) if pnode_col else zone.upper(),
            "pnode_type": (
                frame[pnode_type_col].astype(str).str.upper()
                if pnode_type_col in frame.columns
                else "UNKNOWN"
            ),
            "source": "pjm_api_da_hrl_lmps",
            "pulled_at": pd.Timestamp(datetime.now(UTC)),
        }
    )

    # defensive AEP filter (server filter can fail silently)
    zone_upper = zone.upper()
    mask = (
        out["pnode_name"].astype(str).str.upper().eq(zone_upper)
        | frame.get("zone", pd.Series([None] * len(frame))).astype(str).str.upper().eq(zone_upper)
    )
    out = out[mask].copy()

    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return out


def fetch_da_lmp_for_zone(
    *,
    config: PjmApiConfig,
    zone: str,
    start: date,
    end: date,
    start_row: int = 1,
    row_count: int | None = None,
) -> pd.DataFrame:
    params = build_day_ahead_lmp_params(
        start=start,
        end=end,
        zone=zone,
        start_row=start_row,
        row_count=row_count or config.row_count,
    )
    rows = fetch_pjm_paginated(config=config, endpoint="da_hrl_lmps", base_params=params)
    return normalize_da_lmp_response(rows, zone=zone, timezone=config.timezone)


def write_lmp_cache(
    frame: pd.DataFrame,
    *,
    zone: str,
    start: date,
    end: date,
    output_root: Path | None = None,
) -> Path:
    root = output_root or (get_project_paths().root / "data" / "cache" / "pjm" / "da_hrl_lmps")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"aep_da_lmp_{start.isoformat()}_{end.isoformat()}.parquet"
    frame.to_parquet(path, index=False)
    return path


def read_lmp_cache(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
