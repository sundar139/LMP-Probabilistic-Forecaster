"""PJM Day-Ahead Hourly LMP smoke ingestion adapter.

References:
- Feed definition: https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
- Feed endpoint: https://dataminer2.pjm.com/feed/da_hrl_lmps
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from lmp_forecaster.config.paths import get_project_paths
from lmp_forecaster.data.http_client import (
    HttpClientConfig,
    HttpRequestError,
    request_with_retries,
)

PJM_DA_HRL_LMPS_URL = "https://dataminer2.pjm.com/feed/da_hrl_lmps"


class PjmLmpRequestConfig(BaseModel):
    """Typed request config for the PJM DA hourly LMP feed."""

    start_date: date
    end_date: date
    locations: list[str] = Field(default_factory=lambda: ["AEP"])
    feed: str = "da_hrl_lmps"
    location_type: str = "ZONE"
    max_rows: int = 500


def _zone_token(locations: list[str]) -> str:
    normalized = sorted({item.upper() for item in locations})
    if len(normalized) == 1:
        return normalized[0]
    digest_input = ",".join(normalized).encode("utf-8")
    digest = hashlib.sha1(digest_input, usedforsecurity=False).hexdigest()[:10]
    return f"{len(normalized)}zones-{digest}"


def parse_ept_timestamp(value: Any, timezone: str = "America/New_York") -> pd.Timestamp:
    """Parse PJM-style Eastern Prevailing Time timestamp text safely."""
    if value is None:
        raise ValueError("Timestamp value is required.")

    text = str(value).strip()
    text = text.replace("EPT", "").replace("EST", "").replace("EDT", "").strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse PJM timestamp: {value}")

    ts = pd.Timestamp(parsed)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return ts.tz_convert(timezone)


def build_day_ahead_lmp_request(config: PjmLmpRequestConfig) -> tuple[str, dict[str, str | int]]:
    """Build request URL and params for day-ahead hourly LMP feed.

    Data Miner query keys can evolve. This function centralizes request construction so
    endpoint-specific changes remain isolated and testable.
    """
    params: dict[str, str | int] = {
        "start": config.start_date.isoformat(),
        "end": config.end_date.isoformat(),
        "location_type": config.location_type,
        "rowCount": config.max_rows,
    }
    if config.locations:
        params["locations"] = ",".join(sorted({item.upper() for item in config.locations}))
    return PJM_DA_HRL_LMPS_URL, params


def _extract_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("items", "data", "results"):
            maybe = payload.get(key)
            if isinstance(maybe, list):
                return [item for item in maybe if isinstance(item, dict)]
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in payload.values()):
            return [payload]
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    candidate_set = {item.lower() for item in candidates}
    for col in columns:
        normalized = col.lower().replace(" ", "_")
        if normalized in candidate_set:
            return col
    return None


def normalize_pjm_lmp_rows(
    records: list[dict[str, Any]],
    *,
    fallback_location_type: str,
    source: str = "pjm_dataminer_da_hrl_lmps",
) -> pd.DataFrame:
    """Normalize PJM LMP rows into internal schema."""
    if not records:
        empty_cols = [
            "unique_id",
            "ds",
            "y",
            "market",
            "location_type",
            "source",
            "pulled_at",
        ]
        return pd.DataFrame(columns=empty_cols)

    frame = pd.DataFrame.from_records(records)
    columns = list(frame.columns)

    ts_col = _find_column(
        columns,
        [
            "datetime_beginning_ept",
            "datetime_ending_ept",
            "datetime_beginning_utc",
            "datetime_ending_utc",
            "datetime",
            "timestamp",
            "date",
            "ds",
        ],
    )
    if ts_col is None:
        raise ValueError("PJM payload missing a recognizable timestamp column.")

    y_col = _find_column(
        columns,
        [
            "total_lmp_da",
            "lmp",
            "da_lmp",
            "total_lmp",
            "price",
            "value",
            "system_energy_price_da",
        ],
    )
    if y_col is None:
        raise ValueError(
            "PJM payload missing an LMP/price column. "
            "Checked: total_lmp_da, lmp, da_lmp, total_lmp, price, value."
        )

    zone_col = _find_column(
        columns,
        ["pnode_name", "zone", "location", "location_name", "name", "unique_id"],
    )
    market_col = _find_column(columns, ["market", "market_type", "market_run_id"])
    location_type_col = _find_column(columns, ["location_type", "type"])

    ds = frame[ts_col].map(parse_ept_timestamp)
    y = pd.to_numeric(frame[y_col], errors="coerce")
    unique_id = (
        frame[zone_col].astype(str).str.upper()
        if zone_col is not None
        else pd.Series(["UNKNOWN"] * len(frame), index=frame.index)
    )

    pulled_at = pd.Timestamp(datetime.now(UTC))
    normalized = pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": ds,
            "y": y,
            "market": (
                frame[market_col].astype(str).str.upper()
                if market_col is not None
                else "DAY_AHEAD"
            ),
            "location_type": (
                frame[location_type_col].astype(str).str.upper()
                if location_type_col is not None
                else fallback_location_type
            ),
            "source": source,
            "pulled_at": pulled_at,
        }
    )
    normalized.sort_values(["unique_id", "ds"], inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


@dataclass(frozen=True)
class PjmSmokeResult:
    """Result metadata for PJM smoke ingestion."""

    output_path: Path
    normalized: pd.DataFrame
    request_url: str
    request_params: dict[str, str | int]


def plan_pjm_smoke_output_path(config: PjmLmpRequestConfig) -> Path:
    """Build deterministic local cache path for smoke output."""
    paths = get_project_paths()
    token = _zone_token(config.locations)
    filename = (
        f"{config.feed}_{config.start_date.isoformat()}_"
        f"{config.end_date.isoformat()}_{token}.parquet"
    )
    return paths.root / "data" / "cache" / "pjm" / config.feed / filename


def pull_pjm_lmp_smoke(
    config: PjmLmpRequestConfig,
    *,
    write: bool,
    write_raw: bool = False,
    http_config: HttpClientConfig | None = None,
) -> PjmSmokeResult:
    """Pull small PJM smoke payload and optionally write cache output."""
    if config.max_rows > 5000:
        raise ValueError("max_rows guard exceeded for smoke pull. Use <= 5000 rows.")

    request_url, params = build_day_ahead_lmp_request(config)
    output_path = plan_pjm_smoke_output_path(config)

    if not write:
        return PjmSmokeResult(
            output_path=output_path,
            normalized=pd.DataFrame(),
            request_url=request_url,
            request_params=params,
        )

    response = request_with_retries(
        method="GET",
        url=request_url,
        params=params,
        config=http_config,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HttpRequestError(
            "PJM response was not valid JSON. "
            "Verify feed parameters and endpoint format.",
            url=request_url,
            status_code=response.status_code,
            response_text=response.text[:500],
        ) from exc

    records = _extract_records_from_payload(payload)
    normalized = normalize_pjm_lmp_rows(records, fallback_location_type=config.location_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    if write_raw:
        raw_path = output_path.with_suffix(".raw.json")
        pd.Series(payload).to_json(raw_path, indent=2)

    return PjmSmokeResult(
        output_path=output_path,
        normalized=normalized,
        request_url=request_url,
        request_params=params,
    )
