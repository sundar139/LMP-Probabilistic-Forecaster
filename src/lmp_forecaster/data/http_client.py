"""HTTP helpers for ingestion adapters.

References:
- PJM Data Miner: https://dataminer2.pjm.com/feed/da_hrl_lmps
- Open-Meteo docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import StringIO
from typing import Any

import httpx

DEFAULT_USER_AGENT = "lmp-forecaster/0.1 (+local-smoke-ingestion)"


@dataclass(frozen=True)
class HttpClientConfig:
    """Configuration for robust HTTP requests."""

    timeout_seconds: float = 20.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    max_backoff_seconds: float = 8.0
    rate_limit_sleep_seconds: float = 0.0
    user_agent: str = DEFAULT_USER_AGENT


class HttpRequestError(RuntimeError):
    """Raised when an HTTP request fails after retries."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


def _backoff_delay(attempt: int, cfg: HttpClientConfig) -> float:
    delay = float(cfg.backoff_factor) * float(2 ** max(0, attempt - 1))
    return float(min(delay, float(cfg.max_backoff_seconds)))


def request_with_retries(
    *,
    method: str,
    url: str,
    params: Mapping[str, str | int | float] | None = None,
    headers: Mapping[str, str] | None = None,
    config: HttpClientConfig | None = None,
    expected_status_codes: set[int] | None = None,
    client: httpx.Client | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Execute an HTTP request with timeout, retries, and clear errors."""
    cfg = config or HttpClientConfig()
    expected = expected_status_codes or {200}

    request_headers = {"User-Agent": cfg.user_agent}
    if headers:
        request_headers.update(headers)

    close_client = False
    active_client = client
    if active_client is None:
        active_client = httpx.Client(timeout=cfg.timeout_seconds)
        close_client = True

    last_error: Exception | None = None
    try:
        for attempt in range(1, cfg.max_retries + 1):
            if cfg.rate_limit_sleep_seconds > 0 and attempt > 1:
                sleep_fn(cfg.rate_limit_sleep_seconds)

            try:
                response = active_client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=request_headers,
                    timeout=cfg.timeout_seconds,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == cfg.max_retries:
                    raise HttpRequestError(
                        f"HTTP request failed after {cfg.max_retries} attempts: {url}",
                        url=url,
                    ) from exc
                sleep_fn(_backoff_delay(attempt, cfg))
                continue

            if response.status_code in expected:
                return response

            should_retry = response.status_code >= 500 and attempt < cfg.max_retries
            if should_retry:
                sleep_fn(_backoff_delay(attempt, cfg))
                continue

            snippet = response.text[:500]
            raise HttpRequestError(
                (
                    f"HTTP {response.status_code} for {url}. "
                    f"Expected one of {sorted(expected)}."
                ),
                url=url,
                status_code=response.status_code,
                response_text=snippet,
            )
    finally:
        if close_client:
            active_client.close()

    raise HttpRequestError(
        f"HTTP request failed after retries: {url}",
        url=url,
        response_text=str(last_error) if last_error else None,
    )


def get_json(
    *,
    url: str,
    params: Mapping[str, str | int | float] | None = None,
    headers: Mapping[str, str] | None = None,
    config: HttpClientConfig | None = None,
    expected_status_codes: set[int] | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch and parse a JSON payload."""
    response = request_with_retries(
        method="GET",
        url=url,
        params=params,
        headers=headers,
        config=config,
        expected_status_codes=expected_status_codes,
        client=client,
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise HttpRequestError(
            "Expected JSON object payload.",
            url=url,
            status_code=response.status_code,
        )
    return payload


def get_text(
    *,
    url: str,
    params: Mapping[str, str | int | float] | None = None,
    headers: Mapping[str, str] | None = None,
    config: HttpClientConfig | None = None,
    expected_status_codes: set[int] | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Fetch plain text or CSV response body."""
    response = request_with_retries(
        method="GET",
        url=url,
        params=params,
        headers=headers,
        config=config,
        expected_status_codes=expected_status_codes,
        client=client,
    )
    return response.text


def parse_csv_records(raw_text: str) -> list[dict[str, str]]:
    """Parse CSV text into a list of dictionaries."""
    reader = csv.DictReader(StringIO(raw_text))
    return [dict(row) for row in reader]
