"""Tests for HTTP helper utilities."""

from __future__ import annotations

import httpx
import pytest

from lmp_forecaster.data.http_client import (
    HttpClientConfig,
    HttpRequestError,
    parse_csv_records,
    request_with_retries,
)


def test_request_with_retries_recovers_after_server_error() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500, text="temporary")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    response = request_with_retries(
        method="GET",
        url="https://example.com/test",
        client=client,
        config=HttpClientConfig(max_retries=3, backoff_factor=0.0),
        sleep_fn=lambda _: None,
    )

    assert response.status_code == 200
    assert attempts["count"] == 2


def test_request_with_retries_raises_clear_status_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(404, text="not found"))
    client = httpx.Client(transport=transport)

    with pytest.raises(HttpRequestError) as exc:
        request_with_retries(
            method="GET",
            url="https://example.com/missing",
            client=client,
            config=HttpClientConfig(max_retries=1),
            sleep_fn=lambda _: None,
        )

    assert exc.value.status_code == 404
    assert "Expected one of" in str(exc.value)


def test_parse_csv_records() -> None:
    csv_text = "a,b\n1,2\n3,4\n"
    rows = parse_csv_records(csv_text)
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
