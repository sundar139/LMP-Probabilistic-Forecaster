"""Tests for real PJM API client helpers."""

from __future__ import annotations

from datetime import date

import pytest

from lmp_forecaster.data.pjm_api import (
    PjmApiConfig,
    PjmApiError,
    build_day_ahead_lmp_params,
    build_pjm_api_headers,
    normalize_da_lmp_response,
    redact_headers,
)


def test_headers_include_subscription_key_only_when_present() -> None:
    no_key = PjmApiConfig(api_key=None)
    with_key = PjmApiConfig(api_key="SECRET")

    assert "Ocp-Apim-Subscription-Key" not in build_pjm_api_headers(no_key)
    assert build_pjm_api_headers(with_key)["Ocp-Apim-Subscription-Key"] == "SECRET"


def test_redact_headers_hides_key() -> None:
    headers = {
        "Ocp-Apim-Subscription-Key": "MYSECRET",
        "Accept": "application/json",
    }
    redacted = redact_headers(headers)
    assert redacted["Ocp-Apim-Subscription-Key"] == "***REDACTED***"


def test_params_include_startrow_rowcount_datetime_range() -> None:
    params = build_day_ahead_lmp_params(
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        zone="AEP",
        start_row=1,
        row_count=10,
    )
    assert params["startRow"] == 1
    assert params["rowCount"] == 10
    assert "datetime_beginning_ept" in params


def test_normalization_maps_columns_and_preserves_negative_prices() -> None:
    payload = {
        "items": [
            {
                "datetime_beginning_ept": "01/01/2024 00:00",
                "datetime_beginning_utc": "2024-01-01T05:00:00Z",
                "pnode_name": "AEP",
                "pnode_type": "ZONE",
                "total_lmp_da": -5.25,
                "system_energy_price_da": 10.0,
                "congestion_price_da": -12.0,
                "marginal_loss_price_da": -3.25,
            }
        ]
    }
    df = normalize_da_lmp_response(payload, zone="AEP")
    assert list(df.columns)[:3] == ["unique_id", "ds", "y"]
    assert df.iloc[0]["unique_id"] == "AEP"
    assert df.iloc[0]["y"] == pytest.approx(-5.25)


def test_missing_lmp_column_raises_clear_error() -> None:
    payload = {
        "items": [
            {
                "datetime_beginning_ept": "01/01/2024 00:00",
                "pnode_name": "AEP",
                "pnode_type": "ZONE",
            }
        ]
    }
    with pytest.raises(PjmApiError, match="LMP"):
        normalize_da_lmp_response(payload, zone="AEP")


def test_aep_filter_works_from_zone_like_fields() -> None:
    payload = {
        "items": [
            {
                "datetime_beginning_ept": "01/01/2024 00:00",
                "pnode_name": "AEP",
                "type": "ZONE",
                "total_lmp_da": 20.0,
            },
            {
                "datetime_beginning_ept": "01/01/2024 01:00",
                "pnode_name": "PSEG",
                "type": "ZONE",
                "total_lmp_da": 21.0,
            },
        ]
    }
    df = normalize_da_lmp_response(payload, zone="AEP")
    assert len(df) == 1
    assert df.iloc[0]["unique_id"] == "AEP"


def test_duplicate_timestamps_can_be_detected_post_normalization() -> None:
    payload = {
        "items": [
            {
                "datetime_beginning_ept": "01/01/2024 00:00",
                "pnode_name": "AEP",
                "pnode_type": "ZONE",
                "total_lmp_da": 10.0,
            },
            {
                "datetime_beginning_ept": "01/01/2024 00:00",
                "pnode_name": "AEP",
                "pnode_type": "ZONE",
                "total_lmp_da": 11.0,
            },
        ]
    }
    df = normalize_da_lmp_response(payload, zone="AEP")
    assert bool(df.duplicated(subset=["unique_id", "ds"]).any())
