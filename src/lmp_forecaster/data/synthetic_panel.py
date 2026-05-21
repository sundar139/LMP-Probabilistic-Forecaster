"""Synthetic panel generator for smoke tests and local checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticPanelConfig:
    """Configuration for deterministic synthetic panel generation."""

    zones: tuple[str, ...] = ("AEP",)
    start: str = "2024-03-09 00:00:00"
    periods: int = 240
    freq: str = "h"
    timezone: str = "America/New_York"
    seed: int = 7


def _build_index(config: SyntheticPanelConfig) -> pd.DatetimeIndex:
    return pd.date_range(
        start=config.start,
        periods=config.periods,
        freq=config.freq,
        tz=config.timezone,
    )


def make_synthetic_panel(config: SyntheticPanelConfig | None = None) -> pd.DataFrame:
    """Create deterministic long-format hourly panel with lag features."""
    cfg = config or SyntheticPanelConfig()
    index = _build_index(cfg)
    rng = np.random.default_rng(cfg.seed)

    rows: list[pd.DataFrame] = []
    for i, zone in enumerate(cfg.zones):
        step = np.arange(len(index), dtype=float)
        signal = 30 + 6 * np.sin(2 * np.pi * step / 24) + 2 * np.cos(2 * np.pi * step / 168)
        drift = i * 1.5
        noise = rng.normal(0, 0.35, size=len(index))

        y = signal + drift + noise
        temperature = 60 + 15 * np.sin(2 * np.pi * (step + 5) / 24) + rng.normal(0, 0.5, len(index))
        load_forecast = 10000 + 220 * np.cos(2 * np.pi * step / 24) + rng.normal(0, 12, len(index))

        panel = pd.DataFrame(
            {
                "unique_id": zone,
                "ds": index,
                "y": y,
                "hour": index.hour,
                "day_of_week": index.dayofweek,
                "is_weekend": (index.dayofweek >= 5).astype(int),
                "temperature_2m": temperature,
                "load_forecast": load_forecast,
            }
        )
        panel["lmp_lag_1"] = panel["y"].shift(1)
        panel["lmp_lag_24"] = panel["y"].shift(24)
        panel["lmp_lag_168"] = panel["y"].shift(168)
        rows.append(panel)

    full = pd.concat(rows, axis=0, ignore_index=True)
    full.sort_values(["unique_id", "ds"], inplace=True)
    full.reset_index(drop=True, inplace=True)

    return full
