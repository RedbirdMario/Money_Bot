"""Shared test fixtures — sample OHLCV data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 500 bars of synthetic OHLCV data with a clear trend."""
    rng = np.random.default_rng(123)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    # Trending data: up for first half, down for second half
    trend = np.concatenate([
        np.linspace(0, 0.3, n // 2),
        np.linspace(0.3, 0.1, n - n // 2),
    ])
    noise = rng.normal(0, 0.005, n).cumsum()
    close = 100 * np.exp(trend + noise)

    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.uniform(1000, 50000, n)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """Generate flat/ranging market data."""
    rng = np.random.default_rng(456)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    close = 100 + rng.normal(0, 0.1, n)
    high = close + rng.uniform(0, 0.2, n)
    low = close - rng.uniform(0, 0.2, n)
    open_ = close + rng.normal(0, 0.05, n)
    volume = rng.uniform(1000, 50000, n)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)
