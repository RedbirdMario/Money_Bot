"""Flat market detection filter."""

from __future__ import annotations

import pandas as pd
import numpy as np


class FlatMarketFilter:
    """Detect flat/ranging markets using rolling range as % of price."""

    def __init__(self, window: int = 20, threshold: float = 0.02):
        self.window = window
        self.threshold = threshold  # 2% range = flat

    def is_flat(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True where market is flat."""
        rolling_high = df["high"].rolling(self.window).max()
        rolling_low = df["low"].rolling(self.window).min()
        rolling_mid = (rolling_high + rolling_low) / 2

        range_pct = (rolling_high - rolling_low) / rolling_mid
        return range_pct < self.threshold
