"""MACD Histogram Zero-Crossover strategy."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class MACDHistogram(Strategy):
    """MACD Histogram zero-crossover.

    Long when histogram crosses from neg to pos.
    Short when histogram crosses from pos to neg.
    Optional: trend confirmation (MACD line > 0 for long).
    """

    name = "macd_histogram"
    description = "MACD Histogram — Enter on histogram zero-crossover. Optional trend confirmation via MACD line."

    PARAMS = [
        ParamDescriptor("fast_period", "Fast EMA", ParamType.INT, 12, 5, 30, 1,
                         tooltip="Fast EMA period for MACD"),
        ParamDescriptor("slow_period", "Slow EMA", ParamType.INT, 26, 15, 60, 1,
                         tooltip="Slow EMA period for MACD"),
        ParamDescriptor("signal_period", "Signal Period", ParamType.INT, 9, 3, 20, 1,
                         tooltip="Signal line EMA period"),
        ParamDescriptor("trend_confirm", "Trend Confirmation", ParamType.BOOL, False,
                         tooltip="Require MACD line > 0 for long, < 0 for short"),
    ]

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        trend_confirm: bool = False,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_confirm = trend_confirm

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "trend_confirm": self.trend_confirm,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        close = df["close"]
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        for i in range(1, len(df)):
            ts = df.index[i]
            price = close.iloc[i]
            hist = histogram.iloc[i]
            hist_prev = histogram.iloc[i - 1]
            ml = macd_line.iloc[i]

            if pd.isna(hist) or pd.isna(hist_prev):
                continue

            # Histogram crosses zero: neg → pos
            if hist_prev <= 0 and hist > 0:
                if self.trend_confirm and ml <= 0:
                    continue
                if in_position and position_side == Side.SHORT:
                    signals.append(Signal(ts, SignalType.EXIT, Side.SHORT, price, "macd_hist_bull"))
                    in_position = False
                if not in_position:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "macd_hist_bull"))
                    in_position = True
                    position_side = Side.LONG

            # Histogram crosses zero: pos → neg
            elif hist_prev >= 0 and hist < 0:
                if self.trend_confirm and ml >= 0:
                    continue
                if in_position and position_side == Side.LONG:
                    signals.append(Signal(ts, SignalType.EXIT, Side.LONG, price, "macd_hist_bear"))
                    in_position = False
                if not in_position:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "macd_hist_bear"))
                    in_position = True
                    position_side = Side.SHORT

        if in_position:
            ts = df.index[-1]
            price = close.iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals
