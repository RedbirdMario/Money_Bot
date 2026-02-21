"""Bollinger Band Breakout strategy."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class BollingerBreakout(Strategy):
    """Bollinger Band Breakout.

    Long when close > upper band. Short when close < lower band.
    Exit at middle band. Optional bandwidth filter.
    """

    name = "bollinger_breakout"
    description = "Bollinger Breakout — Enter on band break, exit at middle band. Optional min-bandwidth filter."

    PARAMS = [
        ParamDescriptor("bb_period", "BB Period", ParamType.INT, 20, 10, 50, 1,
                         tooltip="Lookback for Bollinger Bands SMA"),
        ParamDescriptor("bb_std", "BB Std Dev", ParamType.FLOAT, 2.0, 1.0, 3.5, 0.25,
                         tooltip="Standard deviation multiplier for bands"),
        ParamDescriptor("min_bandwidth", "Min Bandwidth %", ParamType.FLOAT, 0.0, 0.0, 0.1, 0.005,
                         tooltip="Minimum bandwidth filter (0 = disabled)"),
    ]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        min_bandwidth: float = 0.0,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_bandwidth = min_bandwidth

    def get_params(self) -> dict:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "min_bandwidth": self.min_bandwidth,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        close = df["close"]
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        bandwidth = (upper - lower) / sma

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        for i in range(self.bb_period, len(df)):
            ts = df.index[i]
            price = close.iloc[i]
            mid = sma.iloc[i]
            up = upper.iloc[i]
            lo = lower.iloc[i]
            bw = bandwidth.iloc[i]

            if pd.isna(mid):
                continue

            # Bandwidth filter
            if self.min_bandwidth > 0 and bw < self.min_bandwidth:
                if in_position:
                    signals.append(Signal(ts, SignalType.EXIT, position_side, price, "bb_squeeze"))
                    in_position = False
                    position_side = None
                continue

            if not in_position:
                # Long breakout
                if price > up:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "bb_upper_break"))
                    in_position = True
                    position_side = Side.LONG
                # Short breakout
                elif price < lo:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "bb_lower_break"))
                    in_position = True
                    position_side = Side.SHORT
            else:
                # Exit at middle band
                if position_side == Side.LONG and price <= mid:
                    signals.append(Signal(ts, SignalType.EXIT, Side.LONG, price, "bb_middle_exit"))
                    in_position = False
                    position_side = None
                elif position_side == Side.SHORT and price >= mid:
                    signals.append(Signal(ts, SignalType.EXIT, Side.SHORT, price, "bb_middle_exit"))
                    in_position = False
                    position_side = None

        if in_position:
            ts = df.index[-1]
            price = close.iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals
