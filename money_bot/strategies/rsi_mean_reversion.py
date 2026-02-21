"""RSI Mean Reversion strategy."""

from __future__ import annotations

import pandas as pd
import numpy as np

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class RSIMeanReversion(Strategy):
    """RSI Oversold/Overbought Crossback.

    Long when RSI drops below oversold then crosses back above.
    Short when RSI rises above overbought then crosses back below.
    Exit at midline or opposite zone.
    """

    name = "rsi_mean_reversion"
    description = "RSI Mean Reversion — Enter on RSI crossback from oversold/overbought zones, exit at midline."

    PARAMS = [
        ParamDescriptor("rsi_period", "RSI Period", ParamType.INT, 14, 5, 50, 1,
                         tooltip="Lookback period for RSI calculation"),
        ParamDescriptor("oversold", "Oversold Level", ParamType.INT, 30, 10, 40, 5,
                         tooltip="RSI level below which market is oversold"),
        ParamDescriptor("overbought", "Overbought Level", ParamType.INT, 70, 60, 90, 5,
                         tooltip="RSI level above which market is overbought"),
        ParamDescriptor("exit_at_midline", "Exit at Midline (50)", ParamType.BOOL, True,
                         tooltip="Close position when RSI reaches 50"),
    ]

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        exit_at_midline: bool = True,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_at_midline = exit_at_midline

    def get_params(self) -> dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "exit_at_midline": self.exit_at_midline,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        rsi = self._calc_rsi(df["close"], self.rsi_period)

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None
        was_oversold = False
        was_overbought = False

        for i in range(1, len(df)):
            ts = df.index[i]
            price = df["close"].iloc[i]
            r = rsi.iloc[i]
            r_prev = rsi.iloc[i - 1]

            if np.isnan(r) or np.isnan(r_prev):
                continue

            # Track zone visits
            if r_prev < self.oversold:
                was_oversold = True
            if r_prev > self.overbought:
                was_overbought = True

            # Long entry: was oversold, now crosses back above oversold
            if not in_position and was_oversold and r_prev <= self.oversold and r > self.oversold:
                signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "rsi_crossback_oversold"))
                in_position = True
                position_side = Side.LONG
                was_oversold = False

            # Short entry: was overbought, now crosses back below overbought
            elif not in_position and was_overbought and r_prev >= self.overbought and r < self.overbought:
                signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "rsi_crossback_overbought"))
                in_position = True
                position_side = Side.SHORT
                was_overbought = False

            # Exits
            elif in_position:
                should_exit = False
                reason = ""

                if self.exit_at_midline and position_side == Side.LONG and r >= 50:
                    should_exit = True
                    reason = "rsi_midline"
                elif self.exit_at_midline and position_side == Side.SHORT and r <= 50:
                    should_exit = True
                    reason = "rsi_midline"
                elif position_side == Side.LONG and r >= self.overbought:
                    should_exit = True
                    reason = "rsi_overbought"
                elif position_side == Side.SHORT and r <= self.oversold:
                    should_exit = True
                    reason = "rsi_oversold"

                if should_exit:
                    signals.append(Signal(ts, SignalType.EXIT, position_side, price, reason))
                    in_position = False
                    position_side = None

        # Close remaining
        if in_position:
            ts = df.index[-1]
            price = df["close"].iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
