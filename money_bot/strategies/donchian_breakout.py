"""Donchian Breakout strategy (Turtle Trading style)."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class DonchianBreakout(Strategy):
    """Turtle-Trading-style Donchian Channel Breakout.

    Long on break above N-period high. Short on break below N-period low.
    Exit via shorter M-period channel.
    """

    name = "donchian_breakout"
    description = "Donchian Breakout — Turtle-style entry on N-period high/low break, exit via shorter M-period channel."

    PARAMS = [
        ParamDescriptor("entry_period", "Entry Period (N)", ParamType.INT, 20, 5, 100, 1,
                         tooltip="Lookback for entry channel (N-period high/low)"),
        ParamDescriptor("exit_period", "Exit Period (M)", ParamType.INT, 10, 3, 50, 1,
                         tooltip="Lookback for exit channel (shorter)"),
    ]

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period

    def get_params(self) -> dict:
        return {
            "entry_period": self.entry_period,
            "exit_period": self.exit_period,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Entry channel
        entry_high = high.rolling(self.entry_period).max()
        entry_low = low.rolling(self.entry_period).min()

        # Exit channel (shorter)
        exit_low = low.rolling(self.exit_period).min()
        exit_high = high.rolling(self.exit_period).max()

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        start = max(self.entry_period, self.exit_period)
        for i in range(start, len(df)):
            ts = df.index[i]
            price = close.iloc[i]
            h = high.iloc[i]
            lo_val = low.iloc[i]

            # Use previous bar's channel values to avoid lookahead
            eh = entry_high.iloc[i - 1]
            el = entry_low.iloc[i - 1]
            xh = exit_high.iloc[i - 1]
            xl = exit_low.iloc[i - 1]

            if pd.isna(eh) or pd.isna(xl):
                continue

            if not in_position:
                # Long entry: high breaks above entry channel
                if h >= eh:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "donchian_break_high"))
                    in_position = True
                    position_side = Side.LONG
                # Short entry: low breaks below entry channel
                elif lo_val <= el:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "donchian_break_low"))
                    in_position = True
                    position_side = Side.SHORT
            else:
                # Long exit: low breaks below exit channel low
                if position_side == Side.LONG and lo_val <= xl:
                    signals.append(Signal(ts, SignalType.EXIT, Side.LONG, price, "donchian_exit_low"))
                    in_position = False
                    position_side = None
                # Short exit: high breaks above exit channel high
                elif position_side == Side.SHORT and h >= xh:
                    signals.append(Signal(ts, SignalType.EXIT, Side.SHORT, price, "donchian_exit_high"))
                    in_position = False
                    position_side = None

        if in_position:
            ts = df.index[-1]
            price = close.iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals
