"""Double EMA crossover strategy."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register
from money_bot.strategies.filters.flat_market import FlatMarketFilter


@register
class DoubleEMA(Strategy):
    """EMA crossover: fast EMA crosses slow EMA."""

    name = "double_ema"
    description = "Double EMA Crossover — Long when fast EMA crosses above slow EMA, short on cross below."

    PARAMS = [
        ParamDescriptor("fast_period", "Fast EMA Period", ParamType.INT, 20, 5, 100, 1,
                         tooltip="Period for the fast exponential moving average"),
        ParamDescriptor("slow_period", "Slow EMA Period", ParamType.INT, 155, 50, 300, 5,
                         tooltip="Period for the slow exponential moving average"),
        ParamDescriptor("use_flat_filter", "Flat Market Filter", ParamType.BOOL, True,
                         tooltip="Exit positions in flat/ranging markets"),
        ParamDescriptor("flat_threshold", "Flat Threshold %", ParamType.FLOAT, 0.02, 0.005, 0.05, 0.005,
                         tooltip="Range threshold below which market is considered flat"),
    ]

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 155,
        use_flat_filter: bool = True,
        flat_window: int = 20,
        flat_threshold: float = 0.02,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_flat_filter = use_flat_filter
        self.flat_filter = FlatMarketFilter(
            window=flat_window, threshold=flat_threshold
        )

    def get_params(self) -> dict:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "use_flat_filter": self.use_flat_filter,
            "flat_threshold": self.flat_filter.threshold,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()

        # Flat market mask
        if self.use_flat_filter:
            flat_mask = self.flat_filter.is_flat(df)
        else:
            flat_mask = pd.Series(False, index=df.index)

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        for i in range(1, len(df)):
            ts = df.index[i]
            price = df["close"].iloc[i]

            if flat_mask.iloc[i]:
                # Close position in flat market
                if in_position:
                    signals.append(
                        Signal(ts, SignalType.EXIT, position_side, price, "flat_market")
                    )
                    in_position = False
                    position_side = None
                continue

            prev_fast = fast_ema.iloc[i - 1]
            prev_slow = slow_ema.iloc[i - 1]
            curr_fast = fast_ema.iloc[i]
            curr_slow = slow_ema.iloc[i]

            # Bullish crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                if in_position and position_side == Side.SHORT:
                    signals.append(
                        Signal(ts, SignalType.EXIT, Side.SHORT, price, "ema_cross_bull")
                    )
                    in_position = False
                if not in_position:
                    signals.append(
                        Signal(ts, SignalType.ENTRY, Side.LONG, price, "ema_cross_bull")
                    )
                    in_position = True
                    position_side = Side.LONG

            # Bearish crossover
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                if in_position and position_side == Side.LONG:
                    signals.append(
                        Signal(ts, SignalType.EXIT, Side.LONG, price, "ema_cross_bear")
                    )
                    in_position = False
                if not in_position:
                    signals.append(
                        Signal(ts, SignalType.ENTRY, Side.SHORT, price, "ema_cross_bear")
                    )
                    in_position = True
                    position_side = Side.SHORT

        # Close any remaining position at end
        if in_position:
            ts = df.index[-1]
            price = df["close"].iloc[-1]
            signals.append(
                Signal(ts, SignalType.EXIT, position_side, price, "end_of_data")
            )

        return signals
