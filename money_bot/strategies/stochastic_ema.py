"""Stochastic + EMA Trend Filter strategy."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class StochasticEMA(Strategy):
    """Stochastic %K/%D crossover with EMA trend filter.

    Long only when EMA is rising AND %K crosses above %D in oversold zone.
    Short only when EMA is falling AND %K crosses below %D in overbought zone.
    Uses EMA direction (not price vs EMA) since price is typically below EMA when oversold.
    """

    name = "stochastic_ema"
    description = "Stochastic + EMA — %K/%D crossover in extreme zones, filtered by EMA trend direction."

    PARAMS = [
        ParamDescriptor("stoch_k", "Stoch %K Period", ParamType.INT, 14, 5, 30, 1,
                         tooltip="Stochastic %K lookback period"),
        ParamDescriptor("stoch_d", "Stoch %D Smooth", ParamType.INT, 3, 2, 10, 1,
                         tooltip="Smoothing period for %D (SMA of %K)"),
        ParamDescriptor("ema_period", "EMA Trend Period", ParamType.INT, 50, 10, 200, 5,
                         tooltip="EMA period for trend filter"),
        ParamDescriptor("oversold", "Oversold Level", ParamType.INT, 20, 5, 40, 5,
                         tooltip="Stochastic oversold threshold"),
        ParamDescriptor("overbought", "Overbought Level", ParamType.INT, 80, 60, 95, 5,
                         tooltip="Stochastic overbought threshold"),
    ]

    def __init__(
        self,
        stoch_k: int = 14,
        stoch_d: int = 3,
        ema_period: int = 50,
        oversold: int = 20,
        overbought: int = 80,
    ):
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.ema_period = ema_period
        self.oversold = oversold
        self.overbought = overbought

    def get_params(self) -> dict:
        return {
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "ema_period": self.ema_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Stochastic %K
        lowest_low = low.rolling(self.stoch_k).min()
        highest_high = high.rolling(self.stoch_k).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(self.stoch_d).mean()

        # EMA trend filter — compare EMA now vs EMA from ema_period/2 bars ago
        # to get broader trend direction (not affected by short-term pullback)
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        trend_lookback = max(self.ema_period // 2, 5)

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        start = max(trend_lookback + 1, self.stoch_k)
        for i in range(start, len(df)):
            ts = df.index[i]
            price = close.iloc[i]
            k_now = k.iloc[i]
            k_prev = k.iloc[i - 1]
            d_now = d.iloc[i]
            d_prev = d.iloc[i - 1]
            ema_now = ema.iloc[i]
            ema_past = ema.iloc[i - trend_lookback]
            ema_uptrend = ema_now > ema_past
            ema_downtrend = ema_now < ema_past

            if pd.isna(k_now) or pd.isna(d_now) or pd.isna(d_prev):
                continue

            if not in_position:
                # Long: EMA in uptrend, %K crosses above %D in oversold zone
                if (ema_uptrend and d_prev < self.oversold
                        and k_prev <= d_prev and k_now > d_now):
                    signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "stoch_cross_up_oversold"))
                    in_position = True
                    position_side = Side.LONG
                # Short: EMA in downtrend, %K crosses below %D in overbought zone
                elif (ema_downtrend and d_prev > self.overbought
                      and k_prev >= d_prev and k_now < d_now):
                    signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "stoch_cross_down_overbought"))
                    in_position = True
                    position_side = Side.SHORT
            else:
                # Exit long: %K crosses below %D in overbought
                if (position_side == Side.LONG and d_now > self.overbought
                        and k_prev >= d_prev and k_now < d_now):
                    signals.append(Signal(ts, SignalType.EXIT, Side.LONG, price, "stoch_exit_overbought"))
                    in_position = False
                    position_side = None
                # Exit short: %K crosses above %D in oversold
                elif (position_side == Side.SHORT and d_now < self.oversold
                      and k_prev <= d_prev and k_now > d_now):
                    signals.append(Signal(ts, SignalType.EXIT, Side.SHORT, price, "stoch_exit_oversold"))
                    in_position = False
                    position_side = None

        if in_position:
            ts = df.index[-1]
            price = close.iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals
