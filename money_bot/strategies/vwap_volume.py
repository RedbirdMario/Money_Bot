"""Rolling VWAP + Volume Spike strategy."""

from __future__ import annotations

import pandas as pd

from money_bot.types import Signal, SignalType, Side
from money_bot.strategies.base import Strategy, ParamDescriptor, ParamType
from money_bot.strategies.registry import register


@register
class VWAPVolume(Strategy):
    """Rolling VWAP crossover with volume spike detection.

    Long on VWAP crossover up + volume > X * SMA(volume).
    Short on VWAP crossover down + volume spike.
    Uses rolling VWAP (not session-reset) for crypto/intraday.
    """

    name = "vwap_volume"
    description = "VWAP + Volume — Rolling VWAP crossover with volume spike filter for intraday/crypto."

    PARAMS = [
        ParamDescriptor("vwap_period", "VWAP Period", ParamType.INT, 20, 5, 100, 1,
                         tooltip="Rolling window for VWAP calculation"),
        ParamDescriptor("vol_sma_period", "Volume SMA Period", ParamType.INT, 20, 5, 100, 1,
                         tooltip="Period for volume moving average"),
        ParamDescriptor("vol_multiplier", "Volume Spike Multiplier", ParamType.FLOAT, 1.5, 1.0, 5.0, 0.25,
                         tooltip="Volume must be > X * SMA(volume) to confirm signal"),
    ]

    def __init__(
        self,
        vwap_period: int = 20,
        vol_sma_period: int = 20,
        vol_multiplier: float = 1.5,
    ):
        self.vwap_period = vwap_period
        self.vol_sma_period = vol_sma_period
        self.vol_multiplier = vol_multiplier

    def get_params(self) -> dict:
        return {
            "vwap_period": self.vwap_period,
            "vol_sma_period": self.vol_sma_period,
            "vol_multiplier": self.vol_multiplier,
        }

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        close = df["close"]
        volume = df["volume"]
        typical_price = (df["high"] + df["low"] + close) / 3

        # Rolling VWAP
        tp_vol = typical_price * volume
        vwap = tp_vol.rolling(self.vwap_period).sum() / volume.rolling(self.vwap_period).sum()

        # Volume SMA for spike detection
        vol_sma = volume.rolling(self.vol_sma_period).mean()

        signals: list[Signal] = []
        in_position = False
        position_side: Side | None = None

        start = max(self.vwap_period, self.vol_sma_period)
        for i in range(start, len(df)):
            ts = df.index[i]
            price = close.iloc[i]
            price_prev = close.iloc[i - 1]
            vw = vwap.iloc[i]
            vw_prev = vwap.iloc[i - 1]
            vol = volume.iloc[i]
            vol_avg = vol_sma.iloc[i]

            if pd.isna(vw) or pd.isna(vw_prev) or pd.isna(vol_avg):
                continue

            vol_spike = vol >= vol_avg * self.vol_multiplier

            if not in_position:
                # Long: price crosses above VWAP + volume spike
                if price_prev <= vw_prev and price > vw and vol_spike:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.LONG, price, "vwap_cross_up_vol"))
                    in_position = True
                    position_side = Side.LONG
                # Short: price crosses below VWAP + volume spike
                elif price_prev >= vw_prev and price < vw and vol_spike:
                    signals.append(Signal(ts, SignalType.ENTRY, Side.SHORT, price, "vwap_cross_down_vol"))
                    in_position = True
                    position_side = Side.SHORT
            else:
                # Exit long: price crosses below VWAP
                if position_side == Side.LONG and price < vw:
                    signals.append(Signal(ts, SignalType.EXIT, Side.LONG, price, "vwap_exit_below"))
                    in_position = False
                    position_side = None
                # Exit short: price crosses above VWAP
                elif position_side == Side.SHORT and price > vw:
                    signals.append(Signal(ts, SignalType.EXIT, Side.SHORT, price, "vwap_exit_above"))
                    in_position = False
                    position_side = None

        if in_position:
            ts = df.index[-1]
            price = close.iloc[-1]
            signals.append(Signal(ts, SignalType.EXIT, position_side, price, "end_of_data"))

        return signals
