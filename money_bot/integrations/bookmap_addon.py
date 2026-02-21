"""Bookmap Python Addon — streams live signals from Money_Bot strategies onto the heatmap."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from money_bot.strategies.registry import get_strategy_class
from money_bot.types import Signal, SignalType, Side

logger = logging.getLogger(__name__)

# Bookmap colors (ARGB int format)
GREEN = 0xFF00CC00
RED = 0xFFCC0000
WHITE = 0xFFFFFFFF
CYAN = 0xFF00CCCC

# ---------------------------------------------------------------------------
# CandleAggregator — builds OHLCV candles from individual trade ticks
# ---------------------------------------------------------------------------

class CandleAggregator:
    """Aggregate trade ticks into OHLCV candles of a fixed time interval."""

    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self._current: dict[str, Any] | None = None

    def add_tick(self, timestamp_ms: int, price: float, size: float) -> dict | None:
        """Feed a trade tick. Returns a finished candle dict when the interval rolls over, else None."""
        bucket_start = (timestamp_ms // (self.interval * 1000)) * (self.interval * 1000)

        if self._current is None:
            self._current = self._new_candle(bucket_start, price, size)
            return None

        # Same bucket — update running candle
        if bucket_start == self._current["start_ms"]:
            self._update(price, size)
            return None

        # New bucket — finalize previous candle, start fresh
        finished = self._finalize()
        self._current = self._new_candle(bucket_start, price, size)
        return finished

    def flush(self) -> dict | None:
        """Force-close the current candle (e.g. on shutdown)."""
        if self._current is None or self._current["volume"] == 0.0:
            return None
        return self._finalize()

    # -- internal helpers --

    def _new_candle(self, start_ms: int, price: float, size: float) -> dict:
        return {
            "start_ms": start_ms,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": abs(size),
        }

    def _update(self, price: float, size: float) -> None:
        c = self._current
        if price > c["high"]:
            c["high"] = price
        if price < c["low"]:
            c["low"] = price
        c["close"] = price
        c["volume"] += abs(size)

    def _finalize(self) -> dict:
        c = self._current
        self._current = None
        return {
            "timestamp": pd.Timestamp(c["start_ms"], unit="ms", tz="UTC"),
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
        }


# ---------------------------------------------------------------------------
# CandleBuffer — rolling DataFrame of the last N candles
# ---------------------------------------------------------------------------

class CandleBuffer:
    """Keep a rolling window of OHLCV candles as a pd.DataFrame."""

    COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        self._rows: list[dict] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, candle: dict) -> None:
        self._rows.append(candle)
        if len(self._rows) > self.max_candles:
            self._rows = self._rows[-self.max_candles:]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.DataFrame(self._rows)
        df.index = pd.DatetimeIndex(df.pop("timestamp"), name="timestamp")
        return df[self.COLUMNS]


# ---------------------------------------------------------------------------
# BookmapAddon — main addon that wires everything together
# ---------------------------------------------------------------------------

class BookmapAddon:
    """Bookmap Python addon that runs a Money_Bot strategy on live trade data.

    Usage::

        addon = BookmapAddon("double_ema", {"fast_period": 20, "slow_period": 155})
        addon.start()
    """

    def __init__(
        self,
        strategy_name: str,
        strategy_params: dict | None = None,
        interval_seconds: int = 3600,
        max_candles: int = 500,
    ):
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.interval_seconds = interval_seconds

        # Core components
        self.aggregator = CandleAggregator(interval_seconds)
        self.buffer = CandleBuffer(max_candles)

        # Strategy (loaded lazily in on_subscribe so settings can override)
        self.strategy = None

        # Pending candles produced by add_tick (consumed by _on_interval)
        self._pending_candles: list[dict] = []

        # Signal tracking for diff logic
        self._prev_signal_count: int = 0
        self._cumulative_pnl: float = 0.0

        # Bookmap handles
        self._addon = None
        self._alias: str | None = None
        self._pips: float = 1.0
        self._indicator_heatmap: int | None = None
        self._indicator_pnl: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to Bookmap and start processing events."""
        try:
            import bookmap as bm
        except ImportError:
            raise ImportError(
                "bookmap package not installed. Run: pip install bookmap"
            )

        addon = bm.create_addon()

        # Register settings UI
        bm.add_number_setting(addon, "interval", "Candle Interval (sec)", self.interval_seconds)
        bm.add_string_setting(addon, "strategy", "Strategy Name", self.strategy_name)
        bm.add_number_setting(addon, "max_candles", "Max Candles Buffer", self.buffer.max_candles)
        bm.add_string_setting(addon, "params_json", "Strategy Params (JSON)", json.dumps(self.strategy_params))

        # Register event handlers
        bm.on_subscribe(addon, self._on_subscribe)
        bm.on_unsubscribe(addon, self._on_unsubscribe)
        bm.on_interval(addon, self._on_interval)

        self._addon = addon
        bm.start_addon(addon)

    # ------------------------------------------------------------------
    # Bookmap event handlers
    # ------------------------------------------------------------------

    def _on_subscribe(self, addon, alias, instrument_name, is_crypto, pips, size_multiplier):
        import bookmap as bm

        self._alias = alias
        self._pips = pips

        # Read settings (may have been changed in Bookmap UI)
        interval_val = bm.get_setting(addon, "interval")
        if interval_val is not None:
            self.aggregator = CandleAggregator(int(interval_val))

        strategy_val = bm.get_setting(addon, "strategy")
        if strategy_val is not None:
            self.strategy_name = strategy_val

        max_candles_val = bm.get_setting(addon, "max_candles")
        if max_candles_val is not None:
            self.buffer = CandleBuffer(int(max_candles_val))

        params_val = bm.get_setting(addon, "params_json")
        if params_val:
            try:
                self.strategy_params = json.loads(params_val)
            except json.JSONDecodeError:
                logger.warning("Invalid params JSON: %s — using defaults", params_val)

        # Load strategy
        cls = get_strategy_class(self.strategy_name)
        self.strategy = cls.from_params(self.strategy_params)
        logger.info("Loaded strategy: %s with params %s", self.strategy_name, self.strategy_params)

        # Register indicators
        self._indicator_heatmap = bm.register_indicator(
            addon,
            alias,
            bm.IndicatorConfig(
                name=f"MB:{self.strategy_name}",
                indicator_type=bm.IndicatorType.PRIMARY,
                line_style=bm.LineStyle.NONE,
            ),
        )
        self._indicator_pnl = bm.register_indicator(
            addon,
            alias,
            bm.IndicatorConfig(
                name=f"MB:PnL",
                indicator_type=bm.IndicatorType.BOTTOM,
                line_style=bm.LineStyle.SOLID,
                color=CYAN,
            ),
        )

        # Register trade handler (needs alias to be set first)
        bm.on_trade(addon, alias, self._on_trade)

        logger.info("Subscribed to %s (pips=%s)", instrument_name, pips)

    def _on_unsubscribe(self, addon, alias):
        logger.info("Unsubscribed from alias %s", alias)
        # Flush any pending candle
        candle = self.aggregator.flush()
        if candle:
            self.buffer.append(candle)

    def _on_trade(self, addon, alias, price_level, size, is_otc, is_bid, is_execution_start, is_execution_end):
        price = price_level * self._pips
        candle = self.aggregator.add_tick(
            timestamp_ms=int(pd.Timestamp.now(tz="UTC").timestamp() * 1000),
            price=price,
            size=size,
        )
        if candle is not None:
            self._pending_candles.append(candle)

    def _on_interval(self, addon, alias):
        """Called every 100ms by Bookmap. Process any candles completed since last call."""
        if self.strategy is None or not self._pending_candles:
            return

        # Drain all pending candles into buffer
        for candle in self._pending_candles:
            self.buffer.append(candle)
        self._pending_candles.clear()

        # Need enough candles for the strategy to work
        if len(self.buffer) < 2:
            return

        df = self.buffer.to_dataframe()
        signals = self.strategy.generate_signals(df)

        # Signal diff — only process new signals
        fresh = signals[self._prev_signal_count:]
        self._prev_signal_count = len(signals)

        for sig in fresh:
            self._draw_signal(sig)
            self._update_pnl(sig)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_signal(self, signal: Signal) -> None:
        import bookmap as bm

        if self._indicator_heatmap is None or self._alias is None:
            return

        ts_ms = int(signal.timestamp.timestamp() * 1000)
        price_level = signal.price / self._pips

        if signal.signal_type == SignalType.ENTRY:
            color = GREEN if signal.side == Side.LONG else RED
        else:
            color = WHITE

        bm.add_point(
            self._addon,
            self._indicator_heatmap,
            self._alias,
            ts_ms,
            price_level,
            color,
        )

        logger.info(
            "Signal: %s %s @ %.2f (%s)",
            signal.signal_type.value,
            signal.side.value,
            signal.price,
            signal.reason,
        )

    def _update_pnl(self, signal: Signal) -> None:
        """Track cumulative PnL line on the bottom subchart."""
        import bookmap as bm

        if self._indicator_pnl is None or self._alias is None:
            return

        ts_ms = int(signal.timestamp.timestamp() * 1000)

        bm.add_point(
            self._addon,
            self._indicator_pnl,
            self._alias,
            ts_ms,
            0,  # PnL level placeholder — real PnL needs position tracking
            CYAN,
        )
