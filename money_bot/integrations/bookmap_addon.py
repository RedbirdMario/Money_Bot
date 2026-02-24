"""Bookmap Python Addon — streams live signals from Money_Bot strategies onto the heatmap.

Uses the real bookmap Python API (pip install bookmap).
API reference: https://github.com/niceprice/bookmap-python-api
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd

from money_bot.strategies.registry import get_strategy_class
from money_bot.types import Signal, SignalType, Side

logger = logging.getLogger(__name__)

# Bookmap colors as RGB tuples
COLOR_GREEN = (0, 204, 0)
COLOR_RED = (204, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (0, 204, 204)

# Indicator request IDs (arbitrary, used to map responses)
REQ_SIGNAL_LONG = 1
REQ_SIGNAL_SHORT = 2
REQ_SIGNAL_EXIT = 3
REQ_PNL_INDICATOR = 4

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

    Registers 3 PRIMARY indicators (Long=green, Short=red, Exit=white) for
    color-differentiated signal markers, plus 1 BOTTOM indicator for cumulative PnL.

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
        self.max_candles = max_candles

        # Core components
        self.aggregator = CandleAggregator(interval_seconds)
        self.buffer = CandleBuffer(max_candles)

        # Strategy (loaded on instrument subscribe)
        self.strategy = None

        # Pending candles produced by trade handler (consumed by interval handler)
        self._pending_candles: list[dict] = []

        # Signal tracking for diff logic
        self._prev_signal_count: int = 0

        # PnL tracking
        self._last_entry: Signal | None = None
        self._cumulative_pnl: float = 0.0

        # Bookmap handles
        self._addon: Any = None
        self._alias: str | None = None
        self._pips: float = 1.0
        # Indicator IDs (resolved async after registration)
        self._indicator_long_id: int | None = None
        self._indicator_short_id: int | None = None
        self._indicator_exit_id: int | None = None
        self._indicator_pnl_id: int | None = None

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
        self._addon = addon

        # Register global handlers
        bm.add_trades_handler(addon, self._on_trade)
        bm.add_on_interval_handler(addon, self._on_interval)
        bm.add_indicator_response_handler(addon, self._on_indicator_response)
        bm.add_on_setting_change_handler(addon, self._on_setting_change)

        # Start addon — Bookmap calls add_instrument_handler when an instrument is subscribed
        bm.start_addon(
            addon,
            add_instrument_handler=self._on_instrument_added,
            detach_instrument_handler=self._on_instrument_detached,
        )

        # Block until Bookmap turns off the addon
        bm.wait_until_addon_is_turned_off(addon)

    # ------------------------------------------------------------------
    # Bookmap event handlers
    # ------------------------------------------------------------------

    def _on_instrument_added(self, addon, alias, full_name, is_crypto, pips,
                              size_multiplier, instrument_multiplier, supported_features):
        """Called when Bookmap subscribes to an instrument."""
        import bookmap as bm

        self._alias = alias
        self._pips = pips

        # Reset all state for new instrument
        self._reset_state()

        # Load strategy
        cls = get_strategy_class(self.strategy_name)
        self.strategy = cls.from_params(self.strategy_params)
        logger.info("Loaded strategy: %s with params %s", self.strategy_name, self.strategy_params)

        # Register settings UI
        bm.add_number_settings_parameter(
            addon, alias, "Candle Interval (sec)",
            default_value=float(self.interval_seconds),
            minimum=10.0, maximum=86400.0, step=10.0,
        )
        bm.add_string_settings_parameter(
            addon, alias, "Strategy Name",
            default_value=self.strategy_name,
        )
        bm.add_number_settings_parameter(
            addon, alias, "Max Candles Buffer",
            default_value=float(self.max_candles),
            minimum=50.0, maximum=5000.0, step=50.0,
        )
        bm.add_string_settings_parameter(
            addon, alias, "Strategy Params JSON",
            default_value=json.dumps(self.strategy_params),
        )

        # Register signal indicators — 3 separate for distinct colors on the heatmap
        bm.register_indicator(
            addon, alias,
            req_id=REQ_SIGNAL_LONG,
            indicator_name=f"MB:{self.strategy_name} Long",
            graph_type="PRIMARY",
            color=COLOR_GREEN,
            line_style="SOLID",
            show_line_by_default=True,
        )
        bm.register_indicator(
            addon, alias,
            req_id=REQ_SIGNAL_SHORT,
            indicator_name=f"MB:{self.strategy_name} Short",
            graph_type="PRIMARY",
            color=COLOR_RED,
            line_style="SOLID",
            show_line_by_default=True,
        )
        bm.register_indicator(
            addon, alias,
            req_id=REQ_SIGNAL_EXIT,
            indicator_name=f"MB:{self.strategy_name} Exit",
            graph_type="PRIMARY",
            color=COLOR_WHITE,
            line_style="SOLID",
            show_line_by_default=True,
        )
        # Cumulative PnL subchart
        bm.register_indicator(
            addon, alias,
            req_id=REQ_PNL_INDICATOR,
            indicator_name="MB:PnL",
            graph_type="BOTTOM",
            color=COLOR_CYAN,
            line_style="SOLID",
            initial_value=0.0,
        )

        # Subscribe to trade data
        bm.subscribe_to_trades(addon, alias, req_id=0)

        logger.info("Subscribed to %s (pips=%s, crypto=%s)", full_name, pips, is_crypto)

    def _on_instrument_detached(self, addon, alias):
        """Called when instrument is unsubscribed."""
        logger.info("Instrument detached: %s", alias)
        candle = self.aggregator.flush()
        if candle:
            self.buffer.append(candle)
        self._alias = None
        self.strategy = None

    def _on_indicator_response(self, addon, req_id, indicator_id):
        """Bookmap confirms indicator registration with the real indicator_id."""
        if req_id == REQ_SIGNAL_LONG:
            self._indicator_long_id = indicator_id
            logger.info("Long indicator registered: id=%d", indicator_id)
        elif req_id == REQ_SIGNAL_SHORT:
            self._indicator_short_id = indicator_id
            logger.info("Short indicator registered: id=%d", indicator_id)
        elif req_id == REQ_SIGNAL_EXIT:
            self._indicator_exit_id = indicator_id
            logger.info("Exit indicator registered: id=%d", indicator_id)
        elif req_id == REQ_PNL_INDICATOR:
            self._indicator_pnl_id = indicator_id
            logger.info("PnL indicator registered: id=%d", indicator_id)

    def _on_setting_change(self, addon, alias, setting_name, field_type, new_value):
        """Called when user changes a setting in Bookmap UI.

        Note: Bookmap's string settings have a bug where defaults show as "1"/"0".
        We ignore bogus values and only apply valid ones.
        """
        logger.debug("Setting changed: %s = %r (%s)", setting_name, new_value, field_type)

        if setting_name == "Candle Interval (sec)":
            val = int(new_value)
            if val < 10:
                return
            self.interval_seconds = val
            self._reset_state()
            logger.info("Interval changed to %ds — buffers reset", self.interval_seconds)

        elif setting_name == "Strategy Name":
            name = str(new_value).strip()
            if len(name) < 3 or "_" not in name:
                # Ignore bogus values like "1" from Bookmap's string default bug
                return
            try:
                cls = get_strategy_class(name)
                self.strategy_name = name
                self.strategy = cls.from_params(self.strategy_params)
                self._prev_signal_count = 0
                self._last_entry = None
                logger.info("Strategy changed to: %s", self.strategy_name)
            except KeyError:
                logger.warning("Unknown strategy: %s — keeping %s", name, self.strategy_name)

        elif setting_name == "Max Candles Buffer":
            val = int(new_value)
            if val < 10:
                return
            self.max_candles = val
            self.buffer = CandleBuffer(self.max_candles)
            self._prev_signal_count = 0

        elif setting_name == "Strategy Params JSON":
            raw = str(new_value).strip()
            if not raw.startswith("{"):
                # Ignore bogus values like "1" from Bookmap's string default bug
                return
            try:
                params = json.loads(raw)
                if not isinstance(params, dict):
                    return
                self.strategy_params = params
                cls = get_strategy_class(self.strategy_name)
                self.strategy = cls.from_params(self.strategy_params)
                self._prev_signal_count = 0
                self._last_entry = None
                logger.info("Strategy params updated: %s", self.strategy_params)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Failed to update params: %s", exc)

    def _on_trade(self, addon, alias, price, size, is_otc, is_bid, is_execution_start,
                  is_execution_end, aggressor_order_id, passive_order_id):
        """Called on every trade tick from Bookmap."""
        if alias != self._alias:
            return

        real_price = price * self._pips
        now_ms = int(time.time() * 1000)
        candle = self.aggregator.add_tick(now_ms, real_price, size)
        if candle is not None:
            self._pending_candles.append(candle)

    def _on_interval(self, addon, alias):
        """Called every ~100ms by Bookmap. Process any finished candles and run strategy."""
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

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_signal(self, signal: Signal) -> None:
        """Draw a signal marker on the Bookmap heatmap and update PnL."""
        import bookmap as bm

        if self._alias is None or self._addon is None:
            return

        price_level = signal.price / self._pips

        # Route to the correct color-coded indicator
        if signal.signal_type == SignalType.ENTRY:
            self._last_entry = signal
            if signal.side == Side.LONG and self._indicator_long_id is not None:
                bm.add_point(self._addon, self._alias, self._indicator_long_id, price_level)
            elif signal.side == Side.SHORT and self._indicator_short_id is not None:
                bm.add_point(self._addon, self._alias, self._indicator_short_id, price_level)

        elif signal.signal_type == SignalType.EXIT:
            if self._indicator_exit_id is not None:
                bm.add_point(self._addon, self._alias, self._indicator_exit_id, price_level)

            # Update cumulative PnL on exit
            if self._last_entry is not None:
                if self._last_entry.side == Side.LONG:
                    pnl = signal.price - self._last_entry.price
                else:
                    pnl = self._last_entry.price - signal.price
                self._cumulative_pnl += pnl
                self._last_entry = None

                if self._indicator_pnl_id is not None:
                    bm.add_point(
                        self._addon, self._alias, self._indicator_pnl_id,
                        self._cumulative_pnl,
                    )

        logger.info(
            "Signal: %s %s @ %.2f (%s) | PnL: %.2f",
            signal.signal_type.value,
            signal.side.value,
            signal.price,
            signal.reason,
            self._cumulative_pnl,
        )

    def _reset_state(self) -> None:
        """Reset all stateful components (on settings change or new instrument)."""
        self.aggregator = CandleAggregator(self.interval_seconds)
        self.buffer = CandleBuffer(self.max_candles)
        self._pending_candles.clear()
        self._prev_signal_count = 0
        self._last_entry = None
        self._cumulative_pnl = 0.0
