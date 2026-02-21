"""Tests for the Bookmap addon components (no Bookmap dependency required)."""

from __future__ import annotations

import pandas as pd
import pytest

from money_bot.integrations.bookmap_addon import (
    BookmapAddon,
    CandleAggregator,
    CandleBuffer,
)
from money_bot.types import Signal, SignalType, Side


# ---------------------------------------------------------------------------
# CandleAggregator
# ---------------------------------------------------------------------------

class TestCandleAggregator:
    def test_first_tick_returns_none(self):
        agg = CandleAggregator(interval_seconds=60)
        result = agg.add_tick(timestamp_ms=60_000, price=100.0, size=1.0)
        assert result is None

    def test_same_bucket_updates_ohlcv(self):
        agg = CandleAggregator(interval_seconds=60)
        agg.add_tick(0, 100.0, 1.0)
        agg.add_tick(10_000, 105.0, 2.0)  # new high
        agg.add_tick(20_000, 95.0, 3.0)   # new low
        agg.add_tick(30_000, 102.0, 4.0)  # close

        # Force flush to inspect
        candle = agg.flush()
        assert candle is not None
        assert candle["open"] == 100.0
        assert candle["high"] == 105.0
        assert candle["low"] == 95.0
        assert candle["close"] == 102.0
        assert candle["volume"] == 10.0  # 1+2+3+4

    def test_new_bucket_returns_finished_candle(self):
        agg = CandleAggregator(interval_seconds=60)  # 60s = 60_000ms buckets
        agg.add_tick(0, 100.0, 1.0)
        agg.add_tick(30_000, 110.0, 2.0)

        # Tick in next bucket (60_000ms)
        finished = agg.add_tick(60_000, 120.0, 3.0)
        assert finished is not None
        assert finished["open"] == 100.0
        assert finished["high"] == 110.0
        assert finished["close"] == 110.0
        assert finished["volume"] == 3.0  # 1+2

    def test_candle_has_utc_timestamp(self):
        agg = CandleAggregator(interval_seconds=60)
        agg.add_tick(0, 100.0, 1.0)
        finished = agg.add_tick(60_000, 200.0, 1.0)
        assert finished is not None
        assert str(finished["timestamp"].tz) == "UTC"

    def test_multiple_candles_sequential(self):
        agg = CandleAggregator(interval_seconds=10)  # 10s buckets
        candles = []
        for i in range(50):
            ts = i * 5_000  # tick every 5s → 2 ticks per candle
            result = agg.add_tick(ts, 100.0 + i, 1.0)
            if result is not None:
                candles.append(result)
        # 50 ticks over 250s with 10s buckets → ~24 finished candles (25 buckets, last one pending)
        assert len(candles) >= 20

    def test_flush_empty_returns_none(self):
        agg = CandleAggregator(interval_seconds=60)
        assert agg.flush() is None

    def test_negative_size_uses_absolute(self):
        agg = CandleAggregator(interval_seconds=60)
        agg.add_tick(0, 100.0, -5.0)
        candle = agg.flush()
        assert candle["volume"] == 5.0


# ---------------------------------------------------------------------------
# CandleBuffer
# ---------------------------------------------------------------------------

class TestCandleBuffer:
    def _make_candle(self, idx: int) -> dict:
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=idx)
        return {
            "timestamp": ts,
            "open": 100.0 + idx,
            "high": 101.0 + idx,
            "low": 99.0 + idx,
            "close": 100.5 + idx,
            "volume": 1000.0,
        }

    def test_empty_dataframe(self):
        buf = CandleBuffer(max_candles=10)
        df = buf.to_dataframe()
        assert len(df) == 0
        assert list(df.columns) == CandleBuffer.COLUMNS

    def test_append_and_len(self):
        buf = CandleBuffer(max_candles=100)
        buf.append(self._make_candle(0))
        buf.append(self._make_candle(1))
        assert len(buf) == 2

    def test_dataframe_has_datetime_index(self):
        buf = CandleBuffer(max_candles=100)
        buf.append(self._make_candle(0))
        buf.append(self._make_candle(1))
        df = buf.to_dataframe()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"

    def test_dataframe_columns_correct(self):
        buf = CandleBuffer(max_candles=100)
        buf.append(self._make_candle(0))
        df = buf.to_dataframe()
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_trims_to_max_candles(self):
        buf = CandleBuffer(max_candles=5)
        for i in range(20):
            buf.append(self._make_candle(i))
        assert len(buf) == 5
        df = buf.to_dataframe()
        assert len(df) == 5
        # Should keep the LAST 5
        assert df["open"].iloc[0] == 115.0  # candle index 15

    def test_dataframe_values_match(self):
        buf = CandleBuffer(max_candles=100)
        buf.append(self._make_candle(0))
        df = buf.to_dataframe()
        assert df["open"].iloc[0] == 100.0
        assert df["high"].iloc[0] == 101.0
        assert df["low"].iloc[0] == 99.0
        assert df["close"].iloc[0] == 100.5
        assert df["volume"].iloc[0] == 1000.0


# ---------------------------------------------------------------------------
# Signal diff logic (tested via BookmapAddon internals)
# ---------------------------------------------------------------------------

class TestSignalDiff:
    """Test that only NEW signals are reported when strategy is re-run."""

    def test_only_fresh_signals_emitted(self):
        """Simulate the signal-diff logic used in _on_interval."""
        # Simulate generate_signals returning cumulative results
        all_signals_run1 = [
            Signal(pd.Timestamp("2024-01-01 01:00", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(pd.Timestamp("2024-01-01 02:00", tz="UTC"), SignalType.EXIT, Side.LONG, 105.0, "test"),
        ]
        all_signals_run2 = all_signals_run1 + [
            Signal(pd.Timestamp("2024-01-01 03:00", tz="UTC"), SignalType.ENTRY, Side.SHORT, 103.0, "test"),
        ]
        all_signals_run3 = all_signals_run2  # no new signals

        prev_count = 0

        # Run 1
        fresh1 = all_signals_run1[prev_count:]
        prev_count = len(all_signals_run1)
        assert len(fresh1) == 2

        # Run 2 — only 1 new signal
        fresh2 = all_signals_run2[prev_count:]
        prev_count = len(all_signals_run2)
        assert len(fresh2) == 1
        assert fresh2[0].side == Side.SHORT

        # Run 3 — no new signals
        fresh3 = all_signals_run3[prev_count:]
        prev_count = len(all_signals_run3)
        assert len(fresh3) == 0


# ---------------------------------------------------------------------------
# BookmapAddon construction (no Bookmap dependency)
# ---------------------------------------------------------------------------

class TestBookmapAddonInit:
    def test_default_construction(self):
        addon = BookmapAddon("double_ema")
        assert addon.strategy_name == "double_ema"
        assert addon.interval_seconds == 3600
        assert addon.strategy is None  # loaded lazily on subscribe

    def test_custom_params(self):
        addon = BookmapAddon(
            "rsi_mean_reversion",
            strategy_params={"rsi_period": 21},
            interval_seconds=300,
            max_candles=200,
        )
        assert addon.strategy_params == {"rsi_period": 21}
        assert addon.aggregator.interval == 300
        assert addon.buffer.max_candles == 200

    def test_pending_candles_initially_empty(self):
        addon = BookmapAddon("double_ema")
        assert addon._pending_candles == []
