"""Tests for new strategies, registry, composite, SL/TP, and trading mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.core.portfolio import Portfolio
from money_bot.types import Signal, SignalType, Side, TradingMode
from money_bot.strategies.base import ParamType
from money_bot.strategies.registry import get_registry, list_strategy_names, get_strategy_class
from money_bot.strategies.composite import CompositeStrategy, VotingMode


# ── Registry ────────────────────────────────────────────────────

class TestRegistry:
    def test_all_strategies_registered(self):
        names = list_strategy_names()
        expected = {
            "double_ema", "rsi_mean_reversion", "bollinger_breakout",
            "macd_histogram", "donchian_breakout", "stochastic_ema", "vwap_volume",
        }
        assert expected.issubset(set(names))

    def test_get_strategy_class(self):
        cls = get_strategy_class("double_ema")
        assert cls.name == "double_ema"

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            get_strategy_class("nonexistent")

    def test_all_strategies_have_params(self):
        for name, cls in get_registry().items():
            assert hasattr(cls, "PARAMS")
            assert isinstance(cls.PARAMS, list)

    def test_all_strategies_have_description(self):
        for name, cls in get_registry().items():
            assert cls.description, f"{name} is missing description"

    def test_from_params_creates_instance(self):
        for name, cls in get_registry().items():
            defaults = {p.name: p.default for p in cls.PARAMS}
            instance = cls.from_params(defaults)
            assert instance.name == name


# ── Individual Strategies ───────────────────────────────────────

class TestStrategiesProduceSignals:
    """Each strategy should produce at least some signals on trending data."""

    @pytest.fixture
    def trending_data(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        trend = np.linspace(0, 0.5, n)
        noise = rng.normal(0, 0.003, n).cumsum()
        close = 100 * np.exp(trend + noise)
        high = close * (1 + rng.uniform(0.002, 0.015, n))
        low = close * (1 - rng.uniform(0.002, 0.015, n))
        open_ = close * (1 + rng.normal(0, 0.003, n))
        volume = rng.uniform(5000, 100000, n)
        return pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }, index=dates)

    @pytest.mark.parametrize("name", [
        "double_ema", "rsi_mean_reversion", "bollinger_breakout",
        "macd_histogram", "donchian_breakout", "vwap_volume",
    ])
    def test_strategy_generates_signals(self, name, trending_data):
        cls = get_strategy_class(name)
        defaults = {p.name: p.default for p in cls.PARAMS}
        strategy = cls.from_params(defaults)
        signals = strategy.generate_signals(trending_data)
        assert len(signals) > 0, f"{name} produced no signals"

    def test_stochastic_ema_generates_signals_on_volatile_data(self):
        """Stochastic+EMA needs deep pullbacks — use strongly oscillating data."""
        rng = np.random.default_rng(99)
        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        # Strong oscillations to push stochastic into extreme zones
        trend = np.linspace(0, 0.1, n)
        cycles = 0.15 * np.sin(np.linspace(0, 30 * np.pi, n))
        noise = rng.normal(0, 0.003, n).cumsum()
        close = 100 * np.exp(trend + cycles + noise)
        high = close * (1 + rng.uniform(0.01, 0.03, n))
        low = close * (1 - rng.uniform(0.01, 0.03, n))
        open_ = close * (1 + rng.normal(0, 0.005, n))
        volume = rng.uniform(5000, 100000, n)
        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }, index=dates)
        cls = get_strategy_class("stochastic_ema")
        # Use wider zones (30/70) for easier triggering
        strategy = cls.from_params({"stoch_k": 14, "stoch_d": 3, "ema_period": 20,
                                     "oversold": 30, "overbought": 70})
        signals = strategy.generate_signals(df)
        assert len(signals) > 0, "stochastic_ema should signal on volatile data"

    @pytest.mark.parametrize("name", [
        "double_ema", "rsi_mean_reversion", "bollinger_breakout",
        "macd_histogram", "donchian_breakout", "stochastic_ema", "vwap_volume",
    ])
    def test_strategy_backtest_runs(self, name, trending_data):
        cls = get_strategy_class(name)
        defaults = {p.name: p.default for p in cls.PARAMS}
        strategy = cls.from_params(defaults)
        engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
        result = engine.run(strategy, trending_data, f"test_{name}")
        assert result.strategy_name == name
        assert len(result.equity_curve) > 0

    def test_signals_are_well_formed(self, trending_data):
        """All signals should have valid types, sides, and positive prices."""
        for name, cls in get_registry().items():
            defaults = {p.name: p.default for p in cls.PARAMS}
            strategy = cls.from_params(defaults)
            signals = strategy.generate_signals(trending_data)
            for sig in signals:
                assert isinstance(sig.signal_type, SignalType)
                assert isinstance(sig.side, Side)
                assert sig.price > 0


# ── Composite Strategy ──────────────────────────────────────────

class TestCompositeStrategy:
    @pytest.fixture
    def trending_data(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        trend = np.linspace(0, 0.5, n)
        noise = rng.normal(0, 0.003, n).cumsum()
        close = 100 * np.exp(trend + noise)
        high = close * (1 + rng.uniform(0.002, 0.015, n))
        low = close * (1 - rng.uniform(0.002, 0.015, n))
        open_ = close * (1 + rng.normal(0, 0.003, n))
        volume = rng.uniform(5000, 100000, n)
        return pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        }, index=dates)

    def test_composite_majority(self, trending_data):
        ema = get_strategy_class("double_ema").from_params({"fast_period": 10, "slow_period": 50})
        macd = get_strategy_class("macd_histogram").from_params({})
        rsi = get_strategy_class("rsi_mean_reversion").from_params({})

        composite = CompositeStrategy([ema, macd, rsi], VotingMode.MAJORITY)
        signals = composite.generate_signals(trending_data)
        assert isinstance(signals, list)

    def test_composite_unanimous(self, trending_data):
        ema = get_strategy_class("double_ema").from_params({"fast_period": 10, "slow_period": 50})
        macd = get_strategy_class("macd_histogram").from_params({})

        composite = CompositeStrategy([ema, macd], VotingMode.UNANIMOUS)
        signals = composite.generate_signals(trending_data)
        assert isinstance(signals, list)

    def test_composite_any(self, trending_data):
        ema = get_strategy_class("double_ema").from_params({"fast_period": 10, "slow_period": 50})
        boll = get_strategy_class("bollinger_breakout").from_params({})

        composite = CompositeStrategy([ema, boll], VotingMode.ANY)
        signals = composite.generate_signals(trending_data)
        assert len(signals) > 0  # ANY should produce signals if either does

    def test_composite_backtest(self, trending_data):
        ema = get_strategy_class("double_ema").from_params({"fast_period": 10, "slow_period": 50})
        macd = get_strategy_class("macd_histogram").from_params({})

        composite = CompositeStrategy([ema, macd], VotingMode.MAJORITY)
        engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
        result = engine.run(composite, trending_data, "composite_test")
        assert len(result.equity_curve) > 0

    def test_empty_sub_strategies(self, trending_data):
        composite = CompositeStrategy([], VotingMode.MAJORITY)
        signals = composite.generate_signals(trending_data)
        assert signals == []


# ── Stop-Loss / Take-Profit ─────────────────────────────────────

class TestStopLossTakeProfit:
    def _make_bars(self, prices):
        """Create a simple DataFrame from close prices with synthetic OHLCV."""
        dates = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
        return pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }, index=dates)

    def test_stop_loss_long(self):
        prices = [100, 100, 95, 94, 93, 96]  # Drops below 5% SL
        bars = self._make_bars(prices)

        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            stop_loss_pct=0.05,
        )
        portfolio = Portfolio(config)
        signals = [
            Signal(bars.index[0], SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(bars.index[-1], SignalType.EXIT, Side.LONG, 96.0, "test_exit"),
        ]
        trades = portfolio.process_signals(signals, bars)
        assert len(trades) >= 1
        assert trades[0].reason_exit == "stop_loss"

    def test_take_profit_long(self):
        prices = [100, 105, 112, 115, 110]  # Rises above 10% TP
        bars = self._make_bars(prices)

        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            take_profit_pct=0.10,
        )
        portfolio = Portfolio(config)
        signals = [
            Signal(bars.index[0], SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(bars.index[-1], SignalType.EXIT, Side.LONG, 110.0, "test_exit"),
        ]
        trades = portfolio.process_signals(signals, bars)
        assert len(trades) >= 1
        assert trades[0].reason_exit == "take_profit"

    def test_stop_loss_short(self):
        prices = [100, 102, 106, 108]  # Rises above 5% SL for short
        bars = self._make_bars(prices)

        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            stop_loss_pct=0.05,
        )
        portfolio = Portfolio(config)
        signals = [
            Signal(bars.index[0], SignalType.ENTRY, Side.SHORT, 100.0, "test"),
            Signal(bars.index[-1], SignalType.EXIT, Side.SHORT, 108.0, "test_exit"),
        ]
        trades = portfolio.process_signals(signals, bars)
        assert len(trades) >= 1
        assert trades[0].reason_exit == "stop_loss"

    def test_no_sl_tp_when_disabled(self):
        prices = [100, 80, 120]
        bars = self._make_bars(prices)

        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            stop_loss_pct=None, take_profit_pct=None,
        )
        portfolio = Portfolio(config)
        signals = [
            Signal(bars.index[0], SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(bars.index[-1], SignalType.EXIT, Side.LONG, 120.0, "test_exit"),
        ]
        trades = portfolio.process_signals(signals, bars)
        assert len(trades) == 1
        assert trades[0].reason_exit == "test_exit"


# ── Trading Mode ────────────────────────────────────────────────

class TestTradingMode:
    def test_long_only_skips_short(self):
        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            trading_mode=TradingMode.LONG_ONLY,
        )
        portfolio = Portfolio(config)

        signals = [
            Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.SHORT, 100.0, "test"),
            Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.SHORT, 90.0, "test"),
            Signal(pd.Timestamp("2024-01-03", tz="UTC"), SignalType.ENTRY, Side.LONG, 95.0, "test"),
            Signal(pd.Timestamp("2024-01-04", tz="UTC"), SignalType.EXIT, Side.LONG, 100.0, "test"),
        ]
        trades = portfolio.process_signals(signals, pd.DataFrame())
        assert len(trades) == 1
        assert trades[0].side == Side.LONG

    def test_short_only_skips_long(self):
        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            trading_mode=TradingMode.SHORT_ONLY,
        )
        portfolio = Portfolio(config)

        signals = [
            Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.LONG, 110.0, "test"),
            Signal(pd.Timestamp("2024-01-03", tz="UTC"), SignalType.ENTRY, Side.SHORT, 105.0, "test"),
            Signal(pd.Timestamp("2024-01-04", tz="UTC"), SignalType.EXIT, Side.SHORT, 95.0, "test"),
        ]
        trades = portfolio.process_signals(signals, pd.DataFrame())
        assert len(trades) == 1
        assert trades[0].side == Side.SHORT

    def test_both_allows_all(self):
        config = BacktestConfig(
            initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0,
            trading_mode=TradingMode.BOTH,
        )
        portfolio = Portfolio(config)

        signals = [
            Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0, "test"),
            Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.LONG, 110.0, "test"),
            Signal(pd.Timestamp("2024-01-03", tz="UTC"), SignalType.ENTRY, Side.SHORT, 105.0, "test"),
            Signal(pd.Timestamp("2024-01-04", tz="UTC"), SignalType.EXIT, Side.SHORT, 95.0, "test"),
        ]
        trades = portfolio.process_signals(signals, pd.DataFrame())
        assert len(trades) == 2


# ── ParamDescriptor ─────────────────────────────────────────────

class TestParamDescriptor:
    def test_param_types_valid(self):
        for name, cls in get_registry().items():
            for p in cls.PARAMS:
                assert isinstance(p.param_type, ParamType)
                if p.param_type in (ParamType.INT, ParamType.FLOAT):
                    assert p.min_val is not None, f"{name}.{p.name} missing min_val"
                    assert p.max_val is not None, f"{name}.{p.name} missing max_val"
                if p.param_type == ParamType.SELECT:
                    assert len(p.options) > 0, f"{name}.{p.name} missing options"
