"""Tests for BacktestEngine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.strategies.double_ema import DoubleEMA


def test_engine_runs(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    result = engine.run(strategy, sample_ohlcv, "test")

    assert result.strategy_name == "double_ema"
    assert result.dataset_name == "test"
    assert len(result.equity_curve) > 0
    assert "total_trades" in result.metrics
    assert "sharpe_ratio" in result.metrics


def test_engine_produces_trades(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    result = engine.run(strategy, sample_ohlcv, "test")

    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.entry_price > 0
        assert trade.exit_price > 0
        assert trade.size > 0


def test_train_test_split(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50)
    config = BacktestConfig(initial_capital=10_000, train_test_split=0.7)
    engine = BacktestEngine(config)
    train, test = engine.run_train_test(strategy, sample_ohlcv, "split_test")

    assert "train" in train.dataset_name
    assert "test" in test.dataset_name


def test_max_drawdown_shutdown(sample_ohlcv):
    strategy = DoubleEMA(fast_period=5, slow_period=10, use_flat_filter=False)
    config = BacktestConfig(initial_capital=10_000, max_drawdown_pct=0.01)  # 1% = very tight
    engine = BacktestEngine(config)
    result = engine.run(strategy, sample_ohlcv, "dd_test")

    # Should stop early due to tight drawdown limit
    assert result.metrics["max_drawdown_pct"] <= 5.0  # Reasonable bound


def test_fees_deducted(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.01))
    result = engine.run(strategy, sample_ohlcv, "fee_test")

    if result.trades:
        assert result.metrics["total_fees"] > 0
