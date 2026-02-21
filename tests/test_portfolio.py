"""Tests for Portfolio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from money_bot.config import BacktestConfig
from money_bot.core.portfolio import Portfolio
from money_bot.types import Signal, SignalType, Side


def test_open_and_close_long():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0)
    portfolio = Portfolio(config)

    signals = [
        Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0, "test_entry"),
        Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.LONG, 110.0, "test_exit"),
    ]
    trades = portfolio.process_signals(signals, pd.DataFrame())

    assert len(trades) == 1
    assert trades[0].side == Side.LONG
    assert trades[0].pnl > 0  # Price went up


def test_open_and_close_short():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.001, slippage_pct=0.0)
    portfolio = Portfolio(config)

    signals = [
        Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.SHORT, 100.0, "test_entry"),
        Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.SHORT, 90.0, "test_exit"),
    ]
    trades = portfolio.process_signals(signals, pd.DataFrame())

    assert len(trades) == 1
    assert trades[0].side == Side.SHORT
    assert trades[0].pnl > 0  # Price went down, short profits


def test_fees_reduce_pnl():
    config_no_fee = BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0)
    config_fee = BacktestConfig(initial_capital=10_000, fee_rate=0.01, slippage_pct=0.0)

    signals = [
        Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0),
        Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.LONG, 110.0),
    ]

    p1 = Portfolio(config_no_fee)
    p1.process_signals(signals, pd.DataFrame())

    p2 = Portfolio(config_fee)
    p2.process_signals(signals, pd.DataFrame())

    assert p1.trades[0].pnl > p2.trades[0].pnl


def test_equity_curve():
    config = BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_pct=0.0)
    portfolio = Portfolio(config)

    signals = [
        Signal(pd.Timestamp("2024-01-01", tz="UTC"), SignalType.ENTRY, Side.LONG, 100.0),
        Signal(pd.Timestamp("2024-01-02", tz="UTC"), SignalType.EXIT, Side.LONG, 110.0),
    ]
    portfolio.process_signals(signals, pd.DataFrame())
    eq = portfolio.get_equity_curve()

    assert len(eq) > 0
    assert eq.iloc[-1] > 10_000  # Should have made money
