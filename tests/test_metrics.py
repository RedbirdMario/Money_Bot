"""Tests for MetricsCalculator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from money_bot.core.metrics import MetricsCalculator
from money_bot.types import Trade, Side


def _make_trades(pnls: list[float]) -> list[Trade]:
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append(Trade(
            entry_time=pd.Timestamp(f"2024-01-{i + 1:02d}", tz="UTC"),
            exit_time=pd.Timestamp(f"2024-01-{i + 1:02d} 12:00", tz="UTC"),
            side=Side.LONG,
            entry_price=100.0,
            exit_price=100.0 + pnl,
            size=1.0,
            pnl=pnl,
            pnl_pct=pnl / 100.0,
            fee_paid=0.1,
        ))
    return trades


def _make_equity(pnls: list[float], initial: float = 10_000) -> pd.Series:
    values = [initial]
    for pnl in pnls:
        values.append(values[-1] + pnl)
    return pd.Series(values)


def test_empty_trades():
    metrics = MetricsCalculator.calculate([], pd.Series(dtype=float))
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0


def test_all_winners():
    pnls = [100, 200, 50, 150]
    trades = _make_trades(pnls)
    equity = _make_equity(pnls)
    metrics = MetricsCalculator.calculate(trades, equity)

    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 1.0
    assert metrics["total_pnl"] == 500.0
    assert metrics["profit_factor"] == float("inf")


def test_mixed_trades():
    pnls = [100, -50, 200, -30, 80]
    trades = _make_trades(pnls)
    equity = _make_equity(pnls)
    metrics = MetricsCalculator.calculate(trades, equity)

    assert metrics["total_trades"] == 5
    assert 0 < metrics["win_rate"] < 1
    assert metrics["total_pnl"] == 300.0
    assert metrics["profit_factor"] > 1.0
    assert metrics["max_drawdown_pct"] >= 0


def test_all_losers():
    pnls = [-100, -50, -200]
    trades = _make_trades(pnls)
    equity = _make_equity(pnls)
    metrics = MetricsCalculator.calculate(trades, equity)

    assert metrics["win_rate"] == 0.0
    assert metrics["total_pnl"] == -350.0
    assert metrics["profit_factor"] == 0.0
