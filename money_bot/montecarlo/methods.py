"""Monte Carlo simulation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from money_bot.types import Trade


def shuffle_trades(trades: list[Trade], rng: np.random.Generator) -> list[float]:
    """Shuffle trade PnL order — tests path dependency."""
    pnls = [t.pnl for t in trades]
    rng.shuffle(pnls)
    return pnls


def bootstrap_trades(trades: list[Trade], rng: np.random.Generator) -> list[float]:
    """Resample trades with replacement — tests sampling variability."""
    pnls = [t.pnl for t in trades]
    indices = rng.integers(0, len(pnls), size=len(pnls))
    return [pnls[i] for i in indices]


def noise_injection(
    trades: list[Trade], rng: np.random.Generator, noise_std: float = 0.1
) -> list[float]:
    """Add Gaussian noise to trade PnL — tests robustness."""
    pnls = np.array([t.pnl for t in trades])
    pnl_std = pnls.std()
    if pnl_std == 0:
        return pnls.tolist()
    noise = rng.normal(0, pnl_std * noise_std, size=len(pnls))
    return (pnls + noise).tolist()


def build_equity_from_pnls(
    pnls: list[float], initial_capital: float
) -> pd.Series:
    """Convert a PnL sequence to an equity curve."""
    equity = [initial_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    return pd.Series(equity, name="equity")
