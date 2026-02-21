"""Quick-run backtest script with sample data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.strategies.double_ema import DoubleEMA


def generate_sample_data(n_bars: int = 2000) -> pd.DataFrame:
    """Generate synthetic OHLCV data with trend + noise."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Random walk with slight upward drift
    returns = rng.normal(0.0002, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + rng.uniform(0, 0.015, n_bars))
    low = close * (1 - rng.uniform(0, 0.015, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.uniform(1000, 50000, n_bars)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


def main():
    print("Money_Bot — Backtest Runner")
    print("=" * 50)

    # Check for CSV file argument or use sample data
    if len(sys.argv) > 1:
        from money_bot.data.loader import load_data
        csv_path = sys.argv[1]
        print(f"Loading data from {csv_path}...")
        df = load_data(csv_path)
    else:
        print("No CSV provided — using synthetic sample data (2000 bars)")
        df = generate_sample_data()

    print(f"Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    print()

    strategy = DoubleEMA(fast_period=20, slow_period=155)
    config = BacktestConfig(initial_capital=10_000)
    engine = BacktestEngine(config)

    result = engine.run(strategy, df, "sample")

    print(f"Strategy: {result.strategy_name}")
    print(f"Trades: {len(result.trades)}")
    print("-" * 40)
    for key, val in result.metrics.items():
        print(f"  {key:25s}: {val}")
    print("-" * 40)

    if result.trades:
        print(f"\nFirst 5 trades:")
        for t in result.trades[:5]:
            print(f"  {t.side.value:5s} | {t.entry_time} -> {t.exit_time} | PnL: ${t.pnl:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
