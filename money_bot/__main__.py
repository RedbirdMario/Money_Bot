"""CLI entry point: python -m money_bot"""

import argparse
import sys
from pathlib import Path

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.data.loader import load_data
from money_bot.strategies.double_ema import DoubleEMA


def main():
    parser = argparse.ArgumentParser(description="Money_Bot Backtesting CLI")
    parser.add_argument("data", help="Data source (CSV path or bybit:SYMBOL:INTERVAL)")
    parser.add_argument("--fast", type=int, default=20, help="Fast EMA period")
    parser.add_argument("--slow", type=int, default=155, help="Slow EMA period")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument("--fee", type=float, default=0.001, help="Fee rate")
    parser.add_argument("--no-flat-filter", action="store_true")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    strategy = DoubleEMA(
        fast_period=args.fast,
        slow_period=args.slow,
        use_flat_filter=not args.no_flat_filter,
    )
    config = BacktestConfig(initial_capital=args.capital, fee_rate=args.fee)
    engine = BacktestEngine(config)

    result = engine.run(strategy, df, args.data)

    print(f"\n{'='*50}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Dataset: {result.dataset_name}")
    print(f"{'='*50}")
    for key, val in result.metrics.items():
        print(f"  {key:25s}: {val}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
