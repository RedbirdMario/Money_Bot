"""Quick-run Monte Carlo simulation script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from money_bot.config import BacktestConfig, MonteCarloConfig
from money_bot.core.engine import BacktestEngine
from money_bot.strategies.double_ema import DoubleEMA
from money_bot.montecarlo.simulator import MonteCarloSimulator


def generate_sample_data(n_bars: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h", tz="UTC")
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
    print("Money_Bot — Monte Carlo Simulation")
    print("=" * 50)

    # Run backtest first
    if len(sys.argv) > 1:
        from money_bot.data.loader import load_data
        df = load_data(sys.argv[1])
    else:
        print("Using synthetic sample data")
        df = generate_sample_data()

    strategy = DoubleEMA(fast_period=20, slow_period=155)
    config = BacktestConfig(initial_capital=10_000)
    engine = BacktestEngine(config)
    bt_result = engine.run(strategy, df, "sample")

    print(f"Backtest: {len(bt_result.trades)} trades, Sharpe: {bt_result.metrics['sharpe_ratio']}")
    print()

    # Run MC for each method
    mc_config = MonteCarloConfig(n_simulations=1000, random_seed=42)
    simulator = MonteCarloSimulator(mc_config)

    for method in ["shuffle", "bootstrap", "noise"]:
        print(f"\n--- {method.upper()} ---")
        mc_result = simulator.run(bt_result, method=method)

        print(f"  Ruin Probability:  {mc_result.ruin_probability * 100:.1f}%")
        print(f"  Curve-Fit Score:   {mc_result.curve_fit_score:.1f}%")

        ci = mc_result.confidence_intervals["final_equity"]
        print(f"  Final Equity CI:")
        for level, val in ci.items():
            print(f"    {level:>6s}: ${val:,.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
