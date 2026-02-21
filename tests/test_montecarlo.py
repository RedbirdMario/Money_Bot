"""Tests for Monte Carlo simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from money_bot.config import BacktestConfig, MonteCarloConfig
from money_bot.core.engine import BacktestEngine
from money_bot.strategies.double_ema import DoubleEMA
from money_bot.montecarlo.simulator import MonteCarloSimulator


def test_mc_shuffle(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    bt_result = engine.run(strategy, sample_ohlcv, "test")

    mc_config = MonteCarloConfig(n_simulations=50, random_seed=42)
    simulator = MonteCarloSimulator(mc_config)
    mc_result = simulator.run(bt_result, method="shuffle")

    assert mc_result.n_simulations == 50
    assert len(mc_result.simulated_equity_curves) == 50
    assert 0 <= mc_result.ruin_probability <= 1
    assert 0 <= mc_result.curve_fit_score <= 100


def test_mc_bootstrap(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    bt_result = engine.run(strategy, sample_ohlcv, "test")

    mc_config = MonteCarloConfig(n_simulations=50, random_seed=42)
    simulator = MonteCarloSimulator(mc_config)
    mc_result = simulator.run(bt_result, method="bootstrap")

    assert mc_result.method == "bootstrap"
    assert "final_equity" in mc_result.confidence_intervals


def test_mc_noise(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    bt_result = engine.run(strategy, sample_ohlcv, "test")

    mc_config = MonteCarloConfig(n_simulations=50, noise_std=0.1, random_seed=42)
    simulator = MonteCarloSimulator(mc_config)
    mc_result = simulator.run(bt_result, method="noise")

    assert mc_result.method == "noise"
    assert len(mc_result.simulated_metrics) == 50


def test_mc_confidence_intervals(sample_ohlcv):
    strategy = DoubleEMA(fast_period=10, slow_period=50, use_flat_filter=False)
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    bt_result = engine.run(strategy, sample_ohlcv, "test")

    mc_config = MonteCarloConfig(n_simulations=100, random_seed=42)
    simulator = MonteCarloSimulator(mc_config)
    mc_result = simulator.run(bt_result, method="shuffle")

    ci = mc_result.confidence_intervals["final_equity"]
    assert "5%" in ci
    assert "50%" in ci
    assert "95%" in ci
    assert ci["5%"] <= ci["50%"] <= ci["95%"]
