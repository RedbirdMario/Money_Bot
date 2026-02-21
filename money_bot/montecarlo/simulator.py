"""Monte Carlo simulator orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from money_bot.config import MonteCarloConfig
from money_bot.core.metrics import MetricsCalculator
from money_bot.types import BacktestResult, MonteCarloResult, Trade
from money_bot.montecarlo.methods import (
    shuffle_trades,
    bootstrap_trades,
    noise_injection,
    build_equity_from_pnls,
)
from money_bot.montecarlo.analyzer import (
    compute_confidence_intervals,
    compute_ruin_probability,
    compute_curve_fit_score,
)


class MonteCarloSimulator:
    """Run Monte Carlo simulations on backtest results."""

    def __init__(self, config: MonteCarloConfig | None = None):
        self.config = config or MonteCarloConfig()

    def run(
        self,
        result: BacktestResult,
        method: str = "shuffle",
    ) -> MonteCarloResult:
        """Run MC simulation using specified method."""
        if not result.trades:
            raise ValueError("No trades to simulate")

        rng = np.random.default_rng(self.config.random_seed)
        initial_capital = result.config.initial_capital if result.config else 10_000.0

        method_fn = {
            "shuffle": lambda t, r: shuffle_trades(t, r),
            "bootstrap": lambda t, r: bootstrap_trades(t, r),
            "noise": lambda t, r: noise_injection(t, r, self.config.noise_std),
        }[method]

        sim_equity_curves: list[pd.Series] = []
        sim_metrics_list: list[dict[str, float]] = []

        for _ in range(self.config.n_simulations):
            pnls = method_fn(result.trades, rng)
            equity = build_equity_from_pnls(pnls, initial_capital)
            sim_equity_curves.append(equity)

            # Compute basic metrics from the simulated equity curve
            returns = equity.pct_change().dropna()
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            cummax = equity.cummax()
            dd = (cummax - equity) / cummax
            max_dd = float(dd.max())

            final_equity = equity.iloc[-1]
            total_return = (final_equity / initial_capital - 1) * 100

            sim_metrics_list.append({
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "final_equity": round(final_equity, 2),
                "total_return_pct": round(total_return, 2),
            })

        # Aggregate analysis
        final_equities = [m["final_equity"] for m in sim_metrics_list]
        sharpes = [m["sharpe_ratio"] for m in sim_metrics_list]
        drawdowns = [m["max_drawdown_pct"] for m in sim_metrics_list]

        ci = {
            "final_equity": compute_confidence_intervals(
                final_equities, self.config.confidence_levels
            ),
            "sharpe_ratio": compute_confidence_intervals(
                sharpes, self.config.confidence_levels
            ),
            "max_drawdown_pct": compute_confidence_intervals(
                drawdowns, self.config.confidence_levels
            ),
        }

        ruin_prob = compute_ruin_probability(final_equities, initial_capital)

        original_sharpe = result.metrics.get("sharpe_ratio", 0.0)
        curve_fit = compute_curve_fit_score(original_sharpe, sharpes)

        return MonteCarloResult(
            original_metrics=result.metrics,
            simulated_equity_curves=sim_equity_curves,
            simulated_metrics=sim_metrics_list,
            confidence_intervals=ci,
            ruin_probability=ruin_prob,
            curve_fit_score=curve_fit,
            n_simulations=self.config.n_simulations,
            method=method,
        )

    def run_all_methods(
        self, result: BacktestResult
    ) -> dict[str, MonteCarloResult]:
        """Run all configured MC methods."""
        return {
            method: self.run(result, method) for method in self.config.methods
        }
