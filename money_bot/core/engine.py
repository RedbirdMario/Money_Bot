"""Backtesting engine — orchestrates strategy, portfolio, metrics."""

from __future__ import annotations

import pandas as pd

from money_bot.config import BacktestConfig
from money_bot.core.portfolio import Portfolio
from money_bot.core.metrics import MetricsCalculator
from money_bot.strategies.base import Strategy
from money_bot.types import BacktestResult


class BacktestEngine:
    """Run a strategy on OHLCV data and produce a BacktestResult."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        dataset_name: str = "",
    ) -> BacktestResult:
        """Execute backtest on full dataset."""
        return self._run_on(strategy, data, dataset_name)

    def run_train_test(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        dataset_name: str = "",
    ) -> tuple[BacktestResult, BacktestResult]:
        """Run backtest split into train and test sets."""
        split_idx = int(len(data) * self.config.train_test_split)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        train_result = self._run_on(strategy, train_data, f"{dataset_name}_train")
        test_result = self._run_on(strategy, test_data, f"{dataset_name}_test")

        return train_result, test_result

    def run_multi(
        self,
        strategy: Strategy,
        datasets: dict[str, pd.DataFrame],
    ) -> dict[str, BacktestResult]:
        """Run backtest on multiple datasets."""
        return {
            name: self._run_on(strategy, data, name)
            for name, data in datasets.items()
        }

    def _run_on(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        dataset_name: str,
    ) -> BacktestResult:
        signals = strategy.generate_signals(data)
        portfolio = Portfolio(self.config)
        trades = portfolio.process_signals(signals, data)
        equity_curve = portfolio.get_equity_curve()

        # Determine periods_per_year from data frequency
        periods_per_year = self._estimate_periods_per_year(data)

        metrics = MetricsCalculator.calculate(
            trades, equity_curve, self.config.initial_capital, periods_per_year
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            signals=signals,
            metrics=metrics,
            config=self.config,
            strategy_name=strategy.name,
            dataset_name=dataset_name,
        )

    @staticmethod
    def _estimate_periods_per_year(data: pd.DataFrame) -> float:
        if len(data) < 2:
            return 252
        median_delta = pd.Series(data.index).diff().median()
        seconds = median_delta.total_seconds()
        if seconds <= 0:
            return 252
        year_seconds = 365.25 * 24 * 3600
        return year_seconds / seconds
