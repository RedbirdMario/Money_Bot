"""Performance metrics calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from money_bot.types import Trade


class MetricsCalculator:
    """Calculate trading performance metrics from trades and equity curve."""

    @staticmethod
    def calculate(
        trades: list[Trade],
        equity_curve: pd.Series,
        initial_capital: float = 10_000.0,
        periods_per_year: float = 252,
    ) -> dict[str, float]:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade_duration": 0.0,
                "total_fees": 0.0,
            }

        pnls = np.array([t.pnl for t in trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        total_pnl = float(pnls.sum())
        win_rate = len(winners) / len(pnls) if len(pnls) > 0 else 0.0
        avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss = float(losers.mean()) if len(losers) > 0 else 0.0
        profit_factor = (
            float(winners.sum() / abs(losers.sum()))
            if len(losers) > 0 and losers.sum() != 0
            else float("inf") if len(winners) > 0 else 0.0
        )
        expectancy = float(pnls.mean()) if len(pnls) > 0 else 0.0

        # Returns from equity curve
        returns = equity_curve.pct_change().dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()

        # Sharpe Ratio (annualized)
        sharpe = (
            float(mean_ret / std_ret * np.sqrt(periods_per_year))
            if std_ret > 0
            else 0.0
        )

        # Sortino Ratio
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        sortino = (
            float(mean_ret / downside_std * np.sqrt(periods_per_year))
            if downside_std > 0
            else 0.0
        )

        # Max Drawdown
        cummax = equity_curve.cummax()
        drawdown = (cummax - equity_curve) / cummax
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        # Calmar Ratio
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) if len(equity_curve) > 0 else 0.0
        calmar = float(total_return / max_dd) if max_dd > 0 else 0.0

        # Avg trade duration in hours
        durations = [t.duration.total_seconds() / 3600 for t in trades]
        avg_duration = float(np.mean(durations)) if durations else 0.0

        total_fees = float(sum(t.fee_paid for t in trades))

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(float(total_return * 100), 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "profit_factor": round(profit_factor, 3),
            "expectancy": round(expectancy, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_trade_duration": round(avg_duration, 1),
            "total_fees": round(total_fees, 2),
        }
