"""Monte Carlo analysis — confidence intervals, ruin probability, curve-fit score."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_confidence_intervals(
    metric_values: list[float],
    levels: list[float],
) -> dict[str, float]:
    """Compute percentile-based confidence intervals."""
    arr = np.array(metric_values)
    result = {}
    for level in levels:
        pct_label = f"{int(level * 100)}%"
        result[pct_label] = float(np.percentile(arr, level * 100))
    result["mean"] = float(arr.mean())
    result["std"] = float(arr.std())
    return result


def compute_ruin_probability(
    final_equities: list[float], initial_capital: float
) -> float:
    """P(equity falls below initial capital) across simulations."""
    arr = np.array(final_equities)
    return float((arr < initial_capital).sum() / len(arr))


def compute_curve_fit_score(
    original_sharpe: float, simulated_sharpes: list[float]
) -> float:
    """Percentile rank of original Sharpe among simulated Sharpes.

    High score (>80%) suggests overfitting — original result is unusually good.
    Low score (<50%) means the strategy is robust.
    """
    arr = np.array(simulated_sharpes)
    return float((arr < original_sharpe).sum() / len(arr) * 100)
