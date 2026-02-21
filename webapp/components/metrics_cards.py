"""Streamlit metric card components."""

from __future__ import annotations

import streamlit as st


def render_metrics_cards(metrics: dict[str, float]):
    """Display key metrics as Streamlit metric cards in columns."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", int(metrics.get("total_trades", 0)))
        st.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.1f}%")

    with col2:
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.1f}%")
        st.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")

    with col3:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")

    with col4:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")


def render_mc_summary(ruin_prob: float, curve_fit: float):
    """Display Monte Carlo summary metrics."""
    col1, col2 = st.columns(2)

    with col1:
        color = "🟢" if ruin_prob < 0.1 else "🟡" if ruin_prob < 0.3 else "🔴"
        st.metric("Ruin Probability", f"{ruin_prob * 100:.1f}% {color}")

    with col2:
        color = "🟢" if curve_fit < 50 else "🟡" if curve_fit < 80 else "🔴"
        st.metric("Curve-Fit Score", f"{curve_fit:.1f}% {color}")
        st.caption("< 50% = robust, > 80% = likely overfit")
