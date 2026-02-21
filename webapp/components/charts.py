"""Plotly chart helpers for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from money_bot.types import BacktestResult, MonteCarloResult, Trade, Side, SignalType


# ── TradingView-Style Main Chart ────────────────────────────────

def trading_chart(
    df: pd.DataFrame,
    trades: list[Trade],
    signals: list = None,
    title: str = "Backtest",
) -> go.Figure:
    """Candlestick chart with trade markers, position shading, and volume."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=("", "", ""),
    )

    # ── 1. Candlesticks ──
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── 2. Position shading (background) ──
    for trade in trades:
        color = "rgba(38,166,154,0.08)" if trade.side == Side.LONG else "rgba(239,83,80,0.08)"
        border = "rgba(38,166,154,0.3)" if trade.side == Side.LONG else "rgba(239,83,80,0.3)"
        fig.add_vrect(
            x0=trade.entry_time, x1=trade.exit_time,
            fillcolor=color, line=dict(width=0.5, color=border),
            layer="below", row=1, col=1,
        )

    # ── 3. Entry/Exit markers ──
    long_entries_x, long_entries_y = [], []
    long_entries_text = []
    short_entries_x, short_entries_y = [], []
    short_entries_text = []
    long_exits_x, long_exits_y = [], []
    long_exits_text = []
    short_exits_x, short_exits_y = [], []
    short_exits_text = []

    for trade in trades:
        pnl_str = f"${trade.pnl:+.2f}"
        if trade.side == Side.LONG:
            long_entries_x.append(trade.entry_time)
            long_entries_y.append(trade.entry_price)
            long_entries_text.append(f"LONG @ {trade.entry_price:.2f}<br>{trade.reason_entry}")
            long_exits_x.append(trade.exit_time)
            long_exits_y.append(trade.exit_price)
            long_exits_text.append(f"EXIT LONG @ {trade.exit_price:.2f}<br>{trade.reason_exit}<br>{pnl_str}")
        else:
            short_entries_x.append(trade.entry_time)
            short_entries_y.append(trade.entry_price)
            short_entries_text.append(f"SHORT @ {trade.entry_price:.2f}<br>{trade.reason_entry}")
            short_exits_x.append(trade.exit_time)
            short_exits_y.append(trade.exit_price)
            short_exits_text.append(f"EXIT SHORT @ {trade.exit_price:.2f}<br>{trade.reason_exit}<br>{pnl_str}")

    if long_entries_x:
        fig.add_trace(go.Scatter(
            x=long_entries_x, y=long_entries_y,
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=12, color="#26a69a",
                        line=dict(width=1, color="white")),
            text=long_entries_text, hoverinfo="text",
        ), row=1, col=1)

    if short_entries_x:
        fig.add_trace(go.Scatter(
            x=short_entries_x, y=short_entries_y,
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                        line=dict(width=1, color="white")),
            text=short_entries_text, hoverinfo="text",
        ), row=1, col=1)

    if long_exits_x:
        fig.add_trace(go.Scatter(
            x=long_exits_x, y=long_exits_y,
            mode="markers", name="Long Exit",
            marker=dict(symbol="x", size=9, color="#26a69a",
                        line=dict(width=2, color="#26a69a")),
            text=long_exits_text, hoverinfo="text",
        ), row=1, col=1)

    if short_exits_x:
        fig.add_trace(go.Scatter(
            x=short_exits_x, y=short_exits_y,
            mode="markers", name="Short Exit",
            marker=dict(symbol="x", size=9, color="#ef5350",
                        line=dict(width=2, color="#ef5350")),
            text=short_exits_text, hoverinfo="text",
        ), row=1, col=1)

    # ── 4. Volume bars (colored by candle direction) ──
    if "volume" in df.columns:
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"],
                marker_color=colors, marker_line_width=0,
                opacity=0.5, name="Volume", showlegend=False,
            ),
            row=2, col=1,
        )

    # ── 5. Equity curve (bottom panel) ──
    # Build equity on trade closes
    if trades:
        eq_times = [trades[0].entry_time]
        eq_values = [0.0]
        cumulative_pnl = 0.0
        for t in trades:
            cumulative_pnl += t.pnl
            eq_times.append(t.exit_time)
            eq_values.append(cumulative_pnl)

        # Color based on positive/negative
        fig.add_trace(go.Scatter(
            x=eq_times, y=eq_values,
            mode="lines", name="Cumulative PnL",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.1)",
        ), row=3, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                      row=3, col=1)

    # ── Layout ──
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(text=title, font=dict(size=16)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=30),
        xaxis_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
    )

    # Axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="PnL ($)", row=3, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")

    return fig


# ── Drawdown Chart ──────────────────────────────────────────────

def drawdown_chart(equity: pd.Series) -> go.Figure:
    cummax = equity.cummax()
    drawdown = (cummax - equity) / cummax * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", name="Drawdown %",
            line=dict(color="#ef5350", width=1),
            fillcolor="rgba(239,83,80,0.2)",
        )
    )
    fig.update_layout(
        title="Drawdown", height=200,
        yaxis=dict(title="Drawdown %", autorange="reversed",
                   gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


# ── Legacy equity_curve_chart (kept for backwards compat) ───────

def equity_curve_chart(
    equity: pd.Series, trades: list[Trade] | None = None, title: str = "Equity Curve"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity",
                   line=dict(color="#2196F3", width=2))
    )
    fig.update_layout(
        title=title, height=350,
        yaxis=dict(title="Equity ($)", gridcolor="rgba(255,255,255,0.06)"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


# ── Monte Carlo Charts ──────────────────────────────────────────

def mc_fan_chart(mc_result: MonteCarloResult, max_curves: int = 200) -> go.Figure:
    fig = go.Figure()
    curves = mc_result.simulated_equity_curves[:max_curves]

    for i, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(y=curve.values, mode="lines", opacity=0.05,
                       line=dict(color="#2196F3", width=0.5),
                       showlegend=False, hoverinfo="skip")
        )

    all_arrays = [c.values for c in curves]
    max_len = max(len(a) for a in all_arrays)
    padded = np.array([np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in all_arrays])
    median_curve = np.nanmedian(padded, axis=0)
    fig.add_trace(
        go.Scatter(y=median_curve, mode="lines", name="Median",
                   line=dict(color="#FF9800", width=2))
    )

    fig.update_layout(
        title=f"Monte Carlo — {mc_result.n_simulations} Simulations ({mc_result.method})",
        height=450, yaxis=dict(title="Equity ($)"), xaxis=dict(title="Trade #"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


def mc_histogram(values: list[float], title: str, xaxis_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=50, marker_color="#2196F3"))
    fig.update_layout(
        title=title, height=300, xaxis=dict(title=xaxis_title),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


# ── Compare Chart ───────────────────────────────────────────────

def compare_equity_curves(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (name, res) in enumerate(results.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(x=res.equity_curve.index, y=res.equity_curve.values,
                       mode="lines", name=name, line=dict(color=color, width=2))
        )
    fig.update_layout(
        title="Equity Curves Comparison", height=450,
        yaxis=dict(title="Equity ($)"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


# ── OHLCV Chart (standalone) ───────────────────────────────────

def ohlcv_chart(df: pd.DataFrame, title: str = "OHLCV") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLCV",
    ))
    if "volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color="rgba(100,100,100,0.3)", yaxis="y2",
        ))
    fig.update_layout(
        title=title, height=500, xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price"), yaxis2=dict(title="Volume", overlaying="y", side="right"),
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig
