"""Backtest page — run strategy and view results."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.data.loader import load_data
from money_bot.types import TradingMode
from money_bot.strategies.registry import get_registry, list_strategy_names
from money_bot.strategies.composite import CompositeStrategy, VotingMode
from webapp.components.strategy_ui import (
    render_strategy_selector,
    render_strategy_params,
    build_strategy_from_ui,
)
from webapp.components.charts import trading_chart, equity_curve_chart, drawdown_chart
from webapp.components.metrics_cards import render_metrics_cards

st.header("Backtest")

# --- Sidebar controls ---
with st.sidebar:
    strategy_mode = st.radio("Strategy Mode", ["Single", "Composite"], horizontal=True)

    if strategy_mode == "Single":
        st.subheader("Strategy")
        selected_name = render_strategy_selector(key_prefix="bt_")
        strat_params = render_strategy_params(selected_name, key_prefix="bt_")
    else:
        st.subheader("Composite Strategy")
        names = list_strategy_names()
        selected_strats = st.multiselect("Strategies", names, default=names[:2] if len(names) >= 2 else names)
        voting = st.selectbox("Voting Mode", [v.value for v in VotingMode])

        sub_params = {}
        for sname in selected_strats:
            with st.expander(sname, expanded=False):
                sub_params[sname] = render_strategy_params(sname, key_prefix=f"comp_{sname}_")

    st.divider()
    st.subheader("Risk Management")
    trading_mode = st.selectbox(
        "Trading Mode",
        [m.value for m in TradingMode],
        format_func=lambda v: v.replace("_", " ").title(),
    )
    sl_pct = st.slider("Stop Loss %", 0.0, 20.0, 0.0, 0.5, help="0 = disabled")
    tp_pct = st.slider("Take Profit %", 0.0, 50.0, 0.0, 1.0, help="0 = disabled")

    st.divider()
    st.subheader("Config")
    initial_capital = st.number_input("Initial Capital ($)", 1000, 1_000_000, 10_000)
    fee_rate = st.number_input("Fee Rate (%)", 0.0, 1.0, 0.1) / 100
    slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.05) / 100
    max_dd = st.slider("Max Drawdown Shutdown %", 5, 50, 25) / 100

    st.subheader("Data")
    data_source = st.text_input("CSV Path or API (e.g. bybit:SOLUSDT:1h)", "")
    uploaded = st.file_uploader("Or upload CSV", type=["csv"])

# --- Main content ---
if st.button("Run Backtest", type="primary"):
    # Load data
    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.lower().str.strip()
            date_col = next((c for c in df.columns if c in ("date", "datetime", "timestamp", "open_time")), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], utc=True)
                df = df.set_index(date_col)
            df.index.name = "datetime"
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[ohlcv].astype(float).sort_index()
        elif data_source:
            df = load_data(data_source)
        else:
            st.warning("Provide a data source or upload a CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Build strategy
    if strategy_mode == "Single":
        strategy = build_strategy_from_ui(selected_name, strat_params)
    else:
        subs = []
        for sname in selected_strats:
            subs.append(build_strategy_from_ui(sname, sub_params[sname]))
        strategy = CompositeStrategy(subs, VotingMode(voting))

    # Build config
    config = BacktestConfig(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_pct=slippage,
        max_drawdown_pct=max_dd,
        stop_loss_pct=sl_pct / 100 if sl_pct > 0 else None,
        take_profit_pct=tp_pct / 100 if tp_pct > 0 else None,
        trading_mode=TradingMode(trading_mode),
    )

    # Run
    engine = BacktestEngine(config)
    with st.spinner("Running backtest..."):
        result = engine.run(strategy, df, dataset_name=data_source or "uploaded")

    # Store in session for MC page
    st.session_state["last_backtest"] = result
    st.session_state["last_data"] = df

    # Display
    render_metrics_cards(result.metrics)

    # Main trading chart: candlesticks + trades + volume + PnL
    st.plotly_chart(trading_chart(df, result.trades, result.signals), use_container_width=True)

    # Drawdown
    st.plotly_chart(drawdown_chart(result.equity_curve), use_container_width=True)

    # Trade list
    if result.trades:
        st.subheader("Trades")
        trade_data = [{
            "Entry": t.entry_time,
            "Exit": t.exit_time,
            "Side": t.side.value,
            "Entry Price": f"{t.entry_price:.4f}",
            "Exit Price": f"{t.exit_price:.4f}",
            "PnL": f"${t.pnl:.2f}",
            "PnL %": f"{t.pnl_pct * 100:.2f}%",
            "Fees": f"${t.fee_paid:.2f}",
            "Reason": t.reason_exit,
        } for t in result.trades]
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
