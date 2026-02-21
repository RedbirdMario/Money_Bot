"""Strategy/dataset comparison page."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from money_bot.config import BacktestConfig
from money_bot.core.engine import BacktestEngine
from money_bot.data.loader import load_data
from money_bot.types import TradingMode
from money_bot.strategies.registry import list_strategy_names
from webapp.components.strategy_ui import render_strategy_params, build_strategy_from_ui
from webapp.components.charts import compare_equity_curves

st.header("Strategy Comparison")

st.markdown("Compare different strategies or parameter sets side-by-side.")

# --- Config columns ---
n_configs = st.number_input("Number of configs to compare", 2, 4, 2)

all_names = list_strategy_names()
configs = []

for i in range(n_configs):
    with st.expander(f"Config {i + 1}", expanded=i < 2):
        col1, col2 = st.columns(2)
        with col1:
            label = st.text_input("Label", f"Config {i + 1}", key=f"label_{i}")
            strat_name = st.selectbox("Strategy", all_names, key=f"strat_{i}")
        with col2:
            data_src = st.text_input("Data source", "", key=f"data_{i}")
            capital = st.number_input("Capital ($)", 1000, 1_000_000, 10_000, key=f"cap_{i}")

        # Auto-rendered params for selected strategy
        params = render_strategy_params(strat_name, key_prefix=f"cmp_{i}_")

        trading_mode = st.selectbox(
            "Trading Mode",
            [m.value for m in TradingMode],
            format_func=lambda v: v.replace("_", " ").title(),
            key=f"tm_{i}",
        )
        sl_col, tp_col = st.columns(2)
        with sl_col:
            sl_pct = st.slider("Stop Loss %", 0.0, 20.0, 0.0, 0.5, key=f"sl_{i}")
        with tp_col:
            tp_pct = st.slider("Take Profit %", 0.0, 50.0, 0.0, 1.0, key=f"tp_{i}")

        configs.append({
            "label": label,
            "strategy_name": strat_name,
            "params": params,
            "data": data_src,
            "capital": capital,
            "trading_mode": trading_mode,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
        })

# Use uploaded data or session data as fallback
uploaded = st.file_uploader("Upload shared CSV (for all configs without data source)", type=["csv"])

if st.button("Run Comparison", type="primary"):
    results = {}

    # Load shared data if uploaded
    shared_df = None
    if uploaded:
        shared_df = pd.read_csv(uploaded)
        shared_df.columns = shared_df.columns.str.lower().str.strip()
        date_col = next((c for c in shared_df.columns if c in ("date", "datetime", "timestamp", "open_time")), None)
        if date_col:
            shared_df[date_col] = pd.to_datetime(shared_df[date_col], utc=True)
            shared_df = shared_df.set_index(date_col)
        shared_df.index.name = "datetime"
        if shared_df.index.tz is None:
            shared_df.index = shared_df.index.tz_localize("UTC")
        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in shared_df.columns]
        shared_df = shared_df[ohlcv].astype(float).sort_index()
    elif "last_data" in st.session_state:
        shared_df = st.session_state["last_data"]

    for cfg in configs:
        try:
            if cfg["data"]:
                df = load_data(cfg["data"])
            elif shared_df is not None:
                df = shared_df
            else:
                st.warning(f"No data for {cfg['label']}")
                continue

            strategy = build_strategy_from_ui(cfg["strategy_name"], cfg["params"])
            bt_config = BacktestConfig(
                initial_capital=cfg["capital"],
                stop_loss_pct=cfg["sl_pct"] / 100 if cfg["sl_pct"] > 0 else None,
                take_profit_pct=cfg["tp_pct"] / 100 if cfg["tp_pct"] > 0 else None,
                trading_mode=TradingMode(cfg["trading_mode"]),
            )
            engine = BacktestEngine(bt_config)
            results[cfg["label"]] = engine.run(strategy, df, cfg["label"])
        except Exception as e:
            st.error(f"Error running {cfg['label']}: {e}")

    if results:
        st.plotly_chart(compare_equity_curves(results), use_container_width=True)

        # Metrics comparison table
        st.subheader("Metrics Comparison")
        metrics_data = {name: res.metrics for name, res in results.items()}
        st.dataframe(pd.DataFrame(metrics_data).T, use_container_width=True)
