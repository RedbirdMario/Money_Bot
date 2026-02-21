"""Monte Carlo simulation page."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from money_bot.config import MonteCarloConfig
from money_bot.montecarlo.simulator import MonteCarloSimulator
from webapp.components.charts import mc_fan_chart, mc_histogram
from webapp.components.metrics_cards import render_mc_summary

st.header("Monte Carlo Simulation")

if "last_backtest" not in st.session_state:
    st.info("Run a backtest first on the Backtest page.")
    st.stop()

result = st.session_state["last_backtest"]
st.success(f"Using backtest: {result.strategy_name} on {result.dataset_name} ({len(result.trades)} trades)")

# Controls
with st.sidebar:
    st.subheader("MC Settings")
    n_sims = st.slider("Simulations", 100, 5000, 1000, step=100)
    method = st.selectbox("Method", ["shuffle", "bootstrap", "noise"])
    noise_std = st.slider("Noise Std (for noise method)", 0.01, 0.5, 0.1) if method == "noise" else 0.1
    seed = st.number_input("Random Seed", 0, 99999, 42)

if st.button("Run Monte Carlo", type="primary"):
    mc_config = MonteCarloConfig(
        n_simulations=n_sims,
        noise_std=noise_std,
        random_seed=seed,
    )
    simulator = MonteCarloSimulator(mc_config)

    with st.spinner(f"Running {n_sims} {method} simulations..."):
        mc_result = simulator.run(result, method=method)

    st.session_state["last_mc"] = mc_result

    # Summary
    render_mc_summary(mc_result.ruin_probability, mc_result.curve_fit_score)

    # Fan chart
    st.plotly_chart(mc_fan_chart(mc_result), use_container_width=True)

    # Histograms
    col1, col2 = st.columns(2)
    with col1:
        equities = [m["final_equity"] for m in mc_result.simulated_metrics]
        st.plotly_chart(mc_histogram(equities, "Final Equity Distribution", "Equity ($)"),
                        use_container_width=True)
    with col2:
        sharpes = [m["sharpe_ratio"] for m in mc_result.simulated_metrics]
        st.plotly_chart(mc_histogram(sharpes, "Sharpe Ratio Distribution", "Sharpe"),
                        use_container_width=True)

    # Confidence intervals table
    st.subheader("Confidence Intervals")
    ci_data = []
    for metric_name, intervals in mc_result.confidence_intervals.items():
        row = {"Metric": metric_name}
        row.update(intervals)
        ci_data.append(row)
    st.dataframe(pd.DataFrame(ci_data), use_container_width=True)
