"""Money_Bot — Streamlit Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Money_Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Money_Bot")
st.markdown("Backtesting Engine + Monte Carlo Stress-Testing")

st.markdown("""
### Pages

- **Backtest** — Run a strategy on historical data
- **Monte Carlo** — Stress-test with MC simulations
- **Journal** — Trading diary (SQLite)
- **Compare** — Side-by-side strategy comparison
- **Data** — Load & manage datasets
""")

st.info("Use the sidebar to navigate between pages.")
