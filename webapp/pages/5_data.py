"""Data management page — load, download, preview."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from money_bot.data.loader import load_data
from webapp.components.charts import ohlcv_chart

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

st.header("Data Management")

tab1, tab2, tab3 = st.tabs(["Upload CSV", "API Download", "Browse Local"])

# --- Tab 1: CSV Upload ---
with tab1:
    uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head(20))

        save_name = st.text_input("Save as (filename)", uploaded.name)
        if st.button("Save to data/raw/"):
            save_path = DATA_DIR / "raw" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            st.success(f"Saved to {save_path}")

# --- Tab 2: API Download ---
with tab2:
    st.subheader("Download from Exchange")
    exchange = st.selectbox("Exchange", ["Bybit", "Binance"])
    symbol = st.text_input("Symbol", "SOLUSDT")
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
    limit = st.number_input("Bars", 100, 10000, 1000)

    if st.button("Download"):
        source = f"{exchange.lower()}:{symbol}:{interval}"
        try:
            with st.spinner(f"Downloading {source}..."):
                df = load_data(source, limit=limit)
            st.success(f"Downloaded {len(df)} bars")
            st.plotly_chart(ohlcv_chart(df, f"{symbol} {interval}"), use_container_width=True)

            # Save option
            fname = f"{symbol}_{interval}_{len(df)}.csv"
            save_path = DATA_DIR / "raw" / fname
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path)
            st.info(f"Auto-saved to {save_path}")
            st.session_state["last_data"] = df
        except Exception as e:
            st.error(f"Download error: {e}")

# --- Tab 3: Browse Local ---
with tab3:
    st.subheader("Local Data Files")
    raw_dir = DATA_DIR / "raw"
    if raw_dir.exists():
        files = list(raw_dir.glob("*.csv"))
        if files:
            selected = st.selectbox("Select file", [f.name for f in files])
            if selected:
                path = raw_dir / selected
                df = load_data(path)
                st.write(f"Shape: {df.shape} | Range: {df.index[0]} — {df.index[-1]}")
                st.plotly_chart(ohlcv_chart(df, selected), use_container_width=True)
                st.session_state["last_data"] = df
        else:
            st.info("No CSV files in data/raw/")
    else:
        st.info("data/raw/ directory not found")
