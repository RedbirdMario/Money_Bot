"""Trading journal page — SQLite-based."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from money_bot.journal.db import JournalDB

st.header("Trading Journal")

db = JournalDB()

# --- Log new entry ---
with st.expander("Add Entry", expanded=False):
    with st.form("journal_entry"):
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.text_input("Strategy", "double_ema")
            dataset = st.text_input("Dataset", "")
            tags = st.text_input("Tags (comma-separated)", "")
        with col2:
            result_type = st.selectbox("Type", ["backtest", "demo", "live-test", "note"])
            sharpe = st.number_input("Sharpe", value=0.0, format="%.3f")
            max_dd = st.number_input("Max DD %", value=0.0, format="%.2f")

        notes = st.text_area("Notes / Learnings")
        params_str = st.text_input("Parameters (JSON)", "{}")

        if st.form_submit_button("Save Entry"):
            db.add_entry(
                strategy=strategy, dataset=dataset, entry_type=result_type,
                tags=tags, sharpe=sharpe, max_dd=max_dd, notes=notes,
                params=params_str,
            )
            st.success("Entry saved!")
            st.rerun()

# --- Auto-log from last backtest ---
if "last_backtest" in st.session_state:
    bt = st.session_state["last_backtest"]
    if st.button("Log Last Backtest"):
        import json
        db.add_entry(
            strategy=bt.strategy_name,
            dataset=bt.dataset_name,
            entry_type="backtest",
            tags="auto",
            sharpe=bt.metrics.get("sharpe_ratio", 0),
            max_dd=bt.metrics.get("max_drawdown_pct", 0),
            notes=f"Auto-logged. Trades: {bt.metrics.get('total_trades', 0)}, "
                  f"Return: {bt.metrics.get('total_return_pct', 0)}%",
            params=json.dumps(bt.config.__dict__) if bt.config else "{}",
        )
        st.success("Backtest logged!")
        st.rerun()

# --- View entries ---
st.subheader("Journal Entries")
filter_type = st.multiselect("Filter by type", ["backtest", "demo", "live-test", "note"])
filter_tags = st.text_input("Filter by tag")

entries = db.get_entries(entry_type=filter_type or None, tag=filter_tags or None)

if entries:
    df = pd.DataFrame(entries)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No journal entries yet.")
