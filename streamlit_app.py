"""Zerve Analytics Agent — productized multi-tab UI.

Thin entrypoint that wires the tabs together. Each tab is a feature module under
``analytics_agent.ui`` exposing a ``render()`` function:

  * Ask         — a conversational chat over the agentic trust loop (route intent
                  -> ground in the semantic layer -> plan -> query -> reflect ->
                  synthesize -> verify against snapshots). Answers can be saved to
                  the Library and generated dashboards approved.
  * Library     — saved reports and approved dashboards, grouped for reuse.
  * Dashboard   — pre-built metrics and charts over the dbt marts, with its own
                  inline filters.
  * Experiments — observational cohort comparisons with a ship/no-ship narrative.
  * About       — architecture and status.
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from analytics_agent.ui import about, ask, dashboard, experiments, library, theme

load_dotenv()

st.set_page_config(page_title="Zerve Analytics Agent", layout="wide")
theme.inject_css()
st.title("Zerve Analytics Agent")
st.caption("Autonomous analysis over a dbt semantic layer · DuckDB · RAG · LangGraph")

ask_tab, library_tab, dash_tab, exp_tab, about_tab = st.tabs(
    ["💬 Ask", "📚 Library", "📊 Dashboard", "🧪 Experiments", "ℹ️ About"]
)

with ask_tab:
    ask.render()

with library_tab:
    library.render()

with dash_tab:
    dashboard.render()

with exp_tab:
    experiments.render()

with about_tab:
    about.render()
