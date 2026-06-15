"""Cosmetic theme polish shared across every tab."""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    """Light cosmetic polish on top of the violet theme in .streamlit/config.toml.

    Kept minimal and scoped to stable test-ids so it survives Streamlit upgrades.
    """
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.4rem; }
          /* KPI metrics as soft lavender cards */
          div[data-testid="stMetric"] {
              background: #FAF8FE;
              border: 1px solid #E7E1F5;
              border-radius: 0.75rem;
              padding: 0.85rem 1rem;
          }
          div[data-testid="stMetric"] label p { color: #6B5B8A; }
          div[data-testid="stMetricValue"] { color: #4C1D95; }
          /* Active tab picks up the accent */
          button[data-baseweb="tab"][aria-selected="true"] p { color: #6D28D9; }
          .za-guide {
              background: #F8FAFC;
              border: 1px solid #E2E8F0;
              border-radius: 0.6rem;
              padding: 1rem 1.1rem;
              margin: 0.5rem 0 1rem 0;
          }
          .za-guide strong { color: #241B35; }
          .za-muted { color: #64748B; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
