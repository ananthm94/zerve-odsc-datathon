"""Helpers shared across more than one tab: the cached config/data accessors
and small formatters.

Tab-specific cached query wrappers live in their own tab module; only the
cross-tab ones (``config``, ``distinct_values``, ``date_bounds``) live here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from analytics_agent import dashboard
from analytics_agent.core.config import load_config


@st.cache_resource
def config():
    return load_config()


@st.cache_data(show_spinner=False)
def distinct(table: str, column: str):
    return dashboard.distinct_values(table, column, config())


@st.cache_data(show_spinner=False)
def date_bounds():
    return dashboard.date_bounds(config())


def fmt(n: float) -> str:
    return f"{n:,.0f}"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
