"""The "About" tab — architecture and status."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.subheader("About")
    st.markdown(
        """
**What this app does**

Zerve Analytics Agent answers product questions over a dbt-modeled analytics
warehouse. It combines a semantic layer, retrieval over dbt metadata, guarded
read-only SQL, and an autonomous analysis loop.

**How to use it**

1. Start in **Ask** with a plain-English product question.
2. Open reasoning, sources, SQL, or model usage only when you need detail.
3. Save useful answers to **Library** so they can be reused.
4. Use **Experiments** for directional cohort comparisons across user segments.

**Trust and guardrails**

- Queries are validated as read-only before execution.
- Metrics are grounded in dbt semantic-layer definitions when available.
- Answers can be checked against trusted snapshots.
- Experiment results are observational and may be confounded by self-selection.

**Current status**

DuckDB is the default warehouse, BigQuery is optional, and the app includes
multi-provider LLM support, report memory, dashboard generation, model-cost
tracking, and observational cohort analysis. The dataset spans 2025-09 through
2026-04 and contains about 3.5M product events.
        """
    )
