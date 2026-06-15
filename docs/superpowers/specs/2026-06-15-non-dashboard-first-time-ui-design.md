# Non-Dashboard First-Time UI Design

## Goal

Improve first-time usability for the Streamlit app while leaving the Dashboard
tab unchanged. The app should make it immediately clear what each non-Dashboard
tab does, what action to take next, and how to interpret agent, library, and
experiment output.

## Scope

In scope:

- Ask tab
- Library tab
- Experiments tab
- About tab
- Shared non-invasive CSS that does not change Dashboard layout or charts
- Small testable helper functions for labels, empty states, counts, or examples

Out of scope:

- Dashboard tab layout, controls, metrics, and chart behavior
- Data model, SQL, dbt, retrieval, or agent orchestration changes
- New external UI dependencies
- Large navigation or page architecture rewrites

## Approach

Use a guided product pass with light visual polish. Keep the existing tab
structure and Streamlit-native controls, but add clearer orientation and action
paths for users who open the app for the first time.

This avoids disrupting existing workflows and keeps the UI changes close to
`streamlit_app.py` and testable helper modules.

## Ask Tab

The Ask tab should become the obvious starting point.

Changes:

- Replace the long empty-state paragraph with a concise orientation block.
- Add example prompt buttons for common first questions.
- Make the clear-conversation control read as a secondary maintenance action.
- Use clearer labels for dashboard preview approval and JSON editing.
- Keep reasoning, model usage, sources, and SQL collapsed by default.
- Preserve the existing chat history and agent streaming behavior.

Expected first-time outcome:

A new user can click an example or type a question without needing to understand
LangGraph, semantic retrieval, snapshots, or SQL validation first.

## Library Tab

The Library tab should read as a saved-work browser.

Changes:

- Add a compact summary of saved report and dashboard counts.
- Improve the empty state with a direct explanation of how items arrive here.
- Keep search and grouping controls, but label grouped sections more clearly.
- Make each saved item easier to scan with type, status, date, theme, and metrics.
- Avoid changing report or dashboard persistence.

Expected first-time outcome:

A new user understands that Library is for approved, reusable analysis artifacts,
not a data source or settings page.

## Experiments Tab

The Experiments tab should feel guided and careful.

Changes:

- Convert the opening warning into plain, compact caveat copy.
- Organize controls into a simple step-like flow: choose comparison, choose
  cohort dimension, choose cohorts, choose outcome, run comparison.
- Improve labels so users understand that these are observational cohorts.
- Frame the result with a clear recommendation heading and keep statistical
  detail available but secondary.
- Preserve the existing statistical functions and quasi-experiment caveats.

Expected first-time outcome:

A new user can run a cohort comparison without mistaking the result for a
randomized A/B test.

## About Tab

The About tab should become a quick orientation page.

Changes:

- Replace the long architecture block with short sections:
  - What this app does
  - How to use it
  - Trust and guardrails
  - Current status
- Keep the core architecture facts from the current About copy.
- Avoid making the page feel like a landing page or marketing page.

Expected first-time outcome:

A new user can understand the app's purpose, workflow, and safety boundaries in
under a minute.

## Shared Visual Treatment

Use restrained Streamlit styling:

- Slightly tighter top spacing and section rhythm.
- Softer, readable containers for first-time guidance blocks.
- Subtle button and expander polish.
- Avoid a one-note purple surface by keeping violet as the accent only.
- Do not add decorative hero sections, large gradients, or card-heavy dashboard
  mosaics.

Dashboard-specific metric cards and charts should not be restructured.

## Testing And Verification

Add focused unit tests for new helper behavior if helpers are added.

Run:

- `python -m unittest tests/test_ui_helpers.py`
- `python -m unittest discover -s tests`

If the local warehouse or credentials prevent full app exercise, report that
clearly and still run the unit tests that do not require external services.

Manual UI verification:

- Launch `streamlit run streamlit_app.py` if dependencies are available.
- Check Ask, Library, Experiments, and About.
- Confirm Dashboard still loads through the same code path and has no deliberate
  layout changes.

## Risks

- Streamlit CSS selectors can change across versions, so CSS should stay minimal.
- Over-explaining could make the UI feel busy, so guidance copy should be short.
- Existing uncommitted changes include `streamlit_app.py`; edits must preserve
  user work and avoid unrelated cleanup.
