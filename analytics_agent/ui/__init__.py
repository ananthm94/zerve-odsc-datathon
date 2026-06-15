"""Streamlit presentation layer.

One module per UI tab — ``ask``, ``library``, ``dashboard``, ``experiments``,
``about`` — each exposing a ``render()`` function, plus shared ``theme`` and
``common`` helpers. ``streamlit_app.py`` is a thin entrypoint that wires the
tabs together.
"""
