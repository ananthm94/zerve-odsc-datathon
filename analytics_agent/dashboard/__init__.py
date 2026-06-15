"""The "Dashboard" tab: prebuilt, cached read-only metric queries over the dbt
marts (``queries``) and rendering of generated dashboard specs (``render``).

``queries``' public API is re-exported here so existing callers can keep using
``from analytics_agent import dashboard`` and ``dashboard.kpis(...)``.
"""

from analytics_agent.dashboard.queries import *  # noqa: F401,F403
from analytics_agent.dashboard.queries import __all__  # noqa: F401
