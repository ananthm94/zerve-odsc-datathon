__all__ = ["run_analytics_question"]


def run_analytics_question(*args, **kwargs):
    from analytics_agent.ask.graph import run_analytics_question as _run_analytics_question

    return _run_analytics_question(*args, **kwargs)
