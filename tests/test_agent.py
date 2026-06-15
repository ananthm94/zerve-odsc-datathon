"""Integration tests for the autonomous analysis agent (v0.5).

The LLM, retriever, and query runner are all faked so the test runs offline and
asserts the control flow: plan -> query x N -> reflect -> synthesize, the
sub-query budget is enforced, and the final report carries query citations.
"""

import unittest
from dataclasses import replace
from unittest import mock

from analytics_agent.ask import graph as graph_module
from analytics_agent.core.config import AgentConfig


def _config(**overrides) -> AgentConfig:
    base = AgentConfig(
        openai_api_key="x",
        qdrant_url="x",
        qdrant_api_key="x",
        qdrant_collection_name="c",
        backend="duckdb",
    )
    return replace(base, **overrides)


class _Resp:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Routes by the system prompt to mimic each agent stage. Records the
    ordered sequence of stages it was asked to perform."""

    def __init__(self, *, sufficient: bool = True, plan_sub_questions=None):
        self.calls: list[str] = []
        self.sufficient = sufficient
        self.plan_sub_questions = plan_sub_questions or [
            "How many users are there?",
            "How many AI generations occurred?",
        ]

    def invoke(self, messages):
        system = messages[0].content
        if "planning an end-to-end analysis" in system:
            self.calls.append("plan")
            subs = ", ".join(f'"{q}"' for q in self.plan_sub_questions)
            return _Resp(
                '{"hypotheses": ["usage is concentrated"], '
                f'"sub_questions": [{subs}]}}'
            )
        if "Generate exactly one read-only" in system:
            self.calls.append("generate_sql")
            return _Resp("SELECT count(*) AS n FROM main.user_summary")
        if "fixing a broken" in system:
            self.calls.append("repair_sql")
            return _Resp("SELECT count(*) AS fixed FROM main.user_summary")
        if "reviewing partial analysis results" in system:
            self.calls.append("reflect")
            if self.sufficient:
                return _Resp('{"sufficient": true, "reasoning": "enough", '
                             '"follow_up_sub_questions": []}')
            return _Resp(
                '{"sufficient": false, "reasoning": "need more", '
                '"follow_up_sub_questions": ["What is the daily trend?"]}'
            )
        if "writing the final report" in system:
            self.calls.append("synthesize")
            return _Resp("Users total 5 [Q1]. AI generations total 5 [Q2].")
        raise AssertionError(f"Unexpected prompt:\n{system}")


class FakeRetriever:
    def invoke(self, query):
        return []


class FakeRunner:
    def __init__(self):
        self.executed: list[str] = []

    def dry_run(self, sql):
        return 0

    def execute(self, sql, max_results=100):
        self.executed.append(sql)
        return [{"n": 5}]


def _patched_build_graph(config, fake_llm, fake_runner):
    with mock.patch.object(graph_module, "build_chat_model", return_value=fake_llm), \
         mock.patch.object(graph_module, "create_retriever", return_value=FakeRetriever()), \
         mock.patch.object(graph_module, "build_runner", return_value=fake_runner):
        return graph_module.build_graph(config)


class AutonomousAgentTests(unittest.TestCase):
    def test_visits_plan_query_reflect_synthesize(self):
        fake_llm = FakeLLM(sufficient=True)
        fake_runner = FakeRunner()
        compiled = _patched_build_graph(_config(), fake_llm, fake_runner)

        state = compiled.invoke({"question": "What drives engagement?"})

        # Two planned sub-questions -> two queries, then one reflect, one synth.
        self.assertEqual(
            fake_llm.calls,
            ["plan", "generate_sql", "generate_sql", "reflect", "synthesize"],
        )
        self.assertEqual(len(fake_runner.executed), 2)
        self.assertEqual(len(state["findings"]), 2)
        self.assertEqual([f["id"] for f in state["findings"]], ["Q1", "Q2"])
        self.assertFalse(any(f["error"] for f in state["findings"]))

    def test_report_is_cited_and_aliased_to_answer(self):
        fake_llm = FakeLLM(sufficient=True)
        compiled = _patched_build_graph(_config(), fake_llm, FakeRunner())

        state = compiled.invoke({"question": "What drives engagement?"})

        self.assertIn("[Q1]", state["report"])
        # answer mirrors report for the legacy single-box UI.
        self.assertEqual(state["answer"], state["report"])
        # last successful query is surfaced for backward compatibility.
        self.assertEqual(state["sql"], "SELECT count(*) AS n FROM main.user_summary")
        self.assertEqual(state["rows"], [{"n": 5}])

    def test_follow_ups_loop_back_through_reflect(self):
        # First reflect requests a follow-up, second is satisfied -> a second
        # query runs and reflect is visited twice.
        fake_llm = FakeLLM(sufficient=False, plan_sub_questions=["q1"])

        # Flip to sufficient after the first reflect so the loop terminates.
        original_invoke = fake_llm.invoke

        def invoke(messages):
            resp = original_invoke(messages)
            if messages[0].content.startswith("You are reviewing"):
                fake_llm.sufficient = True
            return resp

        fake_llm.invoke = invoke
        compiled = _patched_build_graph(_config(), fake_llm, FakeRunner())

        state = compiled.invoke({"question": "trend?"})

        self.assertEqual(state["iterations"], 2)
        self.assertEqual(fake_llm.calls.count("reflect"), 2)
        self.assertEqual(fake_llm.calls.count("generate_sql"), 2)

    def test_failed_query_is_repaired_and_retried(self):
        # Runner raises on the first execute, succeeds on the repaired query.
        class FlakyRunner(FakeRunner):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def execute(self, sql, max_results=100):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError(
                        'Binder Error: column "n" not found. Candidate bindings: "fixed"'
                    )
                self.executed.append(sql)
                return [{"fixed": 7}]

        fake_llm = FakeLLM(sufficient=True, plan_sub_questions=["q1"])
        compiled = _patched_build_graph(_config(), fake_llm, FlakyRunner())

        state = compiled.invoke({"question": "trend?"})

        self.assertEqual(fake_llm.calls.count("repair_sql"), 1)
        self.assertEqual(len(state["findings"]), 1)
        self.assertIsNone(state["findings"][0]["error"])
        self.assertEqual(state["findings"][0]["rows"], [{"fixed": 7}])

    def test_budget_caps_runaway_follow_ups(self):
        # Reflect always asks for more; the ceiling must stop the loop.
        fake_llm = FakeLLM(sufficient=False, plan_sub_questions=["q1"])
        compiled = _patched_build_graph(
            _config(max_sub_queries=3), fake_llm, FakeRunner()
        )

        state = compiled.invoke({"question": "trend?"})

        self.assertEqual(state["iterations"], 3)
        self.assertEqual(fake_llm.calls.count("generate_sql"), 3)
        self.assertEqual(fake_llm.calls.count("synthesize"), 1)


class StreamingEventsTests(unittest.TestCase):
    """stream_analytics_events emits one event per node for live UIs."""

    def test_emits_plan_query_reflect_synthesize_in_order(self):
        fake_llm = FakeLLM(sufficient=True)  # two planned sub-questions
        with mock.patch.object(
            graph_module, "build_chat_model", return_value=fake_llm
        ), mock.patch.object(
            graph_module, "create_retriever", return_value=FakeRetriever()
        ), mock.patch.object(
            graph_module, "build_runner", return_value=FakeRunner()
        ):
            events = list(
                graph_module.stream_analytics_events(
                    "What drives engagement?", config=_config()
                )
            )

        stages = [e["stage"] for e in events]
        # plan, one event per sub-question, reflect, synthesize.
        self.assertEqual(
            stages,
            [
                graph_module.STAGE_PLAN,
                graph_module.STAGE_QUERY,
                graph_module.STAGE_QUERY,
                graph_module.STAGE_REFLECT,
                graph_module.STAGE_SYNTHESIZE,
            ],
        )
        # Query events carry distinct findings; synthesize carries final state.
        query_events = [e for e in events if e["stage"] == graph_module.STAGE_QUERY]
        self.assertEqual([e["finding"]["id"] for e in query_events], ["Q1", "Q2"])
        final = events[-1]
        self.assertIn("[Q1]", final["report"])
        self.assertIn("findings", final["state"])


if __name__ == "__main__":
    unittest.main()
