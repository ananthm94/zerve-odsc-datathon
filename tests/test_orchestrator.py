import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analytics_agent.ask import graph as graph_module
from analytics_agent.core.config import AgentConfig


def _semantic_manifest() -> dict:
    return {
        "semantic_models": [
            {
                "name": "events",
                "defaults": {"agg_time_dimension": "event_date"},
                "node_relation": {"alias": "stg_events"},
                "entities": [{"name": "user", "type": "foreign", "expr": "user_id"}],
                "dimensions": [
                    {"name": "event_date", "type": "time"},
                    {"name": "event_name", "type": "categorical"},
                ],
                "measures": [
                    {
                        "name": "active_users",
                        "agg": "count_distinct",
                        "description": "Distinct active users.",
                        "expr": "user_id",
                    }
                ],
            }
        ],
        "metrics": [
            {
                "name": "daily_active_users",
                "description": "Distinct active users grouped by day.",
                "type": "simple",
                "type_params": {
                    "input_measures": [{"name": "active_users"}],
                    "measure": {"name": "active_users"},
                },
            }
        ],
    }


def _config(manifest_path: str, store_dir: str) -> AgentConfig:
    return AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        semantic_manifest_path=manifest_path,
        agent_store_dir=store_dir,
    )


class _FakeLLM:
    """Minimal chat-model stand-in: returns a fixed response content."""

    def __init__(self, text: str):
        self._text = text

    def invoke(self, messages):
        return mock.Mock(content=self._text)


class CondenseTests(unittest.TestCase):
    def test_condense_noop_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            with mock.patch.object(graph_module, "build_model_for_task") as build:
                result = graph_module._condense_question("How many users?", [], cfg)

        # Empty history means nothing to resolve: return verbatim, no LLM built.
        self.assertEqual(result, "How many users?")
        build.assert_not_called()

    def test_followup_condensed_with_history(self):
        rewritten = "Why did AI adoption change for mobile users?"
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            def fake_stream(question, config=None):
                # The standalone (condensed) question must reach the analysis path.
                self.assertEqual(question, rewritten)
                yield {
                    "stage": graph_module.STAGE_SYNTHESIZE,
                    "report": "Report [Q1].",
                    "state": {"question": question, "findings": [], "report": "Report [Q1]."},
                }

            with mock.patch.object(
                graph_module,
                "build_model_for_task",
                return_value=_FakeLLM(rewritten),
            ), mock.patch.object(
                graph_module, "stream_analytics_events", side_effect=fake_stream
            ):
                events = list(
                    graph_module.stream_agentic_events(
                        "what about for mobile users?",
                        config=cfg,
                        history=[
                            {
                                "question": "How many users use AI?",
                                "answer": "5,023 (28.6%)",
                            }
                        ],
                    )
                )

        condense_events = [
            event for event in events if event["stage"] == graph_module.STAGE_CONDENSE
        ]
        self.assertEqual(len(condense_events), 1)
        self.assertEqual(condense_events[0]["standalone_question"], rewritten)
        self.assertIn("mobile", condense_events[0]["standalone_question"])
        self.assertEqual(
            condense_events[0]["original_question"], "what about for mobile users?"
        )
        # Routing ran on the rewritten question → analysis path → snapshot stage.
        intent_event = next(e for e in events if e["stage"] == graph_module.STAGE_INTENT)
        self.assertEqual(intent_event["intent"], graph_module.INTENT_ANALYSIS)


class OrchestratorTests(unittest.TestCase):
    def test_metric_definition_short_circuits_without_sql(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            with mock.patch.object(
                graph_module,
                "stream_analytics_events",
                side_effect=AssertionError("legacy SQL path should not run"),
            ):
                events = list(
                    graph_module.stream_agentic_events(
                        "What is daily active users?", config=cfg
                    )
                )

        stages = [event["stage"] for event in events]
        self.assertIn(graph_module.STAGE_INTENT, stages)
        self.assertIn(graph_module.STAGE_SEMANTIC, stages)
        self.assertIn(graph_module.STAGE_RESPONSE_MODE, stages)
        self.assertIn(graph_module.STAGE_FINAL, stages)
        self.assertEqual(events[0]["intent"], graph_module.INTENT_METRIC_DEFINITION)
        self.assertEqual(events[1]["response_mode"], graph_module.RESPONSE_DEFINITION)
        self.assertIn("daily_active_users", events[-1]["final_response"])

    def test_dashboard_request_routes_to_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            events = list(
                graph_module.stream_agentic_events(
                    "Monitor daily active users by week", config=cfg
                )
            )

        self.assertEqual(events[0]["intent"], graph_module.INTENT_DASHBOARD_REQUEST)
        self.assertEqual(events[1]["response_mode"], graph_module.RESPONSE_DASHBOARD)
        final = events[-1]
        self.assertEqual(final["stage"], graph_module.STAGE_FINAL)
        self.assertEqual(
            final["dashboard_spec"]["charts"][0]["metric"], "daily_active_users"
        )

    def test_simple_ai_users_question_answers_number_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            with mock.patch.object(
                graph_module.dashboard,
                "kpis",
                return_value={"ai_users": 25, "total_users": 100},
            ):
                events = list(
                    graph_module.stream_agentic_events(
                        "How many users use AI?", config=cfg
                    )
                )

        self.assertEqual(events[0]["intent"], graph_module.INTENT_SIMPLE_METRIC)
        self.assertEqual(events[1]["response_mode"], graph_module.RESPONSE_QUICK)
        self.assertIn("25 users have used AI", events[-1]["final_response"])
        self.assertIn("25.0%", events[-1]["final_response"])
        self.assertIn("Context", events[-1]["final_response"])

    def test_deep_dive_question_selects_deep_response_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            def fake_stream(question, config=None):
                yield {
                    "stage": graph_module.STAGE_SYNTHESIZE,
                    "report": "Deep report [Q1].",
                    "state": {"question": question, "findings": [], "report": "Deep report [Q1]."},
                }

            with mock.patch.object(
                graph_module, "stream_analytics_events", side_effect=fake_stream
            ):
                events = list(
                    graph_module.stream_agentic_events(
                        "Give me a detailed deep dive on AI usage", config=cfg
                    )
                )

        self.assertEqual(events[1]["response_mode"], graph_module.RESPONSE_DEEP)

    def test_analysis_path_reaches_snapshot_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "semantic_manifest.json"
            manifest.write_text(json.dumps(_semantic_manifest()))
            cfg = _config(str(manifest), tmp)

            def fake_stream(question, config=None):
                yield {"stage": graph_module.STAGE_PLAN, "sub_questions": ["q"], "hypotheses": []}
                yield {
                    "stage": graph_module.STAGE_SYNTHESIZE,
                    "report": "Total users is 10 [Q1].",
                    "state": {
                        "question": question,
                        "findings": [
                            {
                                "id": "Q1",
                                "rows": [{"total_users": 10}],
                                "error": None,
                            }
                        ],
                        "report": "Total users is 10 [Q1].",
                    },
                }

            with mock.patch.object(
                graph_module, "stream_analytics_events", side_effect=fake_stream
            ):
                events = list(
                    graph_module.stream_agentic_events(
                        "Why did usage change?", config=cfg
                    )
                )

        self.assertEqual(events[0]["intent"], graph_module.INTENT_ANALYSIS)
        self.assertIn(graph_module.STAGE_SNAPSHOT, [event["stage"] for event in events])
        self.assertEqual(events[-1]["stage"], graph_module.STAGE_FINAL)


if __name__ == "__main__":
    unittest.main()
