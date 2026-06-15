import tempfile
import unittest
from dataclasses import replace

from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import ReportRecord
from analytics_agent.library.memory import ReportMemory


def _config(store_dir: str) -> AgentConfig:
    return AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        agent_store_dir=store_dir,
    )


def _record(**overrides) -> ReportRecord:
    base = ReportRecord(
        id="report-1",
        question="Why did AI credit consumption increase?",
        summary="AI token usage drove higher credit consumption.",
        metrics_used=["total_credits_used", "total_ai_tokens"],
        sql_used=["select sum(credits_used) from main.fct_credits"],
        date_range="2026-03-01 to 2026-04-30",
        source_tables=["fct_credits", "fct_ai_generations"],
        approval_status="approved",
        tags=["credits", "ai"],
        theme="Credits",
        created_at="2026-06-14T12:00:00",
    )
    return replace(base, **overrides)


class ReportMemoryTests(unittest.TestCase):
    def test_save_then_recall_returns_approved_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ReportMemory(_config(tmp))
            memory.save_report(_record())

            recalled = memory.recall_reports("What caused credit consumption?", k=3)

        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].id, "report-1")
        self.assertEqual(recalled[0].metrics_used, ["total_credits_used", "total_ai_tokens"])

    def test_recall_ranks_relevant_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ReportMemory(_config(tmp))
            memory.save_report(_record(id="credits", created_at="2026-06-14T12:00:00"))
            memory.save_report(
                _record(
                    id="errors",
                    question="Which operating systems have the most exceptions?",
                    summary="Windows users had the most exception events.",
                    metrics_used=["total_exceptions"],
                    source_tables=["fct_exceptions"],
                    tags=["errors"],
                    theme="Errors",
                    created_at="2026-06-15T12:00:00",
                )
            )

            recalled = memory.recall_reports("credit usage and token consumption", k=2)

        self.assertEqual([record.id for record in recalled], ["credits", "errors"])

    def test_pending_reports_are_not_recalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ReportMemory(_config(tmp))
            memory.save_report(_record(id="pending", approval_status="pending"))

            recalled = memory.recall_reports("credit consumption", k=3)

        self.assertEqual(recalled, [])


class FakeDoc:
    def __init__(self, report_id: str):
        self.metadata = {"id": report_id}


class FakeVectorStore:
    def __init__(self):
        self.saved = []

    def add_texts(self, texts, metadatas):
        self.saved.append((texts, metadatas))

    def similarity_search(self, query, k):
        return [FakeDoc("vector-hit")]


class VectorHookTests(unittest.TestCase):
    def test_vector_store_hook_is_used_when_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FakeVectorStore()
            memory = ReportMemory(_config(tmp), vector_store=store)
            memory.save_report(_record(id="vector-hit"))

            recalled = memory.recall_reports("anything", k=1)

        self.assertEqual(len(store.saved), 1)
        self.assertEqual(recalled[0].id, "vector-hit")


if __name__ == "__main__":
    unittest.main()
