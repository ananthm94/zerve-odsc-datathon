"""Approved report memory for retrieve-before-compute workflows.

Reports are persisted in a small SQLite database under ``agent_store_dir``.
Recall is deterministic and offline by default, with an optional fakeable vector
store hook for future Qdrant-backed retrieval.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import ReportRecord


REPORTS_DB = "reports.db"
APPROVED_REPORTS_COLLECTION = "approved_reports"


class ReportMemory:
    """Persist and recall approved analysis reports."""

    def __init__(self, config: AgentConfig, vector_store=None):
        self.config = config
        self.store_dir = Path(config.agent_store_dir)
        self.db_path = self.store_dir / REPORTS_DB
        self.vector_store = vector_store
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save_report(self, record: ReportRecord) -> None:
        """Insert or update a report record."""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO reports (
                    id, question, summary, metrics_used, sql_used, date_range,
                    source_tables, approval_status, tags, theme, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    question=excluded.question,
                    summary=excluded.summary,
                    metrics_used=excluded.metrics_used,
                    sql_used=excluded.sql_used,
                    date_range=excluded.date_range,
                    source_tables=excluded.source_tables,
                    approval_status=excluded.approval_status,
                    tags=excluded.tags,
                    theme=excluded.theme,
                    created_at=excluded.created_at
                """,
                _record_to_row(record),
            )
        self._save_to_vector_store(record)

    def recall_reports(self, question: str, k: int = 3) -> list[ReportRecord]:
        """Return the most relevant approved reports for a question."""
        if k <= 0:
            return []

        vector_records = self._recall_from_vector_store(question, k)
        if vector_records:
            return vector_records[:k]

        records = self._approved_reports()
        scored = [
            (_lexical_score(question, record), record.created_at, record.id, record)
            for record in records
        ]
        scored.sort(reverse=True)
        return [record for _, _, _, record in scored[:k]]

    def list_reports(
        self, include_pending: bool = True, include_rejected: bool = True
    ) -> list[ReportRecord]:
        """List saved reports, newest first."""
        statuses = ["approved"]
        if include_pending:
            statuses.append("pending")
        if include_rejected:
            statuses.append("rejected")
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT id, question, summary, metrics_used, sql_used, date_range,
                       source_tables, approval_status, tags, theme, created_at
                FROM reports
                WHERE approval_status IN ({placeholders})
                ORDER BY created_at DESC, id DESC
                """,
                statuses,
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metrics_used TEXT NOT NULL,
                    sql_used TEXT NOT NULL,
                    date_range TEXT NOT NULL,
                    source_tables TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reports_approval_created
                ON reports(approval_status, created_at)
                """
            )

    def _approved_reports(self) -> list[ReportRecord]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, question, summary, metrics_used, sql_used, date_range,
                       source_tables, approval_status, tags, theme, created_at
                FROM reports
                WHERE approval_status = 'approved'
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def _reports_by_ids(self, ids: Iterable[str]) -> list[ReportRecord]:
        ordered_ids = [report_id for report_id in ids if report_id]
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT id, question, summary, metrics_used, sql_used, date_range,
                       source_tables, approval_status, tags, theme, created_at
                FROM reports
                WHERE approval_status = 'approved' AND id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()
        by_id = {_row_to_record(row).id: _row_to_record(row) for row in rows}
        return [by_id[report_id] for report_id in ordered_ids if report_id in by_id]

    def _save_to_vector_store(self, record: ReportRecord) -> None:
        """Best-effort vector hook; tests can pass a fake store."""
        if self.vector_store is None or record.approval_status != "approved":
            return
        text = _record_search_text(record)
        metadata = {"id": record.id, "theme": record.theme}
        try:
            if hasattr(self.vector_store, "add_texts"):
                self.vector_store.add_texts([text], metadatas=[metadata])
        except Exception:
            return

    def _recall_from_vector_store(self, question: str, k: int) -> list[ReportRecord]:
        if self.vector_store is None:
            return []
        try:
            if hasattr(self.vector_store, "similarity_search"):
                docs = self.vector_store.similarity_search(question, k=k)
            elif hasattr(self.vector_store, "as_retriever"):
                docs = self.vector_store.as_retriever(search_kwargs={"k": k}).invoke(
                    question
                )
            else:
                return []
        except Exception:
            return []

        ids = [
            str(getattr(doc, "metadata", {}).get("id", ""))
            for doc in docs
            if getattr(doc, "metadata", {}).get("id")
        ]
        return self._reports_by_ids(ids)


def build_qdrant_report_store(config: AgentConfig):
    """Create a Qdrant vector store for approved-report memory.

    This is intentionally separate from ``ReportMemory.__init__`` so ordinary
    SQLite use and tests never make network calls.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore

    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    return QdrantVectorStore.from_existing_collection(
        collection_name=APPROVED_REPORTS_COLLECTION,
        embedding=embeddings,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        check_compatibility=False,
    )


def _record_to_row(record: ReportRecord) -> tuple:
    return (
        record.id,
        record.question,
        record.summary,
        json.dumps(record.metrics_used),
        json.dumps(record.sql_used),
        record.date_range,
        json.dumps(record.source_tables),
        record.approval_status,
        json.dumps(record.tags),
        record.theme,
        record.created_at,
    )


def _row_to_record(row) -> ReportRecord:
    return ReportRecord(
        id=row[0],
        question=row[1],
        summary=row[2],
        metrics_used=json.loads(row[3]),
        sql_used=json.loads(row[4]),
        date_range=row[5],
        source_tables=json.loads(row[6]),
        approval_status=row[7],
        tags=json.loads(row[8]),
        theme=row[9],
        created_at=row[10],
    )


def _record_search_text(record: ReportRecord) -> str:
    return " ".join(
        [
            record.question,
            record.summary,
            " ".join(record.metrics_used),
            " ".join(record.source_tables),
            " ".join(record.tags),
            record.theme,
        ]
    )


def _lexical_score(question: str, record: ReportRecord) -> int:
    query_terms = set(_terms(question))
    if not query_terms:
        return 0

    score = 0
    score += 5 * len(query_terms & set(_terms(record.question)))
    score += 4 * len(query_terms & set(_terms(" ".join(record.metrics_used))))
    score += 3 * len(query_terms & set(_terms(record.summary)))
    score += 2 * len(query_terms & set(_terms(record.theme)))
    score += len(query_terms & set(_terms(" ".join(record.tags))))
    score += len(query_terms & set(_terms(" ".join(record.source_tables))))
    return score


def _terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    expanded = []
    for term in terms:
        expanded.append(term)
        if term.endswith("s") and len(term) > 3:
            expanded.append(term[:-1])
    return expanded


__all__ = [
    "APPROVED_REPORTS_COLLECTION",
    "REPORTS_DB",
    "ReportMemory",
    "build_qdrant_report_store",
]
