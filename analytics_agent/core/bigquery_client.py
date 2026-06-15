from google.cloud import bigquery

from analytics_agent.core.config import AgentConfig


class BigQueryRunner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        # Created lazily so selecting this backend (and tests) don't require
        # credentials until a query is actually run.
        if self._client is None:
            self._client = bigquery.Client(
                project=self.config.bigquery_project,
                location=self.config.bigquery_location,
            )
        return self._client

    def dry_run(self, sql: str) -> int:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self.client.query(sql, job_config=job_config)
        return int(job.total_bytes_processed or 0)

    def execute(self, sql: str, max_results: int = 100) -> list[dict]:
        job = self.client.query(sql)
        rows = job.result(max_results=max_results)
        return [dict(row.items()) for row in rows]
