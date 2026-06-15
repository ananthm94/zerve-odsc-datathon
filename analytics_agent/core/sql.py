import re
from dataclasses import dataclass


MUTATION_KEYWORDS = {
    "alter",
    "create",
    "delete",
    "drop",
    "insert",
    "merge",
    "truncate",
    "update",
}
TABLE_REF_RE = re.compile(
    r"\b(?:from|join)\s+(`[^`]+`|[a-zA-Z_][\w.-]*(?:\.[a-zA-Z_][\w-]*){0,2})",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SQLValidationResult:
    sql: str
    referenced_tables: set[str]


def strip_sql_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalize_sql(sql: str) -> str:
    return strip_sql_fences(sql).rstrip(";").strip()


def _strip_comments(sql: str) -> str:
    without_line_comments = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", without_line_comments, flags=re.DOTALL)


def _table_name(ref: str) -> str:
    return ref.strip("`").split(".")[-1]


def _is_aggregate(sql: str) -> bool:
    lowered = sql.lower()
    return bool(
        re.search(r"\b(count|sum|avg|min|max|approx_count_distinct)\s*\(", lowered)
        or re.search(r"\bgroup\s+by\b", lowered)
    )


def ensure_default_limit(sql: str, limit: int = 100) -> str:
    cleaned = normalize_sql(sql)
    if _is_aggregate(cleaned) or re.search(r"\blimit\s+\d+\b", cleaned, re.IGNORECASE):
        return cleaned
    return f"{cleaned}\nLIMIT {limit}"


def validate_sql(
    sql: str,
    allowed_table_names: set[str],
    fully_qualified_tables: set[str],
) -> SQLValidationResult:
    cleaned = normalize_sql(sql)
    commentless = _strip_comments(cleaned)
    lowered = commentless.lower().strip()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise SQLValidationError("Only read-only SELECT queries are allowed.")

    tokens = set(re.findall(r"\b[a-z_]+\b", lowered))
    blocked = sorted(tokens & MUTATION_KEYWORDS)
    if blocked:
        raise SQLValidationError(
            f"Query contains blocked keyword(s): {', '.join(blocked)}"
        )

    table_refs = TABLE_REF_RE.findall(commentless)
    if not table_refs:
        raise SQLValidationError("Query must reference at least one allowed dbt model.")

    referenced_tables = {_table_name(ref) for ref in table_refs}
    disallowed = referenced_tables - allowed_table_names
    if disallowed:
        raise SQLValidationError(
            f"Query references disallowed table(s): {', '.join(sorted(disallowed))}"
        )

    normalized_allowed_refs = {ref.strip("`").lower() for ref in fully_qualified_tables}
    for ref in table_refs:
        if ref.strip("`").lower() not in normalized_allowed_refs:
            raise SQLValidationError(
                "Queries must use fully qualified BigQuery table names."
            )

    return SQLValidationResult(sql=cleaned, referenced_tables=referenced_tables)
