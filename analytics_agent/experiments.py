"""Experiment / cohort analysis (v0.6).

The Zerve dataset has **no experiment-assignment column**, so there are no true
randomized A/B tests here. This module compares **observational cohorts** —
slices defined by a person property (role, cloud provider, activity segment) or a
time-based pre/post split — and is explicit that results are **quasi-experimental**
(differences may be confounded by who self-selects into each cohort).

Two comparison families are supported:

* **Continuous metrics** (tokens, latency, events per user): Welch's t-test +
  Mann-Whitney U, with Cohen's d and a CI for the mean difference.
* **Proportions / rates** (AI-adoption rate, conversion): two-proportion z-test +
  chi-square, with the risk difference and its CI.

Cohorts are sampled one row per entity (default ``user_id``) to avoid
pseudo-replication. Stats functions are pure and independently unit-tested; an
LLM then drafts a ship / no-ship narrative that flags the statistical caveats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from analytics_agent.config import AgentConfig
from analytics_agent.llm import build_chat_model
from analytics_agent.runners import build_runner
from analytics_agent.sql import validate_sql


# --- Specs -------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """Defines an observational two-cohort comparison.

    ``metric_table`` holds the per-entity metric; ``group_table`` holds the
    attribute that defines the cohorts. They may be the same model (no join) or
    different models joined on ``entity_key``.
    """

    metric_table: str
    group_table: str
    group_column: str
    cohort_a: str
    cohort_b: str
    entity_key: str = "user_id"
    max_sample: int = 50_000
    # Continuous: the numeric column. Proportion: leave None and set success_expr.
    metric_column: str | None = None
    # Proportion: a boolean SQL expression over the metric table, e.g.
    # "is_ai_user" or "total_credits_used > 0".
    success_expr: str | None = None


# --- Results -----------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousComparison:
    kind: Literal["continuous"]
    metric: str
    dimension: str
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    cohens_d: float
    t_statistic: float
    t_p_value: float
    mw_u_statistic: float
    mw_p_value: float


@dataclass(frozen=True)
class ProportionComparison:
    kind: Literal["proportion"]
    metric: str
    dimension: str
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    successes_a: int
    successes_b: int
    rate_a: float
    rate_b: float
    rate_diff: float
    ci_low: float
    ci_high: float
    z_statistic: float
    z_p_value: float
    chi2_statistic: float
    chi2_p_value: float


# --- Pure statistics ---------------------------------------------------------


def cohens_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Standardized mean difference using the pooled standard deviation."""
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = stats.tvar(sample_a)
    var_b = stats.tvar(sample_b)
    pooled = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled == 0:
        return 0.0
    return (_mean(sample_a) - _mean(sample_b)) / pooled


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def compare_continuous(
    metric: str,
    dimension: str,
    label_a: str,
    label_b: str,
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
) -> ContinuousComparison:
    """Welch's t-test + Mann-Whitney U for two independent continuous samples."""
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError("Each cohort needs at least 2 observations.")

    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = _mean(sample_a), _mean(sample_b)
    var_a, var_b = stats.tvar(sample_a), stats.tvar(sample_b)

    t_stat, t_p = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    # Welch–Satterthwaite CI for the difference in means.
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        df = float(n_a + n_b - 2)
        ci_low = ci_high = mean_a - mean_b
    else:
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        margin = t_crit * se
        ci_low = (mean_a - mean_b) - margin
        ci_high = (mean_a - mean_b) + margin

    mw_u, mw_p = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")

    return ContinuousComparison(
        kind="continuous",
        metric=metric,
        dimension=dimension,
        label_a=label_a,
        label_b=label_b,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_a - mean_b,
        ci_low=ci_low,
        ci_high=ci_high,
        cohens_d=cohens_d(sample_a, sample_b),
        t_statistic=float(t_stat),
        t_p_value=float(t_p),
        mw_u_statistic=float(mw_u),
        mw_p_value=float(mw_p),
    )


def compare_proportions(
    metric: str,
    dimension: str,
    label_a: str,
    label_b: str,
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> ProportionComparison:
    """Two-proportion z-test + chi-square for two independent rates."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Each cohort needs at least 1 observation.")

    rate_a = successes_a / n_a
    rate_b = successes_b / n_b

    z_stat, z_p = proportions_ztest(
        [successes_a, successes_b], [n_a, n_b]
    )

    # 2x2 contingency: rows = cohort, cols = [success, failure].
    table = [
        [successes_a, n_a - successes_a],
        [successes_b, n_b - successes_b],
    ]
    chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)

    # Wald CI for the risk difference.
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(
        rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b
    )
    margin = z_crit * se
    diff = rate_a - rate_b

    return ProportionComparison(
        kind="proportion",
        metric=metric,
        dimension=dimension,
        label_a=label_a,
        label_b=label_b,
        n_a=n_a,
        n_b=n_b,
        successes_a=successes_a,
        successes_b=successes_b,
        rate_a=rate_a,
        rate_b=rate_b,
        rate_diff=diff,
        ci_low=diff - margin,
        ci_high=diff + margin,
        z_statistic=float(z_stat),
        z_p_value=float(z_p),
        chi2_statistic=float(chi2),
        chi2_p_value=float(chi2_p),
    )


# --- Cohort sampling (DB-backed) ---------------------------------------------

# These queries are built internally (not LLM-generated). They are still run
# through validate_sql and use fully-qualified, allow-listed table identifiers.


def _ident(name: str) -> str:
    """Guard against injection in internally-built identifiers."""
    if not name.replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fqtn(config: AgentConfig, table: str) -> str:
    fqtns = config.fully_qualified_tables
    if table not in fqtns:
        raise ValueError(
            f"Table {table!r} is not in the allow-list ({sorted(fqtns)})."
        )
    return fqtns[table]


def _from_clause(config: AgentConfig, spec: CohortSpec) -> str:
    metric_fqtn = _fqtn(config, _ident(spec.metric_table))
    if spec.metric_table == spec.group_table:
        return f"{metric_fqtn} m"
    group_fqtn = _fqtn(config, _ident(spec.group_table))
    key = _ident(spec.entity_key)
    return f"{metric_fqtn} m JOIN {group_fqtn} g ON m.{key} = g.{key}"


def _group_ref(spec: CohortSpec) -> str:
    prefix = "m" if spec.metric_table == spec.group_table else "g"
    return f"{prefix}.{_ident(spec.group_column)}"


def _run(config: AgentConfig, sql: str, max_results: int) -> list[dict]:
    validate_sql(
        sql,
        allowed_table_names=config.allowed_table_names,
        fully_qualified_tables=set(config.fully_qualified_tables.values()),
    )
    return build_runner(config).execute(sql, max_results=max_results)


def fetch_continuous_samples(
    config: AgentConfig, spec: CohortSpec
) -> tuple[list[float], list[float]]:
    if not spec.metric_column:
        raise ValueError("CohortSpec.metric_column is required for continuous tests.")
    metric_ref = f"m.{_ident(spec.metric_column)}"
    group_ref = _group_ref(spec)
    sql = (
        f"SELECT {group_ref} AS cohort, {metric_ref} AS value\n"
        f"FROM {_from_clause(config, spec)}\n"
        f"WHERE {group_ref} IN ({_literal(spec.cohort_a)}, {_literal(spec.cohort_b)})\n"
        f"  AND {metric_ref} IS NOT NULL"
    )
    rows = _run(config, sql, max_results=spec.max_sample)
    sample_a = [float(r["value"]) for r in rows if r["cohort"] == spec.cohort_a]
    sample_b = [float(r["value"]) for r in rows if r["cohort"] == spec.cohort_b]
    return sample_a, sample_b


def fetch_proportion_counts(
    config: AgentConfig, spec: CohortSpec
) -> dict[str, tuple[int, int]]:
    if not spec.success_expr:
        raise ValueError("CohortSpec.success_expr is required for proportion tests.")
    group_ref = _group_ref(spec)
    # success_expr is an internal, trusted boolean expression over alias m.
    sql = (
        f"SELECT {group_ref} AS cohort,\n"
        f"       COUNT(*) AS n,\n"
        f"       SUM(CASE WHEN {spec.success_expr} THEN 1 ELSE 0 END) AS successes\n"
        f"FROM {_from_clause(config, spec)}\n"
        f"WHERE {group_ref} IN ({_literal(spec.cohort_a)}, {_literal(spec.cohort_b)})\n"
        f"GROUP BY {group_ref}"
    )
    rows = _run(config, sql, max_results=10)
    return {
        str(r["cohort"]): (int(r["successes"]), int(r["n"]))
        for r in rows
    }


def run_continuous_experiment(
    config: AgentConfig, spec: CohortSpec
) -> ContinuousComparison:
    sample_a, sample_b = fetch_continuous_samples(config, spec)
    return compare_continuous(
        metric=spec.metric_column or "value",
        dimension=spec.group_column,
        label_a=spec.cohort_a,
        label_b=spec.cohort_b,
        sample_a=sample_a,
        sample_b=sample_b,
    )


def run_proportion_experiment(
    config: AgentConfig, spec: CohortSpec
) -> ProportionComparison:
    counts = fetch_proportion_counts(config, spec)
    if spec.cohort_a not in counts or spec.cohort_b not in counts:
        raise ValueError(
            f"Cohort data missing; got cohorts {sorted(counts)} "
            f"but need {spec.cohort_a!r} and {spec.cohort_b!r}."
        )
    succ_a, n_a = counts[spec.cohort_a]
    succ_b, n_b = counts[spec.cohort_b]
    return compare_proportions(
        metric=spec.success_expr or "rate",
        dimension=spec.group_column,
        label_a=spec.cohort_a,
        label_b=spec.cohort_b,
        successes_a=succ_a,
        n_a=n_a,
        successes_b=succ_b,
        n_b=n_b,
    )


# --- Reporting ---------------------------------------------------------------


def format_comparison(result: ContinuousComparison | ProportionComparison) -> str:
    """A compact, deterministic text summary (also fed to the LLM narrator)."""
    if isinstance(result, ContinuousComparison):
        return (
            f"Quasi-experiment (observational): {result.metric} by "
            f"{result.dimension}\n"
            f"Cohort A = {result.label_a}: n={result.n_a}, mean={result.mean_a:.4g}\n"
            f"Cohort B = {result.label_b}: n={result.n_b}, mean={result.mean_b:.4g}\n"
            f"Mean difference (A-B) = {result.mean_diff:.4g} "
            f"(95% CI {result.ci_low:.4g} to {result.ci_high:.4g})\n"
            f"Cohen's d = {result.cohens_d:.3f}\n"
            f"Welch t-test: t={result.t_statistic:.3f}, p={result.t_p_value:.4g}\n"
            f"Mann-Whitney U: U={result.mw_u_statistic:.0f}, "
            f"p={result.mw_p_value:.4g}"
        )
    return (
        f"Quasi-experiment (observational): {result.metric} by "
        f"{result.dimension}\n"
        f"Cohort A = {result.label_a}: {result.successes_a}/{result.n_a} "
        f"= {result.rate_a:.4f}\n"
        f"Cohort B = {result.label_b}: {result.successes_b}/{result.n_b} "
        f"= {result.rate_b:.4f}\n"
        f"Rate difference (A-B) = {result.rate_diff:.4f} "
        f"(95% CI {result.ci_low:.4f} to {result.ci_high:.4f})\n"
        f"Two-proportion z-test: z={result.z_statistic:.3f}, "
        f"p={result.z_p_value:.4g}\n"
        f"Chi-square: X2={result.chi2_statistic:.3f}, p={result.chi2_p_value:.4g}"
    )


def _narrative_prompt(result: ContinuousComparison | ProportionComparison):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are a data scientist writing a ship / no-ship recommendation "
                "from a COHORT comparison. The cohorts are OBSERVATIONAL (not "
                "randomly assigned), so differences may be confounded by who "
                "selects into each cohort — state this clearly and avoid causal "
                "language. Give a clear recommendation (ship / do not ship / "
                "inconclusive — gather more data), justified by the effect size and "
                "confidence interval, not just the p-value. Explicitly flag "
                "statistical issues you can see: low or imbalanced sample sizes, "
                "wide confidence intervals, a CI that crosses zero, and multiple-"
                "comparison risk if several metrics are being tested. Be concise."
            )
        ),
        HumanMessage(content=f"Comparison results:\n{format_comparison(result)}"),
    ]


def narrate_experiment(
    config: AgentConfig, result: ContinuousComparison | ProportionComparison
) -> str:
    """LLM-drafted ship/no-ship narrative for a comparison."""
    llm = build_chat_model(config)
    return llm.invoke(_narrative_prompt(result)).content


__all__ = [
    "CohortSpec",
    "ContinuousComparison",
    "ProportionComparison",
    "cohens_d",
    "compare_continuous",
    "compare_proportions",
    "fetch_continuous_samples",
    "fetch_proportion_counts",
    "run_continuous_experiment",
    "run_proportion_experiment",
    "format_comparison",
    "narrate_experiment",
]
