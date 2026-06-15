from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics_agent.core.config import AgentConfig, load_config
from analytics_agent.ask.graph import _choose_response_mode, _classify_intent
from analytics_agent.ask.semantic_layer import load_semantic_layer, lookup_metrics


def _load_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text()) or []


def _config() -> AgentConfig:
    try:
        return load_config()
    except Exception:
        return AgentConfig(
            openai_api_key="x",
            qdrant_url="x",
            qdrant_api_key="x",
            qdrant_collection_name="x",
        )


def _score_case(case: dict, layer: dict) -> dict:
    question = case["question"]
    intent = _classify_intent(question)
    response_mode = _choose_response_mode(question, intent)
    metrics = lookup_metrics(question, layer, llm=None)
    metric_names = [metric.metric for metric in metrics]
    source_tables = sorted(
        {
            table.strip()
            for metric in metrics
            for table in metric.source_table.split(",")
            if table.strip()
        }
    )

    checks = {
        "intent": intent == case.get("expected_intent"),
        "response_mode": response_mode == case.get("expected_response_mode"),
        "metrics": set(case.get("expected_metrics", [])) <= set(metric_names),
        "tables": set(case.get("expected_tables", [])) <= set(source_tables),
    }
    return {
        "question": question,
        "intent": intent,
        "response_mode": response_mode,
        "metrics": metric_names,
        "tables": source_tables,
        "checks": checks,
        "score": sum(1 for ok in checks.values() if ok),
        "possible": len(checks),
    }


def main() -> int:
    cases = _load_cases(Path(__file__).with_name("cases.yaml"))
    config = _config()
    layer = load_semantic_layer(config)
    results = [_score_case(case, layer) for case in cases]
    total = sum(result["score"] for result in results)
    possible = sum(result["possible"] for result in results)

    print("Agentic analytics eval scorecard")
    print(f"Score: {total}/{possible} ({(total / possible * 100) if possible else 0:.1f}%)")
    print()
    for result in results:
        status = "PASS" if result["score"] == result["possible"] else "FAIL"
        print(f"[{status}] {result['question']}")
        print(f"  intent: {result['intent']}")
        print(f"  response_mode: {result['response_mode']}")
        print(f"  metrics: {', '.join(result['metrics']) or '(none)'}")
        print(f"  tables: {', '.join(result['tables']) or '(none)'}")
        failed = [name for name, ok in result["checks"].items() if not ok]
        if failed:
            print(f"  failed checks: {', '.join(failed)}")
        print()

    return 0 if total == possible else 1


if __name__ == "__main__":
    raise SystemExit(main())
