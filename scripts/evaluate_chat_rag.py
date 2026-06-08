from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chat RAG retrieval with JSONL cases.")
    parser.add_argument(
        "--cases",
        required=True,
        help="JSONL: query, expected_source_ids[, current_page]",
    )
    parser.add_argument("--final-k", default="4,6,8,10")
    parser.add_argument("--env", choices=["local", "cloud"], default=None)
    args = parser.parse_args()

    from src.config.env_loader import load_profile
    from src.db.postgres import reconfigure_from_env

    load_profile(args.env)
    reconfigure_from_env()

    from src.agents.chat_orchestrator_agent import ChatOrchestratorAgent
    from src.api.chat_schemas import ChatPageContext, ChatTurnRequest

    cases = _load_cases(Path(args.cases))
    final_ks = [int(item.strip()) for item in args.final_k.split(",") if item.strip()]
    agent = ChatOrchestratorAgent(enable_llm=False)

    rows = []
    for case in cases:
        request = ChatTurnRequest(
            message=str(case["query"]),
            current_page=ChatPageContext.model_validate(case["current_page"])
            if isinstance(case.get("current_page"), dict)
            else None,
        )
        candidates = agent._retrieve(request)  # noqa: SLF001 - evaluation probes retrieval directly.
        expected = {str(item) for item in case.get("expected_source_ids") or []}
        candidate_ids = [candidate.source_id for candidate in candidates]
        rows.append({"query": case["query"], "expected": expected, "candidate_ids": candidate_ids})

    summary = {}
    for top_k in final_ks:
        recalls = []
        reciprocal_ranks = []
        for row in rows:
            expected = row["expected"]
            top_ids = row["candidate_ids"][:top_k]
            hits = [source_id for source_id in top_ids if source_id in expected]
            recalls.append(len(hits) / max(len(expected), 1))
            reciprocal_ranks.append(_reciprocal_rank(top_ids, expected))
        summary[f"top_{top_k}"] = {
            "recall": round(mean(recalls), 4) if recalls else 0.0,
            "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        }

    print(json.dumps({"cases": len(rows), "summary": summary}, ensure_ascii=False, indent=2))


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _reciprocal_rank(candidate_ids: list[str], expected: set[str]) -> float:
    for index, source_id in enumerate(candidate_ids, 1):
        if source_id in expected:
            return 1.0 / index
    return 0.0


if __name__ == "__main__":
    main()
