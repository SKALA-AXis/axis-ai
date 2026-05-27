"""Evidence-aware fact ranking and budgeted selection."""

from __future__ import annotations

from statistics import median
from typing import Any

from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy


def rank_facts(
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    if not facts:
        return []
    component_rows = [_components(fact, policy=policy) for fact in facts]
    numeric_max = max(row["numeric_density"] for row in component_rows) or 1.0
    materiality_max = max(row["materiality_density"] for row in component_rows) or 1.0
    evidence_lengths = [
        row["evidence_chars"] for row in component_rows if row["evidence_chars"] > 0
    ]
    evidence_median = median(evidence_lengths) if evidence_lengths else 1.0

    ranked: list[dict[str, Any]] = []
    for fact, row in zip(facts, component_rows, strict=True):
        evidence_balance = _balanced_evidence_score(row["evidence_chars"], evidence_median)
        weights = policy.rank_component_weights
        score = round(
            weights["provenance_score"] * row["provenance_score"]
            + weights["materiality_density"] * (row["materiality_density"] / materiality_max)
            + weights["numeric_density"] * (row["numeric_density"] / numeric_max)
            + weights["section_specificity"] * row["section_specificity"]
            + weights["evidence_balance"] * evidence_balance,
            4,
        )
        ranked.append(
            {
                **fact,
                "integration_rank_score": score,
                "integration_rank_components": {
                    "provenance_score": row["provenance_score"],
                    "materiality_density": row["materiality_density"],
                    "numeric_density": row["numeric_density"],
                    "section_specificity": row["section_specificity"],
                    "evidence_balance": round(evidence_balance, 4),
                },
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("integration_rank_score") or 0.0),
            _safe_int(item.get("article_id")),
            str(item.get("fact_id") or ""),
        ),
        reverse=True,
    )


def select_facts_for_issue(
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = rank_facts(facts, policy=policy)
    if not ranked:
        return [], _selection_metadata([], [], policy=policy, reason="no_facts")

    total_chars = sum(_fact_chars(fact) for fact in ranked)
    if total_chars <= policy.target_evidence_chars:
        return ranked, _selection_metadata(
            ranked,
            ranked,
            policy=policy,
            reason="within_budget_all_facts_selected",
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    coverage_groups = _coverage_groups(ranked)
    for group in coverage_groups.values():
        for fact in group[: policy.minimum_coverage_per_group]:
            _append_if_new(selected, selected_ids, fact)

    budget_used = sum(_fact_chars(fact) for fact in selected)
    for fact in ranked:
        if str(fact.get("fact_id") or "") in selected_ids:
            continue
        fact_chars = _fact_chars(fact)
        if budget_used + fact_chars > policy.target_evidence_chars:
            continue
        _append_if_new(selected, selected_ids, fact)
        budget_used += fact_chars

    selected = sorted(
        selected,
        key=lambda item: float(item.get("integration_rank_score") or 0.0),
        reverse=True,
    )
    return selected, _selection_metadata(
        selected,
        ranked,
        policy=policy,
        reason="budgeted_score_and_coverage_selection",
    )


def _components(fact: dict[str, Any], *, policy: IntegrationPolicy) -> dict[str, float]:
    derived_from = str(fact.get("derived_from") or "unknown")
    text = f"{fact.get('fact') or ''} {fact.get('evidence_text') or ''}".lower()
    provenance_score = policy.provenance_quality.get(
        derived_from,
        policy.provenance_quality.get("unknown", 0.5),
    )
    materiality_density = float(
        sum(1 for term in policy.materiality_terms if term.lower() in text)
    )
    numeric_density = float(len(fact.get("numbers_and_dates") or []))
    section_specificity = 0.0
    if fact.get("section_key"):
        section_specificity += 0.5
    if fact.get("section_title"):
        section_specificity += 0.25
    if fact.get("source_chunk_uid"):
        section_specificity += 0.25
    return {
        "provenance_score": float(provenance_score),
        "materiality_density": materiality_density,
        "numeric_density": numeric_density,
        "section_specificity": min(section_specificity, 1.0),
        "evidence_chars": float(_fact_chars(fact)),
    }


def _balanced_evidence_score(chars: float, median_chars: float) -> float:
    if chars <= 0 or median_chars <= 0:
        return 0.0
    ratio = chars / median_chars
    if ratio <= 1:
        return ratio
    return max(0.0, 1.0 - min(ratio - 1.0, 1.0) * 0.5)


def _coverage_groups(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        article_id = _safe_int(fact.get("article_id"))
        fact_type = str(fact.get("fact_type") or "general_fact")
        derived_from = str(fact.get("derived_from") or "unknown")
        groups.setdefault(f"article:{article_id}", []).append(fact)
        groups.setdefault(f"fact_type:{fact_type}", []).append(fact)
        groups.setdefault(f"derived_from:{derived_from}", []).append(fact)
    for key, group in groups.items():
        groups[key] = sorted(
            group,
            key=lambda item: float(item.get("integration_rank_score") or 0.0),
            reverse=True,
        )
    return groups


def _append_if_new(
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    fact: dict[str, Any],
) -> None:
    fact_id = str(fact.get("fact_id") or "")
    if not fact_id or fact_id in selected_ids:
        return
    selected_ids.add(fact_id)
    selected.append(fact)


def _selection_metadata(
    selected: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
    reason: str,
) -> dict[str, Any]:
    selected_ids = {str(item.get("fact_id") or "") for item in selected}
    return {
        "selection_reason": reason,
        "evidence_budget_chars": policy.target_evidence_chars,
        "total_fact_count": len(ranked),
        "selected_fact_count": len(selected),
        "omitted_fact_count": max(0, len(ranked) - len(selected)),
        "selected_fact_ids": [str(item.get("fact_id") or "") for item in selected],
        "omitted_fact_ids": [
            str(item.get("fact_id") or "")
            for item in ranked
            if str(item.get("fact_id") or "") not in selected_ids
        ],
    }


def _fact_chars(fact: dict[str, Any]) -> int:
    return len(str(fact.get("fact") or "")) + len(str(fact.get("evidence_text") or ""))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["rank_facts", "select_facts_for_issue"]
