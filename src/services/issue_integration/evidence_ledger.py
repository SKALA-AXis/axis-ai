"""Ledger builders for IntegratedIssue v2."""

from __future__ import annotations

from typing import Any


def build_evidence_ledger(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in facts:
        fact_id = str(fact.get("fact_id") or "")
        if not fact_id:
            continue
        evidence_id = f"evidence:{fact_id}"
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        ledger.append(
            {
                "evidence_id": evidence_id,
                "fact_id": fact_id,
                "source_article_ids": _source_article_ids(fact),
                "raw_article_ids": _raw_article_ids(fact),
                "evidence_text": str(fact.get("evidence_text") or fact.get("fact") or "").strip(),
                "source_type": fact.get("source_type"),
                "section_key": fact.get("section_key"),
                "section_title": fact.get("section_title"),
                "source_chunk_uid": fact.get("source_chunk_uid"),
                "derived_from": fact.get("derived_from"),
            }
        )
    return ledger


def build_claim_ledger(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for fact in facts:
        fact_text = str(fact.get("fact") or "").strip()
        fact_id = str(fact.get("fact_id") or "")
        if not fact_text or not fact_id:
            continue
        claims.append(
            {
                "claim_id": f"claim:{fact_id}",
                "claim": fact_text,
                "claim_type": fact.get("fact_type") or "general_fact",
                "fact_ids": [fact_id],
                "evidence_ids": [f"evidence:{fact_id}"],
                "source_article_ids": _source_article_ids(fact),
                "raw_article_ids": _raw_article_ids(fact),
                "confidence": _confidence_from_fact(fact),
            }
        )
    return claims


def build_fact_basis(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for index, fact in enumerate(facts, start=1):
        fact_id = str(fact.get("fact_id") or "")
        if not fact_id:
            continue
        basis.append(
            {
                "summary_line_index": index,
                "source_article_ids": _source_article_ids(fact),
                "raw_article_ids": _raw_article_ids(fact),
                "fact_ids": [fact_id],
                "evidence_ids": [f"evidence:{fact_id}"],
                "evidence_text": str(fact.get("evidence_text") or fact.get("fact") or "").strip(),
                "evidence_texts": [
                    str(fact.get("evidence_text") or fact.get("fact") or "").strip()
                ],
                "evidence_type": _evidence_type(fact),
            }
        )
    return basis


def _source_article_ids(fact: dict[str, Any]) -> list[int]:
    raw = fact.get("source_article_ids")
    if isinstance(raw, list | tuple | set):
        values = raw
    else:
        values = [fact.get("article_id") or fact.get("raw_article_id")]
    out: list[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out


def _raw_article_ids(fact: dict[str, Any]) -> list[int]:
    raw = fact.get("raw_article_ids")
    if isinstance(raw, list | tuple | set):
        values = raw
    else:
        values = [
            fact.get("raw_article_id")
            or fact.get("article_id")
            or _first_source_article_id(fact)
        ]
    out: list[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out


def _first_source_article_id(fact: dict[str, Any]) -> Any:
    raw = fact.get("source_article_ids")
    if isinstance(raw, list | tuple | set):
        for value in raw:
            return value
    return None


def _confidence_from_fact(fact: dict[str, Any]) -> float:
    raw = fact.get("confidence")
    if raw is None:
        return round(float(fact.get("integration_rank_score") or 0.0), 3)
    try:
        return round(max(0.0, min(float(raw), 1.0)), 3)
    except (TypeError, ValueError):
        return round(float(fact.get("integration_rank_score") or 0.0), 3)


def _evidence_type(fact: dict[str, Any]) -> str:
    fact_type = str(fact.get("fact_type") or "")
    if fact_type == "financial_metric":
        return "numeric_fact"
    if fact_type == "risk_fact":
        return "risk_fact"
    if fact.get("numbers_and_dates"):
        return "numeric_fact"
    if fact.get("derived_from") in {"financial_metric", "business_signal", "parser_chunk"}:
        return "reported_fact"
    return "reported_fact"


__all__ = ["build_claim_ledger", "build_evidence_ledger", "build_fact_basis"]
