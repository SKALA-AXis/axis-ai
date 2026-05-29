"""Quality assessment for IntegratedIssue v2."""

from __future__ import annotations

from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy
from src.services.issue_integration.source_profile import SourceProfile


def build_quality_report(
    *,
    input_bundle: AnalysisInputBundle,
    selected_facts: list[dict[str, Any]],
    all_facts: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]],
    source_profile: SourceProfile,
    selection: dict[str, Any],
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    source_count = len(input_bundle.sources or [])
    row_status = _source_row_status(input_bundle.sources or [], policy=policy)
    source_ids = {
        source_id
        for source_id in (_source_id(source) for source in (input_bundle.sources or []))
        if source_id > 0
    }
    fact_source_ids = {
        _safe_int(fact.get("raw_article_id") or fact.get("article_id"))
        for fact in selected_facts
        if _safe_int(fact.get("raw_article_id") or fact.get("article_id")) > 0
    }
    source_coverage = (
        len(fact_source_ids & source_ids) / len(source_ids)
        if source_ids
        else (1.0 if selected_facts else 0.0)
    )
    structured_signal_presence = 1.0 if _has_structured_signal(selected_facts) else 0.0
    evidence_presence = 1.0 if evidence_ledger else 0.0
    fact_presence = 1.0 if selected_facts else 0.0
    uncertainty_ratio = _uncertainty_ratio(selected_facts, policy=policy)
    weights = policy.quality_weights
    confidence = (
        weights["fact_presence"] * fact_presence
        + weights["evidence_presence"] * evidence_presence
        + weights["source_coverage"] * source_coverage
        + weights["structured_signal_presence"] * structured_signal_presence
        - weights["uncertainty_penalty"] * uncertainty_ratio
    )
    confidence = round(max(0.0, min(confidence, 1.0)), 3)
    review_flags = _review_flags(
        input_bundle=input_bundle,
        selected_facts=selected_facts,
        evidence_ledger=evidence_ledger,
        source_profile=source_profile,
        row_status=row_status,
    )
    return {
        "fact_count": len(all_facts),
        "selected_fact_count": len(selected_facts),
        "evidence_count": len(evidence_ledger),
        "source_count": source_count,
        "eligible_source_count": row_status["eligible_source_count"],
        "skipped_source_count": row_status["skipped_source_count"],
        "irrelevant_source_count": row_status["irrelevant_source_count"],
        "crawl_error_source_count": row_status["crawl_error_source_count"],
        "source_row_statuses": row_status["source_row_statuses"],
        "source_coverage": round(source_coverage, 3),
        "has_parser_result": any(_has_parser_result(item) for item in input_bundle.items),
        "has_structured_metrics": any(
            isinstance(item, dict) and bool(item.get("financial_metrics"))
            for item in input_bundle.items
        ),
        "has_business_signals": any(
            isinstance(item, dict) and bool(item.get("business_signals"))
            for item in input_bundle.items
        ),
        "confidence": confidence,
        "review_flags": review_flags,
        "selection": selection,
    }


def missing_or_uncertain_points(
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for fact in facts:
        text = f"{fact.get('fact') or ''} {fact.get('evidence_text') or ''}".lower()
        markers = [marker for marker in policy.uncertainty_markers if marker.lower() in text]
        if not markers:
            continue
        points.append(
            {
                "fact_id": fact.get("fact_id"),
                "uncertainty_markers": markers,
                "caution": "원문이 계획/전망/가능성 표현을 포함하므로 확정 사실로 해석하지 않는다.",
            }
        )
    return points


def _review_flags(
    *,
    input_bundle: AnalysisInputBundle,
    selected_facts: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]],
    source_profile: SourceProfile,
    row_status: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if not selected_facts:
        flags.append("no_selected_facts")
    if selected_facts and not evidence_ledger:
        flags.append("missing_evidence_ledger")
    if input_bundle.sources and row_status["eligible_source_count"] == 0:
        flags.append("no_analysis_eligible_source_rows")
    if row_status["skipped_source_count"] > 0:
        flags.append("contains_skipped_source_rows")
    if row_status["irrelevant_source_count"] > 0:
        flags.append("contains_irrelevant_source_rows")
    if row_status["crawl_error_source_count"] > 0:
        flags.append("contains_crawl_error_source_rows")
    if source_profile.scope_type == "peer_company" and not input_bundle.companies:
        flags.append("peer_scope_without_company")
    if source_profile.source_family == "unknown":
        flags.append("unknown_source_family")
    return flags


def _source_row_status(
    sources: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    eligible_count = 0
    skipped_count = 0
    irrelevant_count = 0
    crawl_error_count = 0
    for source in sources:
        processing_status = str(source.get("processing_status") or "").strip().lower()
        relevance_label = str(source.get("relevance_label") or "").strip().lower()
        crawl_status = str(source.get("crawl_status") or "").strip().lower()
        has_error = bool(source.get("error_message")) or (
            bool(crawl_status) and crawl_status not in policy.eligible_crawl_statuses
        )
        eligible = bool(source.get("is_analysis_eligible", True))
        if processing_status in policy.ineligible_processing_statuses:
            eligible = False
        if relevance_label in policy.ineligible_relevance_labels:
            eligible = False
        if has_error:
            eligible = False
        if processing_status in policy.ineligible_processing_statuses:
            skipped_count += 1
        if relevance_label in policy.ineligible_relevance_labels:
            irrelevant_count += 1
        if has_error:
            crawl_error_count += 1
        if eligible:
            eligible_count += 1
        statuses.append(
            {
                "raw_article_id": _source_id(source),
                "processing_status": source.get("processing_status"),
                "crawl_status": source.get("crawl_status"),
                "error_message": source.get("error_message"),
                "relevance_label": source.get("relevance_label"),
                "relevance_score": source.get("relevance_score"),
                "is_analysis_eligible": eligible,
            }
        )
    return {
        "eligible_source_count": eligible_count,
        "skipped_source_count": skipped_count,
        "irrelevant_source_count": irrelevant_count,
        "crawl_error_source_count": crawl_error_count,
        "source_row_statuses": statuses,
    }


def _uncertainty_ratio(
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
) -> float:
    if not facts:
        return 0.0
    uncertain = 0
    for fact in facts:
        text = f"{fact.get('fact') or ''} {fact.get('evidence_text') or ''}".lower()
        if any(marker.lower() in text for marker in policy.uncertainty_markers):
            uncertain += 1
    return uncertain / len(facts)


def _has_structured_signal(facts: list[dict[str, Any]]) -> bool:
    return any(
        fact.get("derived_from") in {"financial_metric", "business_signal", "parser_chunk"}
        for fact in facts
    )


def _has_parser_result(item: dict[str, Any]) -> bool:
    parser_result = item.get("parser_result")
    if isinstance(parser_result, dict) and parser_result:
        return True
    metadata = item.get("metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("parser_result"), dict)


def _source_id(source: dict[str, Any]) -> int:
    return _safe_int(source.get("raw_article_id") or source.get("article_id"))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_quality_report", "missing_or_uncertain_points"]
