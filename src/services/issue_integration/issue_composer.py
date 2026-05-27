"""IntegratedIssue v2 composer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration.content_digest import build_content_payload
from src.services.issue_integration.evidence_ledger import (
    build_claim_ledger,
    build_evidence_ledger,
    build_fact_basis,
)
from src.services.issue_integration.fact_ranker import select_facts_for_issue
from src.services.issue_integration.issue_frame import build_issue_frame
from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy
from src.services.issue_integration.quality import (
    build_quality_report,
    missing_or_uncertain_points,
)
from src.services.issue_integration.source_map import build_source_map
from src.services.issue_integration.source_profile import SourceProfile, infer_source_profile


class IntegratedIssueComposer:
    """Compose and enrich IntegratedIssue payloads without adding interpretation."""

    def __init__(self, *, policy: IntegrationPolicy = DEFAULT_POLICY) -> None:
        self.policy = policy

    def compose_non_news(self, input_bundle: AnalysisInputBundle) -> dict[str, Any]:
        source_profile = infer_source_profile(input_bundle)
        content_payload = build_content_payload(input_bundle, policy=self.policy)
        source_map = build_source_map(
            input_bundle,
            content_digest=content_payload["content_digest"],
        )
        candidate_facts = _analysis_candidate_facts(input_bundle)
        selected_facts, selection = select_facts_for_issue(
            candidate_facts,
            policy=self.policy,
        )
        evidence_ledger = build_evidence_ledger(selected_facts)
        claim_ledger = build_claim_ledger(selected_facts)
        fact_summary = _build_fact_summary(selected_facts, policy=self.policy)
        document_subject = _document_subject(input_bundle, selected_facts, source_profile)
        integrated_text = _integrated_text(
            document_subject=document_subject,
            fact_summary=fact_summary,
            selected_facts=selected_facts,
            policy=self.policy,
        )
        quality = build_quality_report(
            input_bundle=input_bundle,
            selected_facts=selected_facts,
            all_facts=input_bundle.facts,
            evidence_ledger=evidence_ledger,
            source_profile=source_profile,
            selection=selection,
            policy=self.policy,
        )
        is_valid = _is_valid_issue(
            selected_facts=selected_facts,
            integrated_text=integrated_text,
            source_profile=source_profile,
            input_bundle=input_bundle,
            quality=quality,
        )
        reason = "" if is_valid else _invalid_reason(source_profile=source_profile, quality=quality)
        source_article_ids = _item_ids(input_bundle.items)
        raw_article_ids = _raw_item_ids(input_bundle.items)
        main_company = input_bundle.companies[0] if input_bundle.companies else ""
        issue_frame = build_issue_frame(
            input_bundle=input_bundle,
            source_profile=source_profile,
            facts=selected_facts,
            content_digest=content_payload["content_digest"],
            source_map=source_map,
            policy=self.policy,
        )
        return {
            "schema_version": "integrated_issue_v2",
            "cluster_id": _safe_int(input_bundle.cluster_id),
            "representative_id": _representative_id(input_bundle),
            "source_article_ids": source_article_ids,
            "raw_article_ids": raw_article_ids,
            "cluster_article_ids": source_article_ids,
            "analyzed_article_ids": source_article_ids,
            "summary_scope": "integrated_issue",
            "is_valid_summary": is_valid,
            "main_company": main_company,
            "mentioned_peer_companies": list(input_bundle.companies),
            "mentioned_sectors": list(input_bundle.sectors),
            "sectors": list(input_bundle.sectors),
            "cluster_event_type": input_bundle.event_type or "general_update",
            "event_type": input_bundle.event_type or "general_update",
            "headline": document_subject,
            "one_line_summary": fact_summary[0] if fact_summary else document_subject,
            "main_event": document_subject,
            "main_issue": document_subject,
            "document_subject": document_subject,
            "source_family": source_profile.source_family,
            "scope_type": source_profile.scope_type,
            "source_profile": source_profile.to_dict(),
            "content_digest": content_payload["content_digest"],
            "content_digest_storage": content_payload["content_digest_storage"],
            "source_map": source_map,
            "issue_frame": issue_frame,
            "integrated_text": integrated_text,
            "fact_summary": fact_summary,
            "consolidated_facts": _consolidated_facts(selected_facts),
            "key_numbers": _key_numbers(selected_facts),
            "business_signals": _business_signals(selected_facts),
            "claim_ledger": claim_ledger,
            "evidence_ledger": evidence_ledger,
            "fact_basis": build_fact_basis(selected_facts),
            "missing_or_uncertain_points": missing_or_uncertain_points(
                selected_facts,
                policy=self.policy,
            ),
            "source_count": len(input_bundle.sources),
            "source_coverage": _source_coverage(input_bundle, selected_facts),
            "quality": quality,
            "confidence": quality["confidence"] if is_valid else 0.0,
            "reason": reason,
            "integrated_at": datetime.now(UTC).isoformat(),
        }

    def enrich_existing_summary(
        self,
        summary: dict[str, Any],
        *,
        input_bundle: AnalysisInputBundle,
    ) -> dict[str, Any]:
        source_profile = infer_source_profile(input_bundle)
        content_payload = build_content_payload(input_bundle, policy=self.policy)
        content_digest = summary.get("content_digest") or content_payload["content_digest"]
        source_map = summary.get("source_map") or build_source_map(
            input_bundle,
            content_digest=content_digest,
        )
        facts = input_bundle.facts or _facts_from_existing_summary(summary)
        candidate_facts = _analysis_candidate_facts(input_bundle, facts=facts)
        selected_facts, selection = select_facts_for_issue(candidate_facts, policy=self.policy)
        evidence_ledger = build_evidence_ledger(selected_facts)
        claim_ledger = build_claim_ledger(selected_facts)
        quality = build_quality_report(
            input_bundle=input_bundle,
            selected_facts=selected_facts,
            all_facts=facts,
            evidence_ledger=evidence_ledger,
            source_profile=source_profile,
            selection=selection,
            policy=self.policy,
        )
        main_issue = _first_non_empty(
            summary.get("main_issue"),
            summary.get("main_topic"),
            summary.get("headline"),
            summary.get("one_line_summary"),
            summary.get("main_event"),
        )
        integrated_text = _first_non_empty(
            summary.get("integrated_text"),
            summary.get("summary"),
            summary.get("one_line_summary"),
            " ".join(_normalize_string_list(summary.get("fact_summary"))),
            main_issue,
        )
        consolidated = summary.get("consolidated_facts") or summary.get("key_facts")
        if not consolidated:
            consolidated = _consolidated_facts(selected_facts)
        fact_basis = summary.get("fact_basis") or build_fact_basis(selected_facts)
        issue_frame = summary.get("issue_frame") or build_issue_frame(
            input_bundle=input_bundle,
            source_profile=source_profile,
            facts=selected_facts,
            content_digest=content_digest,
            source_map=source_map,
            policy=self.policy,
        )
        return {
            **summary,
            "schema_version": "integrated_issue_v2",
            "issue_component": "IssueIntegrationAgent",
            "integration_component": "IssueIntegrationAgent",
            "integration_input": "analysis_input_bundle",
            "bundle_id": input_bundle.bundle_id,
            "issue_source_type": input_bundle.source_type,
            "source_family": source_profile.source_family,
            "scope_type": source_profile.scope_type,
            "source_profile": source_profile.to_dict(),
            "content_digest": content_digest,
            "content_digest_storage": summary.get("content_digest_storage")
            or content_payload["content_digest_storage"],
            "source_map": source_map,
            "issue_frame": issue_frame,
            "source_article_ids": summary.get("source_article_ids")
            or _item_ids(input_bundle.items),
            "raw_article_ids": summary.get("raw_article_ids") or _raw_item_ids(input_bundle.items),
            "main_issue": str(main_issue),
            "integrated_text": str(integrated_text),
            "document_subject": _first_non_empty(summary.get("document_subject"), main_issue),
            "consolidated_facts": consolidated,
            "business_signals": summary.get("business_signals", _business_signals(selected_facts)),
            "key_numbers": summary.get("key_numbers", _key_numbers(selected_facts)),
            "claim_ledger": summary.get("claim_ledger") or claim_ledger,
            "evidence_ledger": summary.get("evidence_ledger") or evidence_ledger,
            "fact_basis": fact_basis,
            "missing_or_uncertain_points": summary.get("missing_or_uncertain_points")
            or missing_or_uncertain_points(selected_facts, policy=self.policy),
            "quality": {**quality, **(summary.get("quality") or {})},
            "input_bundle_ref": _input_bundle_ref(input_bundle),
        }


def _build_fact_summary(
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
) -> list[str]:
    selected: list[str] = []
    used_chars = 0
    for fact in _coverage_ordered_facts(facts):
        line = str(fact.get("fact") or fact.get("evidence_text") or "").strip()
        if not line or line in selected:
            continue
        projected = used_chars + len(line)
        if selected and projected > policy.fact_summary_chars:
            continue
        selected.append(line)
        used_chars = projected
    return selected


def _analysis_candidate_facts(
    input_bundle: AnalysisInputBundle,
    *,
    facts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source_eligibility = _source_eligibility_by_raw_id(input_bundle)
    candidate_facts = facts if facts is not None else input_bundle.facts
    if not source_eligibility:
        return candidate_facts
    return [
        fact
        for fact in candidate_facts
        if _fact_source_id(fact) <= 0 or source_eligibility.get(_fact_source_id(fact), True)
    ]


def _source_eligibility_by_raw_id(input_bundle: AnalysisInputBundle) -> dict[int, bool]:
    if not any("is_analysis_eligible" in source for source in input_bundle.sources or []):
        return {}
    eligibility: dict[int, bool] = {}
    for source in input_bundle.sources or []:
        source_id = _safe_int(source.get("raw_article_id") or source.get("article_id"))
        if source_id <= 0:
            continue
        eligibility[source_id] = bool(source.get("is_analysis_eligible", True))
    return eligibility


def _fact_source_id(fact: dict[str, Any]) -> int:
    return _safe_int(fact.get("raw_article_id") or fact.get("article_id"))


def _coverage_ordered_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        grouped.setdefault(str(fact.get("fact_type") or "general_fact"), []).append(fact)
    for key, values in grouped.items():
        grouped[key] = sorted(
            values,
            key=lambda item: float(item.get("integration_rank_score") or 0.0),
            reverse=True,
        )
    ordered: list[dict[str, Any]] = []
    while grouped:
        for key in sorted(grouped):
            values = grouped.get(key) or []
            if not values:
                grouped.pop(key, None)
                continue
            ordered.append(values.pop(0))
    return ordered


def _integrated_text(
    *,
    document_subject: str,
    fact_summary: list[str],
    selected_facts: list[dict[str, Any]],
    policy: IntegrationPolicy,
) -> str:
    parts = [document_subject, *fact_summary]
    if not any(parts):
        parts = [str(fact.get("fact") or "") for fact in selected_facts]
    text = " ".join(part for part in parts if part)
    if len(text) <= policy.target_evidence_chars:
        return text
    boundary = text.rfind(" ", 0, policy.target_evidence_chars)
    return text[: boundary if boundary > 0 else policy.target_evidence_chars].rstrip()


def _document_subject(
    input_bundle: AnalysisInputBundle,
    facts: list[dict[str, Any]],
    source_profile: SourceProfile,
) -> str:
    for item in input_bundle.items:
        title = str(item.get("title") or "").strip()
        if title:
            return title
    if facts:
        return str(facts[0].get("fact") or facts[0].get("evidence_text") or "").strip()
    if input_bundle.sectors:
        return f"{input_bundle.sectors[0]} {source_profile.source_family} 자료"
    return f"{source_profile.source_family} 분석 대상"


def _is_valid_issue(
    *,
    selected_facts: list[dict[str, Any]],
    integrated_text: str,
    source_profile: SourceProfile,
    input_bundle: AnalysisInputBundle,
    quality: dict[str, Any],
) -> bool:
    if not selected_facts or not integrated_text:
        return False
    if input_bundle.sources and quality.get("eligible_source_count") == 0:
        return False
    if source_profile.scope_type == "peer_company":
        return bool(input_bundle.companies)
    return True


def _invalid_reason(*, source_profile: SourceProfile, quality: dict[str, Any]) -> str:
    flags = quality.get("review_flags") or []
    if flags:
        return f"통합 이슈 구성 품질 기준 미충족: {', '.join(str(flag) for flag in flags)}"
    if source_profile.source_family == "unknown":
        return "자료 유형을 판정하지 못했습니다."
    return "통합 이슈를 구성할 근거가 부족합니다."


def _consolidated_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fact_id": fact.get("fact_id"),
            "fact": fact.get("fact"),
            "source_article_ids": _source_article_ids_from_fact(fact),
            "raw_article_ids": _raw_article_ids_from_fact(fact),
            "evidence_texts": [fact.get("evidence_text")] if fact.get("evidence_text") else [],
            "source_type": fact.get("source_type"),
            "fact_type": fact.get("fact_type") or "general_fact",
            "integration_rank_score": fact.get("integration_rank_score"),
        }
        for fact in facts
    ]


def _key_numbers(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("fact_type") != "financial_metric" and not fact.get("numbers_and_dates"):
            continue
        numbers.append(
            {
                "article_id": fact.get("article_id"),
                "raw_article_id": fact.get("raw_article_id") or fact.get("article_id"),
                "raw_article_ids": _raw_article_ids_from_fact(fact),
                "metric_name": fact.get("metric_name"),
                "metric_label": fact.get("metric_label"),
                "value": fact.get("value"),
                "unit": fact.get("unit"),
                "period": fact.get("period"),
                "evidence_text": fact.get("evidence_text"),
                "fact_id": fact.get("fact_id"),
                "numbers_and_dates": fact.get("numbers_and_dates") or [],
            }
        )
    return numbers


def _business_signals(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("fact_type") not in {"business_signal", "market_fact", "risk_fact"}:
            continue
        signals.append(
            {
                "article_id": fact.get("article_id"),
                "raw_article_id": fact.get("raw_article_id") or fact.get("article_id"),
                "raw_article_ids": _raw_article_ids_from_fact(fact),
                "business_area": fact.get("business_area"),
                "signal_type": fact.get("signal_type") or fact.get("fact_type"),
                "sentiment": fact.get("sentiment"),
                "summary": fact.get("fact"),
                "evidence_text": fact.get("evidence_text"),
                "confidence": fact.get("confidence")
                or fact.get("integration_rank_score")
                or 0.0,
                "fact_id": fact.get("fact_id"),
            }
        )
    return signals


def _source_coverage(
    input_bundle: AnalysisInputBundle,
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = [
        _safe_int(source.get("raw_article_id") or source.get("article_id"))
        for source in input_bundle.sources or []
        if _safe_int(source.get("raw_article_id") or source.get("article_id")) > 0
    ]
    fact_source_ids = [
        _safe_int(fact.get("raw_article_id") or fact.get("article_id"))
        for fact in facts
        if _safe_int(fact.get("raw_article_id") or fact.get("article_id")) > 0
    ]
    covered = sorted({source_id for source_id in source_ids if source_id in fact_source_ids})
    return {
        "source_article_ids": source_ids,
        "raw_article_ids": source_ids,
        "covered_source_article_ids": covered,
        "covered_raw_article_ids": covered,
        "uncovered_source_article_ids": [
            source_id for source_id in source_ids if source_id not in covered
        ],
        "uncovered_raw_article_ids": [
            source_id for source_id in source_ids if source_id not in covered
        ],
    }


def _facts_from_existing_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, item in enumerate(summary.get("consolidated_facts") or [], start=1):
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact") or "").strip()
        if not fact_text:
            continue
        fact_id = str(item.get("fact_id") or f"summary:fact:{index}")
        facts.append(
            {
                "fact_id": fact_id,
                "article_id": _first_source_article_id(item),
                "raw_article_id": _first_raw_article_id(item),
                "source_article_ids": item.get("source_article_ids") or [],
                "raw_article_ids": item.get("raw_article_ids") or [],
                "fact": fact_text,
                "evidence_text": " ".join(_normalize_string_list(item.get("evidence_texts")))
                or fact_text,
                "source_type": item.get("source_type"),
                "fact_type": item.get("fact_type") or "general_fact",
                "derived_from": "existing_summary",
            }
        )
    return facts


def _input_bundle_ref(input_bundle: AnalysisInputBundle) -> dict[str, Any]:
    return {
        "bundle_id": input_bundle.bundle_id,
        "cluster_id": input_bundle.cluster_id,
        "source_type": input_bundle.source_type,
        "source_count": len(input_bundle.sources),
        "raw_article_ids": _raw_item_ids(input_bundle.items),
    }


def _source_article_ids_from_fact(fact: dict[str, Any]) -> list[int]:
    raw = fact.get("source_article_ids")
    values = (
        raw
        if isinstance(raw, list | tuple | set)
        else [fact.get("article_id") or fact.get("raw_article_id")]
    )
    ids: list[int] = []
    for value in values:
        number = _safe_int(value)
        if number > 0 and number not in ids:
            ids.append(number)
    return ids


def _raw_article_ids_from_fact(fact: dict[str, Any]) -> list[int]:
    raw = fact.get("raw_article_ids")
    if isinstance(raw, list | tuple | set):
        values = raw
    else:
        values = [
            fact.get("raw_article_id")
            or fact.get("article_id")
            or _first_source_article_id(fact)
        ]
    ids: list[int] = []
    for value in values:
        number = _safe_int(value)
        if number > 0 and number not in ids:
            ids.append(number)
    return ids


def _first_source_article_id(item: dict[str, Any]) -> int:
    ids = item.get("source_article_ids")
    if isinstance(ids, list) and ids:
        return _safe_int(ids[0])
    return _safe_int(item.get("article_id"))


def _first_raw_article_id(item: dict[str, Any]) -> int:
    ids = item.get("raw_article_ids")
    if isinstance(ids, list) and ids:
        return _safe_int(ids[0])
    return _safe_int(item.get("raw_article_id") or item.get("article_id"))


def _representative_id(input_bundle: AnalysisInputBundle) -> int:
    value = input_bundle.metadata.get("representative_id")
    representative_id = _safe_int(value)
    if representative_id > 0:
        return representative_id
    item_ids = _item_ids(input_bundle.items)
    return item_ids[0] if item_ids else 0


def _item_ids(items: list[dict[str, Any]]) -> list[int]:
    return _dedupe_ints(_item_id(item) for item in items)


def _raw_item_ids(items: list[dict[str, Any]]) -> list[int]:
    return _dedupe_ints(_raw_item_id(item) for item in items)


def _item_id(item: dict[str, Any]) -> int:
    return _safe_int(
        item.get("article_id")
        or item.get("raw_article_id")
        or item.get("id")
        or item.get("preprocess_id")
    )


def _raw_item_id(item: dict[str, Any]) -> int:
    return _safe_int(
        item.get("raw_article_id")
        or item.get("id")
        or item.get("preprocess_id")
        or item.get("article_id")
    )


def _dedupe_ints(values: Any) -> list[int]:
    ids: list[int] = []
    for value in values:
        number = _safe_int(value)
        if number > 0 and number not in ids:
            ids.append(number)
    return ids


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["IntegratedIssueComposer"]
