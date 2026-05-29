"""IntegratedIssue v2 composer."""

from __future__ import annotations

import re
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
        return _public_issue_payload({
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
            "sources": source_map.get("sources", []),
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
        })

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
        return _public_issue_payload({
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
            "sources": summary.get("sources") or source_map.get("sources", []),
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
        })


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
        "total_source_count": len(source_ids),
        "covered_source_count": len(covered),
        "uncovered_source_count": len(source_ids) - len(covered),
    }


def _public_issue_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text_refs = _EvidenceTextRegistry()
    issue_frame = _public_issue_frame(payload.get("issue_frame", {}), text_refs=text_refs)
    evidence = _public_evidence(payload, text_refs=text_refs)
    return {
        "schema_version": "integrated_issue_v3",
        "issue_brief": _issue_brief(payload),
        "analysis_ready_inputs": _analysis_ready_inputs(payload),
        "content_digest": payload.get("content_digest", {}),
        "issue_frame": issue_frame,
        "sources": payload.get("sources", []),
        "evidence": evidence,
        "quality": _public_quality(payload.get("quality", {})),
        "metadata": _public_metadata(payload),
    }


def _issue_brief(payload: dict[str, Any]) -> dict[str, Any]:
    analyzed_ids = _normalize_int_list(
        payload.get("analyzed_article_ids")
        or payload.get("source_article_ids")
        or payload.get("raw_article_ids")
    )
    return {
        "is_valid": bool(payload.get("is_valid_summary", True)),
        "headline": _first_non_empty(payload.get("headline"), payload.get("main_issue")),
        "one_line_summary": _first_non_empty(
            payload.get("one_line_summary"),
            payload.get("integrated_text"),
        ),
        "main_company": payload.get("main_company", ""),
        "mentioned_peer_companies": payload.get("mentioned_peer_companies", []),
        "event_type": payload.get("cluster_event_type") or payload.get("event_type"),
        "sectors": payload.get("sectors") or payload.get("mentioned_sectors") or [],
        "source_family": payload.get("source_family", ""),
        "scope_type": payload.get("scope_type", ""),
        "analysis_scope": {
            "analyzed_source_ids": analyzed_ids,
        },
        "confidence": payload.get("confidence", 0.0),
        "reason": payload.get("reason", ""),
    }


def _analysis_ready_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content_digest") if isinstance(payload.get("content_digest"), dict) else {}
    sections = [
        {
            "section": section.get("section"),
            "title": section.get("title"),
            "summary": section.get("summary", ""),
            "keywords": section.get("keywords", []),
        }
        for section in _as_dict_list(content.get("sections"))
    ]
    return {
        "core_question": _core_question(payload),
        "key_developments": _key_developments(payload),
        "materiality_signals": _materiality_signals(payload),
        "uncertainty_points": _uncertainty_points(payload),
        "suggested_sections": sections,
    }


def _public_evidence(
    payload: dict[str, Any],
    *,
    text_refs: "_EvidenceTextRegistry",
) -> dict[str, Any]:
    facts = _public_evidence_facts(payload, text_refs=text_refs)
    by_section: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        section = str(fact.get("section") or "general_fact")
        by_section.setdefault(section, []).append(fact)
    return {
        "references": text_refs.items(),
        "by_section": [
            {
                "section": section,
                "title": _evidence_section_title(section),
                "facts": section_facts,
            }
            for section, section_facts in by_section.items()
        ],
        "claims": _public_claims(payload.get("claim_ledger")),
        "uncertain_points": _uncertainty_points(payload),
    }


def _public_evidence_facts(
    payload: dict[str, Any],
    *,
    text_refs: "_EvidenceTextRegistry",
) -> list[dict[str, Any]]:
    extracted = payload.get("extracted_facts") or []
    facts = [
        _public_extracted_fact(fact, text_refs=text_refs)
        for fact in _as_dict_list(extracted)
    ]
    facts = [fact for fact in facts if fact.get("fact")]
    if facts:
        return facts
    basis = payload.get("fact_basis") or payload.get("consolidated_facts") or []
    return [
        fact
        for fact in (
            _public_basis_fact(item, text_refs=text_refs)
            for item in _as_dict_list(basis)
        )
        if fact.get("fact")
    ]


def _public_extracted_fact(
    fact: dict[str, Any],
    *,
    text_refs: "_EvidenceTextRegistry",
) -> dict[str, Any]:
    fact_text = _first_non_empty(fact.get("normalized_fact"), fact.get("fact"))
    evidence_text = str(fact.get("evidence_text") or "").strip()
    source_ids = _normalize_int_list(fact.get("article_id"))
    out: dict[str, Any] = {
        "fact_id": fact.get("fact_id"),
        "fact": fact_text,
        "source_ids": source_ids,
        "section": _evidence_section(fact),
        "fact_type": fact.get("fact_type"),
        "summary_role": fact.get("summary_role"),
        "entities": _clean_entities(fact.get("entities")),
        "numbers": fact.get("numbers", []),
        "dates": fact.get("dates", []),
        "confidence": fact.get("confidence"),
    }
    if evidence_text and _compact_text(evidence_text) != _compact_text(fact_text):
        ref_id = text_refs.add(evidence_text, source_ids=source_ids)
        if ref_id:
            out["evidence_ref_id"] = ref_id
    return out


def _public_basis_fact(
    item: dict[str, Any],
    *,
    text_refs: "_EvidenceTextRegistry",
) -> dict[str, Any]:
    evidence_texts = _normalize_string_list(item.get("evidence_texts"))
    evidence_text = evidence_texts[0] if evidence_texts else str(item.get("evidence_text") or "")
    fact_text = _first_non_empty(item.get("fact"), item.get("claim"), evidence_text)
    source_ids = _normalize_int_list(item.get("source_article_ids") or item.get("raw_article_ids"))
    out: dict[str, Any] = {
        "fact_id": (item.get("fact_ids") or [item.get("fact_id")])[0]
        if isinstance(item.get("fact_ids"), list)
        else item.get("fact_id"),
        "fact": fact_text,
        "source_ids": source_ids,
        "section": _evidence_section(item),
        "fact_type": item.get("fact_type"),
        "summary_role": item.get("summary_role"),
        "numbers": _numbers_from_text(f"{fact_text} {evidence_text}"),
        "dates": [],
        "confidence": item.get("confidence"),
    }
    if evidence_text and _compact_text(evidence_text) != _compact_text(fact_text):
        ref_id = text_refs.add(evidence_text, source_ids=source_ids)
        if ref_id:
            out["evidence_ref_id"] = ref_id
    return out


def _public_claims(value: Any) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for claim in _as_dict_list(value):
        claims.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim": claim.get("claim"),
                "claim_type": claim.get("claim_type"),
                "fact_ids": claim.get("fact_ids", []),
                "confidence": claim.get("confidence"),
            }
        )
    return claims


def _public_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": payload.get("cluster_id"),
        "representative_id": payload.get("representative_id"),
        "bundle_id": payload.get("bundle_id"),
        "summary_scope": payload.get("summary_scope"),
        "issue_component": payload.get("issue_component"),
        "integration_component": payload.get("integration_component"),
        "integration_input": payload.get("integration_input"),
        "issue_source_type": payload.get("issue_source_type"),
        "model": payload.get("model"),
        "coverage": payload.get("coverage"),
        "source_count": payload.get("source_count"),
        "cluster_source_count": payload.get("cluster_source_count"),
        "validation_warnings": payload.get("validation_warnings", []),
        "fact_extraction_failed": bool(payload.get("fact_extraction_failed", False)),
        "integrated_at": payload.get("integrated_at"),
        "input_bundle_ref": payload.get("input_bundle_ref", {}),
    }


def _public_quality(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return _strip_internal_refs(value)


def _core_question(payload: dict[str, Any]) -> str:
    company = _first_non_empty(payload.get("main_company"), "해당 자료")
    event_type = _first_non_empty(payload.get("cluster_event_type"), payload.get("event_type"))
    topic = _first_non_empty(payload.get("headline"), payload.get("main_issue"), payload.get("one_line_summary"))
    if topic:
        return f"{company}의 {event_type or '이슈'}에서 '{topic}'이 보여주는 사업/시장 의미는 무엇인가?"
    return f"{company}의 {event_type or '이슈'}가 보여주는 사업/시장 의미는 무엇인가?"


def _key_developments(payload: dict[str, Any]) -> list[str]:
    values = _normalize_string_list(payload.get("fact_summary"))
    if not values:
        values = [
            str(item.get("fact") or "")
            for item in _as_dict_list(payload.get("consolidated_facts"))
        ]
    if not values:
        content = payload.get("content_digest") if isinstance(payload.get("content_digest"), dict) else {}
        values = [
            str(point.get("point") or point.get("fact") or "")
            for point in _as_dict_list(content.get("key_points"))
        ]
    return _dedupe_strings(values)[:5]


def _materiality_signals(payload: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    content = payload.get("content_digest") if isinstance(payload.get("content_digest"), dict) else {}
    for section in _as_dict_list(content.get("sections")):
        signals.append(
            {
                "type": section.get("section") or "content_section",
                "title": section.get("title") or section.get("section"),
                "keywords": section.get("keywords", []),
                "numbers_and_dates": section.get("numbers_and_dates", []),
            }
        )
    for number in _as_dict_list(payload.get("key_numbers")):
        signals.append(
            {
                "type": "key_number",
                "metric_name": number.get("metric_name"),
                "metric_label": number.get("metric_label"),
                "value": number.get("value"),
                "unit": number.get("unit"),
                "period": number.get("period"),
                "numbers_and_dates": number.get("numbers_and_dates", []),
            }
        )
    for signal in _as_dict_list(payload.get("business_signals")):
        signals.append({"type": "business_signal", **signal})
    return signals[:8]


def _public_issue_frame(
    value: Any,
    *,
    text_refs: "_EvidenceTextRegistry",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return _strip_internal_refs(value, text_refs=text_refs)


def _strip_internal_refs(
    value: Any,
    *,
    text_refs: "_EvidenceTextRegistry | None" = None,
) -> Any:
    if isinstance(value, dict):
        omitted = {
            "raw_article_id",
            "raw_article_ids",
            "basis_raw_article_ids",
            "eligible_raw_article_ids",
            "source_article_ids",
            "source_indexes",
            "source_index",
            "source_fields",
        }
        source_ids = _normalize_int_list(
            value.get("source_ids")
            or value.get("raw_article_ids")
            or value.get("source_article_ids")
        )
        public: dict[str, Any] = {}
        for key, item in value.items():
            if key in omitted:
                continue
            if key == "evidence_text":
                ref_id = text_refs.add(item, source_ids=source_ids) if text_refs else ""
                if ref_id:
                    public["evidence_ref_id"] = ref_id
                continue
            public[key] = _strip_internal_refs(item, text_refs=text_refs)
        return public
    if isinstance(value, list):
        return [_strip_internal_refs(item, text_refs=text_refs) for item in value]
    return value


def _uncertainty_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points = [
        point for point in _as_dict_list(payload.get("missing_or_uncertain_points"))
    ]
    for fact in _as_dict_list(payload.get("extracted_facts")):
        if fact.get("fact_type") == "uncertain_fact" or fact.get("summary_role") == "uncertainty_detail":
            points.append(
                {
                    "fact_id": fact.get("fact_id"),
                    "point": _first_non_empty(fact.get("normalized_fact"), fact.get("evidence_text")),
                }
            )
    return points[:6]


def _evidence_section(fact: dict[str, Any]) -> str:
    role = str(fact.get("summary_role") or "")
    fact_type = str(fact.get("fact_type") or "")
    if role in {"main_event", "product_definition", "service_function", "application_case", "numeric_effect"}:
        return role
    if fact_type in {"risk_fact", "uncertain_fact"} or role in {"risk_detail", "uncertainty_detail"}:
        return "risk_or_uncertainty"
    if fact_type:
        return fact_type
    return "general_fact"


def _evidence_section_title(section: str) -> str:
    return {
        "main_event": "핵심 사건",
        "product_definition": "제품/서비스 정의",
        "service_function": "기능/역할",
        "application_case": "적용 사례",
        "numeric_effect": "수치/효과",
        "risk_or_uncertainty": "리스크/불확실성",
        "business_signal": "사업 변화",
        "general_fact": "주요 근거",
    }.get(section, section.replace("_", " "))


def _clean_entities(value: Any) -> list[str]:
    return _dedupe_strings(
        entity
        for entity in _normalize_string_list(value)
        if _valid_entity(entity)
    )[:8]


def _valid_entity(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) < 2:
        return False
    stopwords = {
        "며", "고", "및", "등", "은", "는", "이", "가", "을", "를", "의", "에", "에서",
        "으로", "로", "까지", "부터", "했다", "말했다", "통해",
    }
    if text in stopwords:
        return False
    if re.fullmatch(r"[가-힣]{1,2}", text) and text not in {"AI", "RX"}:
        return False
    return bool(re.search(r"[A-Za-z0-9가-힣]", text))


def _numbers_from_text(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|건|명|개|년|월|일)?", text or "")


def _compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in (value or []) if isinstance(item, dict)]


class _EvidenceTextRegistry:
    """Deduplicate repeated evidence snippets in the public IntegratedIssue payload."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._id_by_key: dict[str, str] = {}

    def add(self, text: Any, *, source_ids: list[int] | None = None) -> str:
        cleaned = _reference_text(text)
        if not cleaned:
            return ""
        key = _compact_text(cleaned)
        ref_id = self._id_by_key.get(key)
        if ref_id:
            self._merge_source_ids(ref_id, source_ids)
            return ref_id
        ref_id = f"ev{len(self._items) + 1}"
        item: dict[str, Any] = {"id": ref_id, "text": cleaned}
        ids = _normalize_int_list(source_ids)
        if ids:
            item["source_ids"] = ids
        self._items.append(item)
        self._id_by_key[key] = ref_id
        return ref_id

    def items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    def _merge_source_ids(self, ref_id: str, source_ids: list[int] | None) -> None:
        ids = _normalize_int_list(source_ids)
        if not ids:
            return
        for item in self._items:
            if item.get("id") != ref_id:
                continue
            merged = _normalize_int_list([*item.get("source_ids", []), *ids])
            if merged:
                item["source_ids"] = merged
            return


def _reference_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if text.startswith(("input_bundle.", "raw_articles.", "sources.")):
        return ""
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        return ""
    return text


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


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, list | tuple | set):
        values = list(value)
    else:
        values = [value]
    return _dedupe_ints(values)


def _dedupe_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


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
