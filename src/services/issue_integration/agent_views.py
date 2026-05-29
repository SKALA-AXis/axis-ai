"""Purpose-built IntegratedIssue input views for downstream agents.

IntegratedIssue is the canonical storage/debug payload. Downstream agents should
not receive that whole object by default because it contains overlapping views of
the same source text. This module builds small, role-specific projections.
"""

from __future__ import annotations

from typing import Any

from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy

_VIEW_VERSION = "issue_agent_input_view_v1"


def analysis_agent_issue_input(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Input contract for AnalysisAgent.

    The analysis agent needs enough material to interpret meaning: subject,
    event, content digest, structured facts, source refs, and quality gate. It
    does not need full source_map, full content_digest storage projection, or
    duplicated section body arrays.
    """

    content = _dict(integrated_issue.get("content_digest"))
    frame = _dict(integrated_issue.get("issue_frame"))
    brief = _dict(integrated_issue.get("issue_brief"))
    return {
        "view_version": _VIEW_VERSION,
        "view": "analysis_agent",
        "gate": _gate(integrated_issue, frame),
        "identity": _identity(integrated_issue),
        "subject": _subject(integrated_issue, frame),
        "content": {
            "summary": _first_non_empty(
                content.get("summary"),
                integrated_issue.get("integrated_text"),
                integrated_issue.get("one_line_summary"),
                brief.get("one_line_summary"),
                brief.get("headline"),
            ),
            "detailed_explanation": content.get("detailed_explanation", ""),
            "sections": _sections(content, policy=policy),
            "key_points": _key_points(content, policy=policy),
            "supporting_extracts": _body_extracts(content, policy=policy),
        },
        "structured_frame": _structured_frame(frame, policy=policy),
        "evidence": _evidence(integrated_issue, policy=policy),
        "source_refs": _source_refs(integrated_issue, policy=policy),
    }


def implication_agent_issue_input(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Input contract for ImplicationAgent.

    Implication generation should be grounded mostly in fact_basis and prior
    AnalysisResult. The view intentionally keeps content body excerpts out unless
    they are already selected as facts.
    """

    content = _dict(integrated_issue.get("content_digest"))
    frame = _dict(integrated_issue.get("issue_frame"))
    brief = _dict(integrated_issue.get("issue_brief"))
    return {
        "view_version": _VIEW_VERSION,
        "view": "implication_agent",
        "gate": _gate(integrated_issue, frame),
        "subject": _subject(integrated_issue, frame),
        "issue_summary": {
            "main_issue": brief.get("headline", integrated_issue.get("main_issue", "")),
            "integrated_text": brief.get(
                "one_line_summary",
                integrated_issue.get("integrated_text", ""),
            ),
            "content_summary": content.get("summary", ""),
            "event": _event(frame),
            "topics": _topics(frame, policy=policy),
        },
        "grounding": _evidence(integrated_issue, policy=policy),
        "source_refs": _source_refs(integrated_issue, policy=policy),
    }


def card_news_issue_input(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Input contract for CardNewsAgent or a future card-writing LLM."""

    content = _dict(integrated_issue.get("content_digest"))
    frame = _dict(integrated_issue.get("issue_frame"))
    brief = _dict(integrated_issue.get("issue_brief"))
    return {
        "view_version": _VIEW_VERSION,
        "view": "card_news_agent",
        "gate": _gate(integrated_issue, frame),
        "card_seed": {
            "headline": _first_non_empty(
                brief.get("headline"),
                integrated_issue.get("headline"),
                integrated_issue.get("main_issue"),
                content.get("summary"),
            ),
            "one_line_summary": _first_non_empty(
                brief.get("one_line_summary"),
                integrated_issue.get("one_line_summary"),
                content.get("summary"),
            ),
            "sections": _sections(content, policy=policy),
            "key_points": _key_points(content, policy=policy),
            "event": _event(frame),
            "subject": _subject(integrated_issue, frame),
        },
        "evidence": {
            "fact_basis": _limit(
                _evidence(integrated_issue, policy=policy)["fact_basis"],
                policy.agent_view_fact_limit,
            ),
            "source_refs": _source_refs(integrated_issue, policy=policy),
        },
    }


def mixer_agent_issue_input(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Small issue view for comparing multiple cards/issues."""

    content = _dict(integrated_issue.get("content_digest"))
    frame = _dict(integrated_issue.get("issue_frame"))
    brief = _dict(integrated_issue.get("issue_brief"))
    evidence = _evidence(integrated_issue, policy=policy)
    return {
        "view_version": _VIEW_VERSION,
        "view": "mixer_agent",
        "gate": _gate(integrated_issue, frame),
        "subject": _subject(integrated_issue, frame),
        "main_issue": brief.get("headline")
        or integrated_issue.get("main_issue")
        or integrated_issue.get("headline"),
        "content_summary": content.get("summary", ""),
        "event": _event(frame),
        "topics": _topics(frame, policy=policy),
        "key_numbers": evidence["key_numbers"],
        "business_signals": evidence["business_signals"],
        "quality": _quality(integrated_issue, frame),
    }


def _gate(integrated_issue: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    brief = _dict(integrated_issue.get("issue_brief"))
    quality = _quality(integrated_issue, frame)
    is_valid = bool(brief.get("is_valid", integrated_issue.get("is_valid_summary", True)))
    return {
        "is_valid_summary": is_valid,
        "can_analyze": bool(is_valid and not _has_blocking_quality_flag(quality)),
        "reason": brief.get("reason", integrated_issue.get("reason", "")),
        "confidence": brief.get("confidence", integrated_issue.get("confidence", 0.0)),
        "quality": quality,
    }


def _identity(integrated_issue: dict[str, Any]) -> dict[str, Any]:
    brief = _dict(integrated_issue.get("issue_brief"))
    metadata = _dict(integrated_issue.get("metadata"))
    scope = _dict(brief.get("analysis_scope"))
    return {
        "cluster_id": metadata.get("cluster_id", integrated_issue.get("cluster_id")),
        "bundle_id": metadata.get("bundle_id", integrated_issue.get("bundle_id")),
        "representative_id": metadata.get(
            "representative_id", integrated_issue.get("representative_id")
        ),
        "source_family": brief.get("source_family", integrated_issue.get("source_family")),
        "scope_type": brief.get("scope_type", integrated_issue.get("scope_type")),
        "analyzed_article_ids": scope.get("analyzed_source_ids")
        or scope.get("analyzed_raw_article_ids")
        or integrated_issue.get("analyzed_article_ids", []),
        "source_ids": [
            source.get("id")
            for source in _list(integrated_issue.get("sources"))
            if isinstance(source, dict) and source.get("id")
        ],
    }


def _subject(integrated_issue: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    brief = _dict(integrated_issue.get("issue_brief"))
    companies = _dict(frame.get("companies"))
    return {
        "main_company": brief.get("main_company", integrated_issue.get("main_company", "")),
        "mentioned_peer_companies": brief.get(
            "mentioned_peer_companies",
            integrated_issue.get("mentioned_peer_companies", []),
        ),
        "companies": {
            "primary": _company(_dict(companies.get("primary"))),
            "mentioned": [_company(_dict(item)) for item in _list(companies.get("mentioned"))],
        },
        "sectors": _sectors(frame, integrated_issue),
        "document_subject": brief.get("headline", integrated_issue.get("document_subject", "")),
        "main_issue": brief.get("headline", integrated_issue.get("main_issue", "")),
    }


def _structured_frame(frame: dict[str, Any], *, policy: IntegrationPolicy) -> dict[str, Any]:
    return {
        "event": _event(frame),
        "topics": _topics(frame, policy=policy),
        "entities": _entities(frame, policy=policy),
        "key_numbers": _key_numbers(frame, policy=policy),
        "timeline": _timeline(frame, policy=policy),
        "relations": _relations(frame, policy=policy),
    }


def _evidence(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy,
) -> dict[str, Any]:
    evidence = _dict(integrated_issue.get("evidence"))
    reference_text_by_id = _evidence_reference_text_by_id(evidence)
    evidence_facts = [
        fact
        for section in _list(evidence.get("by_section"))
        if isinstance(section, dict)
        for fact in _list(section.get("facts"))
        if isinstance(fact, dict)
    ]
    return {
        "fact_basis": _limit(
            _compact_fact_basis(
                evidence_facts or _list(integrated_issue.get("fact_basis")),
                reference_text_by_id=reference_text_by_id,
            ),
            policy.agent_view_fact_limit,
        ),
        "consolidated_facts": _limit(
            evidence_facts or _list(integrated_issue.get("consolidated_facts")),
            policy.agent_view_fact_limit,
        ),
        "key_numbers": _limit(
            _list(integrated_issue.get("key_numbers")),
            policy.agent_view_fact_limit,
        ),
        "business_signals": _limit(
            _list(integrated_issue.get("business_signals")),
            policy.agent_view_signal_limit,
        ),
        "missing_or_uncertain_points": integrated_issue.get(
            "missing_or_uncertain_points",
            evidence.get("uncertain_points", []),
        ),
    }


def _sections(content: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section in _limit(_list(content.get("sections")), policy.agent_view_section_limit):
        if not isinstance(section, dict):
            continue
        sections.append(
            {
                "id": section.get("id"),
                "title": section.get("title"),
                "summary": section.get("summary", ""),
                "keywords": section.get("keywords", []),
                "raw_article_ids": section.get("raw_article_ids", []),
                "source_indexes": section.get("source_indexes", []),
            }
        )
    return sections


def _key_points(content: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for point in _limit(_list(content.get("key_points")), policy.agent_view_key_point_limit):
        if not isinstance(point, dict):
            continue
        points.append(
            {
                "id": point.get("id"),
                "point": point.get("point", ""),
                "section": point.get("section", ""),
                "raw_article_ids": point.get("raw_article_ids", []),
                "source_index": point.get("source_index"),
                "numbers_and_dates": point.get("numbers_and_dates", []),
            }
        )
    return points


def _body_extracts(content: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    extracts: list[dict[str, Any]] = []
    for extract in _limit(_list(content.get("body_extracts")), policy.agent_view_extract_limit):
        if not isinstance(extract, dict):
            continue
        extracts.append(
            {
                "id": extract.get("id"),
                "text": extract.get("text", ""),
                "section": extract.get("section", ""),
                "raw_article_id": extract.get("raw_article_id"),
                "source_index": extract.get("source_index"),
                "numbers_and_dates": extract.get("numbers_and_dates", []),
            }
        )
    return extracts


def _source_refs(
    integrated_issue: dict[str, Any],
    *,
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    source_map = _dict(integrated_issue.get("source_map"))
    content = _dict(integrated_issue.get("content_digest"))
    sources = (
        _list(integrated_issue.get("sources"))
        or _list(integrated_issue.get("cluster_sources"))
        or _list(integrated_issue.get("source_references"))
        or _list(source_map.get("sources"))
        or _list(content.get("sources"))
    )
    refs: list[dict[str, Any]] = []
    for source in _limit(sources, policy.agent_view_source_limit):
        if not isinstance(source, dict):
            continue
        refs.append(
            {
                "id": source.get("id") or source.get("raw_article_id") or source.get("article_id"),
                "title": source.get("title"),
                "source_name": source.get("source_name") or source.get("source"),
                "source_type": source.get("source_type"),
                "publisher": source.get("publisher"),
                "published_at": source.get("published_at"),
                "url": source.get("url"),
                "relevance_label": source.get("relevance_label"),
                "relevance_score": source.get("relevance_score"),
            }
        )
    return refs


def _quality(integrated_issue: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    quality = _dict(integrated_issue.get("quality"))
    frame_quality = _dict(frame.get("quality"))
    return {
        "review_flags": quality.get("review_flags", []),
        "source_count": quality.get("source_count"),
        "eligible_source_count": quality.get("eligible_source_count"),
        "source_coverage": quality.get("source_coverage"),
        "selected_fact_count": quality.get("selected_fact_count"),
        "evidence_count": quality.get("evidence_count"),
        "frame_completeness": frame_quality.get("frame_completeness"),
        "frame_warnings": frame_quality.get("warnings", []),
    }


def _has_blocking_quality_flag(quality: dict[str, Any]) -> bool:
    flags = set(str(flag) for flag in quality.get("review_flags") or [])
    frame_warnings = set(str(flag) for flag in quality.get("frame_warnings") or [])
    blocking = {
        "no_analysis_eligible_source_rows",
        "contains_irrelevant_source_rows",
        "no_selected_facts",
    }
    return bool(flags & blocking or frame_warnings & {"no_analysis_eligible_source_rows"})


def _event(frame: dict[str, Any]) -> dict[str, Any]:
    event = _dict(frame.get("event"))
    return {
        "type": event.get("type"),
        "confidence": event.get("confidence"),
        "trigger_terms": event.get("trigger_terms", []),
        "source": event.get("source"),
        "raw_article_ids": event.get("raw_article_ids", []),
        "source_indexes": event.get("source_indexes", []),
    }


def _topics(frame: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for topic in _limit(_list(frame.get("topics")), policy.agent_view_section_limit):
        if not isinstance(topic, dict):
            continue
        topics.append(
            {
                "id": topic.get("id"),
                "label": topic.get("label"),
                "keywords": topic.get("keywords", []),
                "confidence": topic.get("confidence"),
            }
        )
    return topics


def _entities(frame: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for entity in _limit(_list(frame.get("entities")), policy.agent_view_section_limit):
        if not isinstance(entity, dict):
            continue
        entities.append(
            {
                "name": entity.get("name"),
                "type": entity.get("type"),
                "role": entity.get("role"),
                "normalized_id": entity.get("normalized_id"),
                "confidence": entity.get("confidence"),
            }
        )
    return entities


def _key_numbers(frame: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    for number in _limit(_list(frame.get("key_numbers")), policy.agent_view_fact_limit):
        if not isinstance(number, dict):
            continue
        numbers.append(
            {
                "text": number.get("text"),
                "type": number.get("type"),
                "metric_name": number.get("metric_name"),
                "metric_label": number.get("metric_label"),
                "period": number.get("period"),
                "fact_id": number.get("fact_id"),
                "raw_article_ids": number.get("raw_article_ids", []),
                "source_indexes": number.get("source_indexes", []),
            }
        )
    return numbers


def _timeline(frame: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for item in _limit(_list(frame.get("timeline")), policy.agent_view_section_limit):
        if not isinstance(item, dict):
            continue
        timeline.append(
            {
                "date": item.get("date"),
                "event": item.get("event"),
                "context": item.get("context"),
                "fact_id": item.get("fact_id"),
                "raw_article_ids": item.get("raw_article_ids", []),
            }
        )
    return timeline


def _relations(frame: dict[str, Any], *, policy: IntegrationPolicy) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for relation in _limit(_list(frame.get("relations")), policy.agent_view_section_limit):
        if not isinstance(relation, dict):
            continue
        relations.append(
            {
                "subject": relation.get("subject"),
                "subject_id": relation.get("subject_id"),
                "relation": relation.get("relation"),
                "object": relation.get("object"),
                "object_type": relation.get("object_type"),
                "fact_ids": relation.get("fact_ids", []),
                "confidence": relation.get("confidence"),
            }
        )
    return relations


def _sectors(frame: dict[str, Any], integrated_issue: dict[str, Any]) -> list[dict[str, Any]]:
    sectors = _list(frame.get("sectors"))
    if sectors:
        return [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "matched_keywords": item.get("matched_keywords", []),
            }
            for item in sectors
            if isinstance(item, dict)
        ]
    return [{"id": sector, "label": sector} for sector in integrated_issue.get("sectors", [])]


def _company(company: dict[str, Any]) -> dict[str, Any]:
    if not company:
        return {}
    return {
        "id": company.get("id"),
        "label": company.get("label"),
        "confidence": company.get("confidence"),
        "source_indexes": company.get("source_indexes", []),
    }


def _compact_fact_basis(
    items: list[Any],
    *,
    reference_text_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    reference_text_by_id = reference_text_by_id or {}
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_text = item.get("evidence_text") or reference_text_by_id.get(
            str(item.get("evidence_ref_id") or "")
        )
        out.append(
            {
                "fact_id": item.get("fact_id"),
                "fact": item.get("fact"),
                "evidence_text": evidence_text,
                "evidence_type": item.get("evidence_type") or item.get("fact_type"),
                "raw_article_ids": item.get("raw_article_ids") or item.get("source_ids", []),
                "source_indexes": item.get("source_indexes", []),
            }
        )
    return out


def _evidence_reference_text_by_id(evidence: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for item in _list(evidence.get("references")):
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("id") or "")
        text = str(item.get("text") or "")
        if ref_id and text:
            refs[ref_id] = text
    return refs


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _limit(values: list[Any], limit: int) -> list[Any]:
    return values[: max(0, int(limit))]


__all__ = [
    "analysis_agent_issue_input",
    "card_news_issue_input",
    "implication_agent_issue_input",
    "mixer_agent_issue_input",
]
