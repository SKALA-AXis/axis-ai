"""Rule-based issue frame builder.

issue_frame is not a card-news payload. It is a normalized, evidence-linked
analysis material layer that downstream agents can use without rereading raw rows.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.config.companies import COMPANIES, company_name_ko
from src.config.event_types import (
    EVENT_KEYWORDS_REQUIRING_CONTEXT,
    EVENT_TYPE_CONTEXT_KEYWORDS,
    EVENT_TYPE_KEYWORDS,
    EVENT_TYPE_TIE_BREAK_PRIORITY,
    EVENT_TYPES,
)
from src.config.global_companies import GLOBAL_COMPANIES, global_company_name_ko
from src.config.sectors import SECTOR_IDS, match_sector_details, sector_name_ko
from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy
from src.services.issue_integration.source_profile import SourceProfile

_NUMBER_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|USD|KRW|usd|krw|건|명|개|분기|년|월|일)?"
)
_DATE_YMD_PATTERN = re.compile(r"(20\d{2})[년.\-/]\s*(\d{1,2})[월.\-/]\s*(\d{1,2})일?")
_ISO_DATE_PATTERN = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_PERSON_ROLE_PATTERN = re.compile(r"([가-힣]{2,4})\s*(회장|대표이사|대표|관장|대법관|사장|부회장)")
_INSTITUTION_PATTERN = re.compile(
    r"([A-Za-z가-힣0-9㈜().&·\s]{0,18}(?:대법원|법원|위원회|정부|재단|그룹|센터|연구원|거래소|금융감독원))"
)
_INVALID_PERSON_NAMES = {
    "그룹",
    "나비",
    "선대",
    "원고",
    "피고",
    "대법원",
    "서울고법",
    "재판부",
}
_INVALID_ENTITY_FRAGMENTS = (
    "했다",
    "됐다",
    "된다",
    "밝혔다",
    "설명했다",
    "판단했다",
    "못박았다",
    "취지다",
)


def build_issue_frame(
    *,
    input_bundle: AnalysisInputBundle,
    source_profile: SourceProfile,
    facts: list[dict[str, Any]],
    content_digest: dict[str, Any],
    source_map: dict[str, Any],
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Build a deterministic issue frame from structured and text signals."""

    fact_basis = facts if facts is not None else input_bundle.facts or []
    text_by_article = _text_by_article(input_bundle.items)
    eligible_raw_ids = {
        _safe_int(raw_id) for raw_id in source_map.get("eligible_raw_article_ids") or []
    }
    has_source_rows = bool(source_map.get("sources"))
    allow_content_mining = bool(eligible_raw_ids) or not has_source_rows
    content_text_by_article = (
        {raw_id: text for raw_id, text in text_by_article.items() if raw_id in eligible_raw_ids}
        if eligible_raw_ids
        else (text_by_article if allow_content_mining else {})
    )
    source_index_by_raw_id = {
        _safe_int(raw_id): _safe_int(index)
        for raw_id, index in (source_map.get("raw_article_id_to_source_index") or {}).items()
    }
    companies = _company_candidates(
        input_bundle=input_bundle,
        text_by_article=text_by_article,
        content_text_by_article=content_text_by_article,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    sectors = _sector_candidates(
        input_bundle=input_bundle,
        text_by_article=text_by_article,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    event = _event_frame(
        input_bundle=input_bundle,
        facts=fact_basis,
        text_by_article=text_by_article,
        content_text_by_article=content_text_by_article,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    topics = _topics(
        input_bundle=input_bundle,
        facts=fact_basis,
        content_digest=content_digest,
        sectors=sectors,
        event=event,
        allow_content_mining=allow_content_mining,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    key_numbers = _key_numbers(
        facts=fact_basis,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    timeline = _timeline(
        input_bundle=input_bundle,
        facts=fact_basis,
        source_map=source_map,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    entities = _entities(
        companies=companies,
        input_bundle=input_bundle,
        text_by_article=content_text_by_article,
        source_index_by_raw_id=source_index_by_raw_id,
        policy=policy,
    )
    relations = (
        _relations(
            companies=companies,
            sectors=sectors,
            event=event,
            topics=topics,
            facts=fact_basis,
            source_index_by_raw_id=source_index_by_raw_id,
            policy=policy,
        )
        if allow_content_mining
        else []
    )
    return {
        "frame_version": "issue_frame_v1",
        "source_scope": {
            "bundle_id": input_bundle.bundle_id,
            "cluster_id": input_bundle.cluster_id,
            "source_type": input_bundle.source_type,
            "source_family": source_profile.source_family,
            "scope_type": source_profile.scope_type,
            "raw_article_ids": source_map.get("raw_article_ids", []),
            "basis_raw_article_ids": source_map.get("basis_raw_article_ids", []),
            "eligible_raw_article_ids": source_map.get("eligible_raw_article_ids", []),
            "source_count": source_map.get("source_count", 0),
        },
        "companies": _company_frame(companies),
        "sectors": sectors,
        "event": event,
        "topics": topics,
        "entities": entities,
        "key_numbers": key_numbers,
        "timeline": timeline,
        "relations": relations,
        "unresolved_candidates": _unresolved_candidates(
            companies=companies,
            sectors=sectors,
            event=event,
            topics=topics,
            source_profile=source_profile,
        ),
        "quality": _quality(
            companies=companies,
            sectors=sectors,
            event=event,
            topics=topics,
            entities=entities,
            source_profile=source_profile,
            has_analysis_eligible_source=allow_content_mining,
        ),
    }


def _company_candidates(
    *,
    input_bundle: AnalysisInputBundle,
    text_by_article: dict[int, str],
    content_text_by_article: dict[int, str],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for company_id in input_bundle.companies or []:
        normalized = _normalize_company_id(company_id)
        if normalized:
            candidates.append(
                _candidate(
                    key=normalized,
                    label=_company_label(normalized),
                    source="input_bundle",
                    confidence=_confidence(policy, "input_bundle"),
                    raw_article_ids=_all_raw_ids(text_by_article),
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text="input_bundle.companies",
                    source_fields=["input_bundle.companies"],
                )
            )
    for source in input_bundle.sources or []:
        raw_id = _source_id(source)
        source_companies = [
            *_string_list(source.get("companies")),
            *_string_list(source.get("matched_companies")),
        ]
        for value in source_companies:
            normalized = _normalize_company_id(value)
            if normalized:
                candidates.append(
                    _candidate(
                        key=normalized,
                        label=_company_label(normalized),
                        source="source_metadata",
                        confidence=_confidence(policy, "source_metadata"),
                        raw_article_ids=[raw_id],
                        source_index_by_raw_id=source_index_by_raw_id,
                        evidence_text=str(value),
                        source_fields=["sources.companies", "sources.matched_companies"],
                    )
                )
    for item in input_bundle.items or []:
        raw_id = _raw_id(item)
        item_companies = [
            *_string_list(item.get("company")),
            *_string_list(item.get("matched_companies")),
        ]
        for value in item_companies:
            normalized = _normalize_company_id(value)
            if normalized:
                candidates.append(
                    _candidate(
                        key=normalized,
                        label=_company_label(normalized),
                        source="article_metadata",
                        confidence=_confidence(policy, "article_metadata"),
                        raw_article_ids=[raw_id],
                        source_index_by_raw_id=source_index_by_raw_id,
                        evidence_text=str(value),
                        source_fields=["raw_articles.company", "raw_articles.matched_companies"],
                    )
                )
        title = _item_title(input_bundle.items, raw_id) if raw_id in content_text_by_article else ""
        content = content_text_by_article.get(raw_id, "")
        for normalized, alias in _company_alias_hits(title):
            candidates.append(
                _candidate(
                    key=normalized,
                    label=_company_label(normalized),
                    source="title_keyword",
                    confidence=_confidence(policy, "title_keyword"),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=title,
                    source_fields=["raw_articles.title"],
                    trigger_terms=[alias],
                )
            )
        for normalized, alias in _company_alias_hits(content):
            candidates.append(
                _candidate(
                    key=normalized,
                    label=_company_label(normalized),
                    source="content_keyword",
                    confidence=_confidence(policy, "content_keyword"),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=_sentence_containing(content, alias),
                    source_fields=["raw_articles.content"],
                    trigger_terms=[alias],
                )
            )
    return _merge_candidates(candidates, limit=12)


def _sector_candidates(
    *,
    input_bundle: AnalysisInputBundle,
    text_by_article: dict[int, str],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sector_id in input_bundle.sectors or []:
        normalized = _normalize_sector_id(sector_id)
        candidates.append(
            _candidate(
                key=normalized,
                label=sector_name_ko(normalized),
                source="input_bundle",
                confidence=_sector_confidence(policy, "input_bundle", normalized),
                raw_article_ids=_all_raw_ids(text_by_article),
                source_index_by_raw_id=source_index_by_raw_id,
                evidence_text="input_bundle.sectors",
                source_fields=["input_bundle.sectors"],
            )
        )
    for source in input_bundle.sources or []:
        raw_id = _source_id(source)
        for sector_id in _string_list(source.get("matched_sectors")):
            normalized = _normalize_sector_id(sector_id)
            candidates.append(
                _candidate(
                    key=normalized,
                    label=sector_name_ko(normalized),
                    source="source_metadata",
                    confidence=_sector_confidence(policy, "source_metadata", normalized),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=str(sector_id),
                    source_fields=["sources.matched_sectors"],
                )
            )
    for item in input_bundle.items or []:
        raw_id = _raw_id(item)
        item_sectors = [
            *_string_list(item.get("sector")),
            *_string_list(item.get("matched_sectors")),
        ]
        for sector_id in item_sectors:
            normalized = _normalize_sector_id(sector_id)
            candidates.append(
                _candidate(
                    key=normalized,
                    label=sector_name_ko(normalized),
                    source="article_metadata",
                    confidence=_sector_confidence(policy, "article_metadata", normalized),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=str(sector_id),
                    source_fields=["raw_articles.sector", "raw_articles.matched_sectors"],
                )
            )
        title = str(item.get("title") or "")
        corpus = text_by_article.get(raw_id, "")
        for detail in match_sector_details(title):
            candidates.append(
                _candidate(
                    key=detail["sector_id"],
                    label=detail["sector_name_ko"],
                    source="title_keyword",
                    confidence=_confidence(policy, "title_keyword"),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=title,
                    source_fields=["raw_articles.title"],
                    trigger_terms=[detail["keyword"]],
                )
            )
        for detail in match_sector_details(corpus):
            candidates.append(
                _candidate(
                    key=detail["sector_id"],
                    label=detail["sector_name_ko"],
                    source="content_keyword",
                    confidence=_confidence(policy, "content_keyword"),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=_sentence_containing(corpus, detail["keyword"]),
                    source_fields=["raw_articles.content", "parser_result"],
                    trigger_terms=[detail["keyword"]],
                )
            )
    merged = _merge_candidates(candidates, limit=8)
    return merged or [
        {
            "id": "other",
            "label": sector_name_ko("other"),
            "confidence": _sector_confidence(policy, "fallback", "other"),
            "raw_article_ids": [],
            "source_indexes": [],
            "evidence_text": "",
            "source_fields": [],
            "matched_keywords": [],
            "sources": ["fallback"],
        }
    ]


def _event_frame(
    *,
    input_bundle: AnalysisInputBundle,
    facts: list[dict[str, Any]],
    text_by_article: dict[int, str],
    content_text_by_article: dict[int, str],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> dict[str, Any]:
    explicit = str(input_bundle.event_type or "").strip()
    if explicit in EVENT_TYPES:
        raw_ids = _all_raw_ids(text_by_article)
        trigger_terms = _event_trigger_terms(explicit, " ".join(text_by_article.values()))
        return {
            "type": explicit,
            "confidence": _confidence(policy, "input_bundle"),
            "trigger_terms": trigger_terms,
            "raw_article_ids": raw_ids,
            "source_indexes": _source_indexes(raw_ids, source_index_by_raw_id),
            "evidence_text": "input_bundle.event_type",
            "source": "input_bundle",
        }

    matches: list[dict[str, Any]] = []
    for raw_id, text in content_text_by_article.items():
        title = _item_title(input_bundle.items, raw_id)
        matches.extend(
            _event_matches(
                text=title,
                raw_id=raw_id,
                source="title_keyword",
                confidence=_confidence(policy, "title_keyword"),
                source_index_by_raw_id=source_index_by_raw_id,
            )
        )
        matches.extend(
            _event_matches(
                text=text,
                raw_id=raw_id,
                source="content_keyword",
                confidence=_confidence(policy, "content_keyword"),
                source_index_by_raw_id=source_index_by_raw_id,
            )
        )
    for fact in facts:
        raw_id = _fact_raw_id(fact)
        matches.extend(
            _event_matches(
                text=f"{fact.get('fact') or ''} {fact.get('evidence_text') or ''}",
                raw_id=raw_id,
                source="fact_keyword",
                confidence=_confidence(policy, "fact_keyword"),
                source_index_by_raw_id=source_index_by_raw_id,
            )
        )
    if matches:
        selected = sorted(
            matches,
            key=lambda item: (
                item["match_count"],
                item["confidence"],
                -EVENT_TYPE_TIE_BREAK_PRIORITY.get(item["type"], 99),
            ),
            reverse=True,
        )[0]
        return {
            "type": selected["type"],
            "confidence": selected["confidence"],
            "trigger_terms": selected["trigger_terms"],
            "raw_article_ids": selected["raw_article_ids"],
            "source_indexes": selected["source_indexes"],
            "evidence_text": selected["evidence_text"],
            "source": selected["source"],
        }

    raw_ids = _all_raw_ids(text_by_article)
    return {
        "type": "company",
        "confidence": _confidence(policy, "fallback"),
        "trigger_terms": [],
        "raw_article_ids": raw_ids,
        "source_indexes": _source_indexes(raw_ids, source_index_by_raw_id),
        "evidence_text": "",
        "source": "fallback",
    }


def _topics(
    *,
    input_bundle: AnalysisInputBundle,
    facts: list[dict[str, Any]],
    content_digest: dict[str, Any],
    sectors: list[dict[str, Any]],
    event: dict[str, Any],
    allow_content_mining: bool,
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for sector in sectors:
        keywords = list(sector.get("matched_keywords") or [])
        if sector.get("id") != "other" or keywords:
            topics.append(
                _topic(
                    label=str(sector.get("label") or sector.get("id") or ""),
                    source="sector",
                    keywords=keywords,
                    raw_article_ids=sector.get("raw_article_ids") or [],
                    source_index_by_raw_id=source_index_by_raw_id,
                    confidence=float(sector.get("confidence") or 0.0),
                )
            )
    if event.get("trigger_terms"):
        topics.append(
            _topic(
                label=str(event.get("type") or ""),
                source="event",
                keywords=list(event.get("trigger_terms") or []),
                raw_article_ids=event.get("raw_article_ids") or [],
                source_index_by_raw_id=source_index_by_raw_id,
                confidence=float(event.get("confidence") or 0.0),
            )
        )
    for item in input_bundle.items or []:
        raw_id = _raw_id(item)
        parsed = _parser_result(item)
        for signal in _dict_list(parsed.get("topic_signals")):
            label = str(
                signal.get("topic")
                or signal.get("summary")
                or signal.get("signal")
                or signal.get("text")
                or ""
            ).strip()
            if not label:
                continue
            topics.append(
                _topic(
                    label=label,
                    source="parser_result",
                    keywords=[],
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    confidence=_confidence(policy, "parser_result"),
                )
            )
    for fact in facts:
        label = _first_non_empty(fact.get("business_area"), fact.get("signal_type"))
        if label:
            raw_id = _fact_raw_id(fact)
            topics.append(
                _topic(
                    label=label,
                    source="business_signal",
                    keywords=[],
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    confidence=_confidence(policy, "fact_keyword"),
                )
            )
    if not allow_content_mining:
        return _merge_topics(topics)[: policy.frame_topic_limit]
    for section in content_digest.get("sections") or []:
        if not isinstance(section, dict):
            continue
        label = str(section.get("title") or section.get("section") or "").strip()
        if not label:
            continue
        raw_ids = [_safe_int(raw_id) for raw_id in section.get("raw_article_ids") or []]
        topics.append(
            _topic(
                label=label,
                source="content_section",
                keywords=list(section.get("keywords") or []),
                raw_article_ids=raw_ids,
                source_index_by_raw_id=source_index_by_raw_id,
                confidence=0.68,
            )
        )
    return _merge_topics(topics)[: policy.frame_topic_limit]


def _key_numbers(
    *,
    facts: list[dict[str, Any]],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    numbers: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for fact in facts:
        raw_id = _fact_raw_id(fact)
        context = _first_non_empty(fact.get("evidence_text"), fact.get("fact"))
        for text in fact.get("numbers_and_dates") or _NUMBER_PATTERN.findall(context):
            key = (str(text), raw_id)
            if key in seen:
                continue
            seen.add(key)
            numbers.append(
                {
                    "text": str(text),
                    "type": _number_type(str(text)),
                    "context": context,
                    "metric_name": fact.get("metric_name"),
                    "metric_label": fact.get("metric_label"),
                    "period": fact.get("period"),
                    "fact_id": fact.get("fact_id"),
                    "raw_article_ids": [raw_id] if raw_id > 0 else [],
                    "source_indexes": _source_indexes([raw_id], source_index_by_raw_id),
                }
            )
    return numbers[: policy.frame_number_limit]


def _timeline(
    *,
    input_bundle: AnalysisInputBundle,
    facts: list[dict[str, Any]],
    source_map: dict[str, Any],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for source in source_map.get("sources") or []:
        if not isinstance(source, dict):
            continue
        date = _normalize_date(source.get("published_at"))
        raw_id = _safe_int(source.get("raw_article_id") or source.get("id"))
        if not date or (date, "source_published", raw_id) in seen:
            continue
        seen.add((date, "source_published", raw_id))
        entries.append(
            {
                "date": date,
                "event": "source_published",
                "context": source.get("title") or "",
                "raw_article_ids": [raw_id] if raw_id > 0 else [],
                "source_indexes": (
                    [source.get("source_index")] if source.get("source_index") else []
                ),
            }
        )
    for fact in facts:
        raw_id = _fact_raw_id(fact)
        context = _first_non_empty(fact.get("evidence_text"), fact.get("fact"))
        for date in _dates_in_text(context):
            key = (date, context[:80], raw_id)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "date": date,
                    "event": "reported_date",
                    "context": context,
                    "fact_id": fact.get("fact_id"),
                    "raw_article_ids": [raw_id] if raw_id > 0 else [],
                    "source_indexes": _source_indexes([raw_id], source_index_by_raw_id),
                }
            )
    return entries[: policy.frame_timeline_limit]


def _entities(
    *,
    companies: list[dict[str, Any]],
    input_bundle: AnalysisInputBundle,
    text_by_article: dict[int, str],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for index, company in enumerate(companies):
        entity_type = "global_company" if str(company.get("id")) in GLOBAL_COMPANIES else "company"
        entities.append(
            {
                "name": company.get("label"),
                "type": entity_type,
                "role": "primary_company" if index == 0 else "mentioned_company",
                "normalized_id": company.get("id"),
                "raw_article_ids": company.get("raw_article_ids") or [],
                "source_indexes": company.get("source_indexes") or [],
                "evidence_text": company.get("evidence_text") or "",
                "confidence": company.get("confidence"),
            }
        )
    for item in input_bundle.items or []:
        raw_id = _raw_id(item)
        parsed = _parser_result(item)
        for entity in _dict_list(parsed.get("entities")):
            name = str(entity.get("name") or entity.get("text") or "").strip()
            if not name:
                continue
            entities.append(
                _entity(
                    name=name,
                    entity_type=str(entity.get("type") or "unknown"),
                    role=str(entity.get("role") or "mentioned"),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=str(entity.get("evidence_text") or ""),
                    confidence=_confidence(policy, "parser_result"),
                )
            )
        text = text_by_article.get(raw_id, "")
        for match in _PERSON_ROLE_PATTERN.finditer(text):
            name = match.group(1)
            if not _is_valid_person_name(name):
                continue
            entities.append(
                _entity(
                    name=name,
                    entity_type="person",
                    role=match.group(2),
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=_sentence_containing(text, match.group(0)),
                    confidence=_confidence(policy, "content_keyword"),
                )
            )
        for match in _INSTITUTION_PATTERN.finditer(text):
            name = " ".join(match.group(1).split())
            if not _is_valid_entity_name(name):
                continue
            entities.append(
                _entity(
                    name=name,
                    entity_type="organization",
                    role="mentioned",
                    raw_article_ids=[raw_id],
                    source_index_by_raw_id=source_index_by_raw_id,
                    evidence_text=_sentence_containing(text, name),
                    confidence=_confidence(policy, "content_keyword"),
                )
            )
    return _merge_entities(entities)[: policy.frame_entity_limit]


def _relations(
    *,
    companies: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
    event: dict[str, Any],
    topics: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    source_index_by_raw_id: dict[int, int],
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    if not companies:
        return []
    primary = companies[0]
    relations: list[dict[str, Any]] = []
    evidence_fact = _first_fact_matching_event(facts, event.get("trigger_terms") or [])
    raw_id = _fact_raw_id(evidence_fact) if evidence_fact else 0
    object_label = _first_non_empty(
        (topics[0] or {}).get("label") if topics else "",
        (sectors[0] or {}).get("label") if sectors else "",
        event.get("type"),
    )
    if event.get("type") != "company" and object_label:
        relations.append(
            {
                "subject": primary.get("label"),
                "subject_id": primary.get("id"),
                "relation": event.get("type"),
                "object": object_label,
                "object_type": "topic",
                "evidence_text": _first_non_empty(
                    evidence_fact.get("evidence_text") if evidence_fact else "",
                    evidence_fact.get("fact") if evidence_fact else "",
                    event.get("evidence_text"),
                ),
                "fact_ids": [evidence_fact.get("fact_id")] if evidence_fact else [],
                "raw_article_ids": [raw_id] if raw_id > 0 else event.get("raw_article_ids", []),
                "source_indexes": _source_indexes(
                    [raw_id] if raw_id > 0 else event.get("raw_article_ids", []),
                    source_index_by_raw_id,
                ),
                "confidence": min(
                    float(primary.get("confidence") or 0.0),
                    float(event.get("confidence") or 0.0),
                ),
            }
        )
    if event.get("type") == "partnership" and len(companies) >= 2:
        relations.append(
            {
                "subject": companies[0].get("label"),
                "subject_id": companies[0].get("id"),
                "relation": "partnership",
                "object": companies[1].get("label"),
                "object_id": companies[1].get("id"),
                "object_type": "company",
                "evidence_text": event.get("evidence_text") or "",
                "fact_ids": [evidence_fact.get("fact_id")] if evidence_fact else [],
                "raw_article_ids": event.get("raw_article_ids", []),
                "source_indexes": event.get("source_indexes", []),
                "confidence": float(event.get("confidence") or 0.0),
            }
        )
    return relations[: policy.frame_relation_limit]


def _quality(
    *,
    companies: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
    event: dict[str, Any],
    topics: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    source_profile: SourceProfile,
    has_analysis_eligible_source: bool,
) -> dict[str, Any]:
    parts = [
        bool(companies) or source_profile.scope_type in {"industry", "market", "mixed"},
        bool(sectors),
        bool(event.get("type")),
        bool(topics),
        bool(entities),
    ]
    score = sum(1 for part in parts if part) / len(parts)
    warnings: list[str] = []
    if not has_analysis_eligible_source:
        warnings.append("no_analysis_eligible_source_rows")
        score = min(score, 0.4)
    if source_profile.scope_type == "peer_company" and not companies:
        warnings.append("missing_primary_company")
    if not topics:
        warnings.append("missing_topics")
    if float(event.get("confidence") or 0.0) < 0.5:
        warnings.append("low_confidence_event")
    completeness = "complete" if score >= 0.8 else "partial" if score >= 0.45 else "thin"
    return {
        "frame_completeness": completeness,
        "completeness_score": round(score, 3),
        "warnings": warnings,
    }


def _unresolved_candidates(
    *,
    companies: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
    event: dict[str, Any],
    topics: list[dict[str, Any]],
    source_profile: SourceProfile,
) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    if source_profile.scope_type == "peer_company" and not companies:
        unresolved.append({"kind": "company", "reason": "peer scope but no company candidate"})
    if not sectors:
        unresolved.append({"kind": "sector", "reason": "no sector candidate"})
    if not event.get("type"):
        unresolved.append({"kind": "event_type", "reason": "no event candidate"})
    if not topics:
        unresolved.append({"kind": "topic", "reason": "no topic candidate"})
    return unresolved


def _candidate(
    *,
    key: str,
    label: str,
    source: str,
    confidence: float,
    raw_article_ids: list[int],
    source_index_by_raw_id: dict[int, int],
    evidence_text: str,
    source_fields: list[str],
    trigger_terms: list[str] | None = None,
) -> dict[str, Any]:
    raw_ids = _dedupe_ints(raw_article_ids)
    return {
        "id": key,
        "label": label,
        "confidence": round(confidence, 3),
        "raw_article_ids": raw_ids,
        "source_indexes": _source_indexes(raw_ids, source_index_by_raw_id),
        "evidence_text": evidence_text,
        "source_fields": source_fields,
        "matched_keywords": list(trigger_terms or []),
        "sources": [source],
    }


def _merge_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate.get("id") or "").strip()
        if not key:
            continue
        if key not in merged:
            merged[key] = {**candidate}
            continue
        existing = merged[key]
        existing["confidence"] = round(
            max(
                float(existing.get("confidence") or 0.0),
                float(candidate.get("confidence") or 0.0),
            ),
            3,
        )
        existing["raw_article_ids"] = _dedupe_ints(
            [*existing.get("raw_article_ids", []), *candidate.get("raw_article_ids", [])]
        )
        existing["source_indexes"] = _dedupe_ints(
            [*existing.get("source_indexes", []), *candidate.get("source_indexes", [])]
        )
        existing["source_fields"] = _dedupe_strings(
            [*existing.get("source_fields", []), *candidate.get("source_fields", [])]
        )
        existing["matched_keywords"] = _dedupe_strings(
            [*existing.get("matched_keywords", []), *candidate.get("matched_keywords", [])]
        )
        existing["sources"] = _dedupe_strings(
            [*existing.get("sources", []), *candidate.get("sources", [])]
        )
        if len(str(candidate.get("evidence_text") or "")) > len(
            str(existing.get("evidence_text") or "")
        ):
            existing["evidence_text"] = candidate.get("evidence_text")
    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            len(item.get("raw_article_ids", [])),
        ),
        reverse=True,
    )[:limit]


def _company_frame(companies: list[dict[str, Any]]) -> dict[str, Any]:
    primary = companies[0] if companies else None
    return {
        "primary": primary,
        "mentioned": companies[1:] if primary else [],
    }


def _topic(
    *,
    label: str,
    source: str,
    keywords: list[str],
    raw_article_ids: list[int],
    source_index_by_raw_id: dict[int, int],
    confidence: float,
) -> dict[str, Any]:
    raw_ids = _dedupe_ints(raw_article_ids)
    return {
        "id": _slug(label),
        "label": label,
        "keywords": _dedupe_strings(keywords),
        "source": source,
        "raw_article_ids": raw_ids,
        "source_indexes": _source_indexes(raw_ids, source_index_by_raw_id),
        "confidence": round(confidence, 3),
    }


def _merge_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for topic in topics:
        label = str(topic.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key not in merged:
            merged[key] = {**topic}
            continue
        existing = merged[key]
        existing["keywords"] = _dedupe_strings(
            [*existing.get("keywords", []), *topic.get("keywords", [])]
        )
        existing["raw_article_ids"] = _dedupe_ints(
            [*existing.get("raw_article_ids", []), *topic.get("raw_article_ids", [])]
        )
        existing["source_indexes"] = _dedupe_ints(
            [*existing.get("source_indexes", []), *topic.get("source_indexes", [])]
        )
        existing["confidence"] = round(
            max(float(existing.get("confidence") or 0.0), float(topic.get("confidence") or 0.0)),
            3,
        )
    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            len(item.get("raw_article_ids", [])),
        ),
        reverse=True,
    )


def _entity(
    *,
    name: str,
    entity_type: str,
    role: str,
    raw_article_ids: list[int],
    source_index_by_raw_id: dict[int, int],
    evidence_text: str,
    confidence: float,
) -> dict[str, Any]:
    raw_ids = _dedupe_ints(raw_article_ids)
    return {
        "name": name,
        "type": entity_type,
        "role": role,
        "raw_article_ids": raw_ids,
        "source_indexes": _source_indexes(raw_ids, source_index_by_raw_id),
        "evidence_text": evidence_text,
        "confidence": round(confidence, 3),
    }


def _merge_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "unknown").strip()
        if not name:
            continue
        key = (name.lower(), entity_type)
        if key not in merged:
            merged[key] = {**entity}
            continue
        existing = merged[key]
        existing["raw_article_ids"] = _dedupe_ints(
            [*existing.get("raw_article_ids", []), *entity.get("raw_article_ids", [])]
        )
        existing["source_indexes"] = _dedupe_ints(
            [*existing.get("source_indexes", []), *entity.get("source_indexes", [])]
        )
        existing["confidence"] = round(
            max(float(existing.get("confidence") or 0.0), float(entity.get("confidence") or 0.0)),
            3,
        )
    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            len(item.get("raw_article_ids", [])),
        ),
        reverse=True,
    )


def _is_valid_person_name(name: str) -> bool:
    text = str(name or "").strip()
    return bool(text) and text not in _INVALID_PERSON_NAMES


def _is_valid_entity_name(name: str) -> bool:
    text = str(name or "").strip()
    if len(text) < 2 or len(text) > 32:
        return False
    if any(mark in text for mark in (".", '"', "“", "”")):
        return False
    if any(fragment in text for fragment in _INVALID_ENTITY_FRAGMENTS):
        return False
    return True


def _event_matches(
    *,
    text: str,
    raw_id: int,
    source: str,
    confidence: float,
    source_index_by_raw_id: dict[int, int],
) -> list[dict[str, Any]]:
    lowered = str(text or "").lower()
    if not lowered:
        return []
    matches: list[dict[str, Any]] = []
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        trigger_terms = [
            keyword
            for keyword in keywords
            if keyword.lower() in lowered
            and _event_keyword_has_context(
                event_type=event_type,
                keyword=keyword,
                lowered_text=lowered,
            )
        ]
        if not trigger_terms:
            continue
        matches.append(
            {
                "type": event_type,
                "match_count": len(trigger_terms),
                "confidence": round(confidence, 3),
                "trigger_terms": trigger_terms,
                "raw_article_ids": [raw_id] if raw_id > 0 else [],
                "source_indexes": _source_indexes([raw_id], source_index_by_raw_id),
                "evidence_text": _sentence_containing(text, trigger_terms[0]),
                "source": source,
            }
        )
    return matches


def _event_trigger_terms(event_type: str, text: str) -> list[str]:
    lowered = str(text or "").lower()
    return [
        keyword
        for keyword in EVENT_TYPE_KEYWORDS.get(event_type, [])
        if keyword.lower() in lowered
        and _event_keyword_has_context(
            event_type=event_type,
            keyword=keyword,
            lowered_text=lowered,
        )
    ]


def _event_keyword_has_context(
    *,
    event_type: str,
    keyword: str,
    lowered_text: str,
) -> bool:
    guarded_keywords = EVENT_KEYWORDS_REQUIRING_CONTEXT.get(event_type, [])
    if not any(keyword.lower() == guarded.lower() for guarded in guarded_keywords):
        return True
    context_terms = EVENT_TYPE_CONTEXT_KEYWORDS.get(event_type, [])
    return any(term.lower() in lowered_text for term in context_terms)


def _first_fact_matching_event(
    facts: list[dict[str, Any]],
    trigger_terms: list[str],
) -> dict[str, Any] | None:
    lowered_terms = [term.lower() for term in trigger_terms]
    for fact in facts:
        text = f"{fact.get('fact') or ''} {fact.get('evidence_text') or ''}".lower()
        if any(term in text for term in lowered_terms):
            return fact
    return facts[0] if facts else None


def _text_by_article(items: list[dict[str, Any]]) -> dict[int, str]:
    by_id: dict[int, str] = {}
    for item in items or []:
        raw_id = _raw_id(item)
        parts = [
            str(item.get("title") or ""),
            str(item.get("content") or ""),
            *_parser_texts(item),
        ]
        by_id[raw_id] = " ".join(part for part in parts if part).strip()
    return by_id


def _parser_texts(item: dict[str, Any]) -> list[str]:
    parsed = _parser_result(item)
    texts: list[str] = []
    for chunk in _dict_list(parsed.get("document_chunks")):
        texts.append(str(chunk.get("text") or ""))
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        texts.extend(str(value or "") for value in sections.values())
    for signal in _dict_list(parsed.get("topic_signals")):
        texts.append(
            str(
                signal.get("summary")
                or signal.get("topic")
                or signal.get("signal")
                or signal.get("text")
                or ""
            )
        )
    return texts


def _parser_result(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("parser_result")
    if isinstance(value, dict):
        return value
    metadata = _metadata(item)
    value = metadata.get("parser_result")
    return value if isinstance(value, dict) else {}


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _company_alias_hits(text: str) -> list[tuple[str, str]]:
    lowered = str(text or "").lower()
    if not lowered:
        return []
    hits: list[tuple[str, str]] = []
    for company_id, config in COMPANIES.items():
        for alias in config.get("aliases", [company_id]):
            if str(alias).lower() in lowered:
                hits.append((company_id, str(alias)))
    for company_id, global_config in GLOBAL_COMPANIES.items():
        for alias in global_config.get("aliases", [company_id]):
            if str(alias).lower() in lowered:
                hits.append((company_id, str(alias)))
    return _dedupe_company_hits(hits)


def _normalize_company_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in COMPANIES or text in GLOBAL_COMPANIES:
        return text
    lowered = text.lower()
    for company_id, config in COMPANIES.items():
        if lowered == company_id.lower():
            return company_id
        if any(lowered == str(alias).lower() for alias in config.get("aliases", [])):
            return company_id
    for company_id, global_config in GLOBAL_COMPANIES.items():
        if lowered == company_id.lower():
            return company_id
        if any(lowered == str(alias).lower() for alias in global_config.get("aliases", [])):
            return company_id
    return ""


def _company_label(company_id: str) -> str:
    if company_id in COMPANIES:
        return company_name_ko(company_id)
    if company_id in GLOBAL_COMPANIES:
        return global_company_name_ko(company_id)
    return company_id


def _normalize_sector_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in SECTOR_IDS else "other"


def _sector_confidence(policy: IntegrationPolicy, key: str, sector_id: str) -> float:
    value = _confidence(policy, key)
    return min(value, 0.55) if sector_id == "other" else value


def _item_title(items: list[dict[str, Any]], raw_id: int) -> str:
    for item in items or []:
        if _raw_id(item) == raw_id:
            return str(item.get("title") or "")
    return ""


def _sentence_containing(text: str, term: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact or not term:
        return ""
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*|(?<=다)\s+", compact)
    lowered_term = term.lower()
    for part in parts:
        if lowered_term in part.lower():
            return part.strip()
    index = compact.lower().find(lowered_term)
    if index < 0:
        return compact[:240]
    start = max(0, index - 100)
    end = min(len(compact), index + len(term) + 140)
    return compact[start:end].strip()


def _number_type(text: str) -> str:
    lowered = text.lower()
    if "%" in lowered:
        return "percentage"
    if any(unit in lowered for unit in ("원", "달러", "usd", "krw")):
        return "money"
    if any(unit in lowered for unit in ("년", "월", "일", "분기")):
        return "date_or_period"
    if any(unit in lowered for unit in ("건", "명", "개")):
        return "count"
    return "number"


def _normalize_date(value: Any) -> str:
    text = str(value or "")
    match = _ISO_DATE_PATTERN.search(text)
    if match:
        return "-".join(match.groups())
    match = _DATE_YMD_PATTERN.search(text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _dates_in_text(text: str) -> list[str]:
    dates: list[str] = []
    for match in _ISO_DATE_PATTERN.finditer(str(text or "")):
        dates.append("-".join(match.groups()))
    for match in _DATE_YMD_PATTERN.finditer(str(text or "")):
        year, month, day = match.groups()
        dates.append(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")
    return _dedupe_strings(dates)


def _source_indexes(
    raw_article_ids: list[int],
    source_index_by_raw_id: dict[int, int],
) -> list[int]:
    return _dedupe_ints(
        source_index_by_raw_id.get(_safe_int(raw_id), 0) for raw_id in raw_article_ids
    )


def _source_id(source: dict[str, Any]) -> int:
    return _safe_int(source.get("raw_article_id") or source.get("article_id"))


def _fact_raw_id(fact: dict[str, Any]) -> int:
    return _safe_int(fact.get("raw_article_id") or fact.get("article_id"))


def _raw_id(item: dict[str, Any]) -> int:
    return _safe_int(
        item.get("raw_article_id")
        or item.get("id")
        or item.get("preprocess_id")
        or item.get("article_id")
    )


def _all_raw_ids(text_by_article: dict[int, str]) -> list[int]:
    return [raw_id for raw_id in text_by_article if raw_id > 0]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    return [str(value).strip()] if str(value).strip() else []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _confidence(policy: IntegrationPolicy, key: str) -> float:
    return float(
        policy.frame_candidate_confidence.get(
            key,
            policy.frame_candidate_confidence["fallback"],
        )
    )


def _slug(value: str) -> str:
    slug = re.sub(r"\s+", "_", str(value or "").strip().lower())
    slug = re.sub(r"[^0-9a-zA-Z가-힣_:-]+", "", slug)
    return slug or "topic"


def _dedupe_company_hits(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for company_id, alias in values:
        key = (company_id, alias.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((company_id, alias))
    return out


def _dedupe_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _dedupe_ints(values: Any) -> list[int]:
    out: list[int] = []
    for value in values:
        number = _safe_int(value)
        if number > 0 and number not in out:
            out.append(number)
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_issue_frame"]
