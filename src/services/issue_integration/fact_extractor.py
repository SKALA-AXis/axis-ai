"""Deterministic fact extraction for AnalysisInputBundle construction."""

from __future__ import annotations

import json
import re
from typing import Any

from src.services.issue_integration.lexicon import FACT_TYPE_INDICATORS
from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy

_NUMBER_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|USD|KRW|usd|krw|건|명|개|분기|년|월|일)?"
)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|(?<=[다요음임함됨됨\.])\s+")


def extract_facts_from_articles(
    articles: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for article in articles:
        article_id = _raw_article_id(article)
        title = str(article.get("title") or "").strip()
        if title:
            fact_id = (
                f"raw_article:{article_id}:title"
                if article_id
                else f"title:{len(facts) + 1}"
            )
            facts.append(
                _fact(
                    fact_id=fact_id,
                    article_id=article_id,
                    fact=title,
                    evidence_text=title,
                    source_type=article.get("source_type"),
                    fact_type="general_fact",
                    derived_from="title",
                )
            )
        facts.extend(_facts_from_structured_rows(article, article_id))
        facts.extend(_facts_from_parser_result(article, article_id, policy=policy))
        if not _article_has_structured_or_parser_facts(article):
            facts.extend(_facts_from_content_sentences(article, article_id, policy=policy))
    return _dedupe_facts(facts)


def companies_from_articles(
    articles: list[dict[str, Any]],
    classification: dict[str, Any] | None = None,
) -> list[str]:
    classification = classification or {}
    values: list[str] = []
    for key in ("company", "companies", "main_company", "matched_companies"):
        values.extend(_normalize_string_list(classification.get(key)))
    for article in articles:
        values.extend(_normalize_string_list(article.get("company")))
        values.extend(_normalize_string_list(article.get("matched_companies")))
        metadata = _metadata(article)
        values.extend(_normalize_string_list(metadata.get("company")))
        values.extend(_normalize_string_list(metadata.get("matched_companies")))
    return _dedupe_strings(values)


def sectors_from_articles(
    articles: list[dict[str, Any]],
    classification: dict[str, Any] | None = None,
) -> list[str]:
    classification = classification or {}
    values: list[str] = []
    for key in ("sector", "sectors", "matched_sectors"):
        values.extend(_normalize_string_list(classification.get(key)))
    for article in articles:
        values.extend(_normalize_string_list(article.get("matched_sectors")))
        values.extend(_normalize_string_list(article.get("sector")))
        metadata = _metadata(article)
        values.extend(_normalize_string_list(metadata.get("matched_sectors")))
        values.extend(_normalize_string_list(metadata.get("sector")))
    return _dedupe_strings(values)


def dominant_source_type(articles: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for article in articles:
        source_type = str(article.get("source_type") or "").strip()
        if source_type:
            counts[source_type] = counts.get(source_type, 0) + 1
    if not counts:
        return "unknown"
    top_count = max(counts.values())
    winners = sorted(source_type for source_type, count in counts.items() if count == top_count)
    return winners[0] if len(winners) == 1 else "mixed"


def sources_from_articles(
    articles: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    return [
        {
            "article_id": _raw_article_id(article),
            "raw_article_id": _raw_article_id(article),
            "title": article.get("title"),
            "url": article.get("url"),
            "source_name": article.get("source_name") or article.get("publisher"),
            "source_type": article.get("source_type"),
            "content_type": article.get("content_type"),
            "published_at": article.get("published_at"),
            "collected_at": article.get("collected_at"),
            "processing_status": article.get("processing_status"),
            "crawl_status": article.get("crawl_status"),
            "error_message": article.get("error_message"),
            "relevance_label": article.get("relevance_label"),
            "relevance_score": article.get("relevance_score"),
            "relevance_reason": article.get("relevance_reason"),
            "importance_score": article.get("importance_score"),
            "importance_level": article.get("importance_level"),
            "companies": _normalize_string_list(article.get("company")),
            "matched_companies": _normalize_string_list(article.get("matched_companies")),
            "matched_sectors": _normalize_string_list(article.get("matched_sectors")),
            "link_check": _link_check(article),
            "is_analysis_eligible": is_analysis_eligible_row(article, policy=policy),
        }
        for article in articles
    ]


def evidence_snippets(
    items: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    if facts:
        return [
            {
                "article_id": fact.get("article_id"),
                "raw_article_id": fact.get("raw_article_id") or fact.get("article_id"),
                "text": fact.get("evidence_text") or fact.get("fact"),
                "source_type": fact.get("source_type"),
                "fact_id": fact.get("fact_id"),
            }
            for fact in facts
            if fact.get("evidence_text") or fact.get("fact")
        ]
    snippets: list[dict[str, Any]] = []
    for item in items:
        article_id = _raw_article_id(item)
        sentence = _best_content_sentence(item, policy=policy)
        if sentence:
            snippets.append(
                {
                    "article_id": article_id,
                    "raw_article_id": article_id,
                    "text": sentence,
                    "source_type": item.get("source_type"),
                }
            )
    return snippets


def parser_result(article: dict[str, Any]) -> dict[str, Any]:
    value = article.get("parser_result")
    if isinstance(value, dict):
        return value
    metadata = _metadata(article)
    if isinstance(metadata.get("parser_result"), dict):
        return metadata["parser_result"]
    return {}


def _facts_from_structured_rows(article: dict[str, Any], article_id: int) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, metric in enumerate(_as_dict_list(article.get("financial_metrics")), start=1):
        label = str(metric.get("metric_label") or metric.get("metric_name") or "").strip()
        value = metric.get("value_numeric") or metric.get("value_krwbn") or metric.get("value_krw")
        evidence = str(metric.get("evidence_text") or "").strip()
        if not label and value is None and not evidence:
            continue
        fact_text = " ".join(str(part) for part in (label, value) if part not in {None, ""})
        facts.append(
            _fact(
                fact_id=f"raw_article:{article_id}:metric:{index}",
                article_id=article_id,
                fact=fact_text or evidence,
                evidence_text=evidence or fact_text,
                source_type=article.get("source_type"),
                fact_type="financial_metric",
                derived_from="financial_metric",
                extra={
                    "metric_name": metric.get("metric_name"),
                    "metric_label": label,
                    "value": value,
                    "unit": metric.get("unit"),
                    "period": metric.get("period"),
                },
            )
        )
    for index, signal in enumerate(_as_dict_list(article.get("business_signals")), start=1):
        summary = str(signal.get("summary") or "").strip()
        evidence = str(signal.get("evidence_text") or "").strip()
        if not summary and not evidence:
            continue
        facts.append(
            _fact(
                fact_id=f"raw_article:{article_id}:business_signal:{index}",
                article_id=article_id,
                fact=summary or evidence,
                evidence_text=evidence or summary,
                source_type=article.get("source_type"),
                fact_type="business_signal",
                derived_from="business_signal",
                extra={
                    "business_area": signal.get("business_area"),
                    "signal_type": signal.get("signal_type"),
                    "sentiment": signal.get("sentiment"),
                    "confidence": signal.get("confidence"),
                },
            )
        )
    return facts


def _facts_from_parser_result(
    article: dict[str, Any],
    article_id: int,
    *,
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    parsed = parser_result(article)
    if not parsed:
        return []
    facts: list[dict[str, Any]] = []
    source_type = article.get("source_type")
    for index, signal in enumerate(_as_dict_list(parsed.get("topic_signals")), start=1):
        summary = str(
            signal.get("summary")
            or signal.get("topic")
            or signal.get("signal")
            or signal.get("text")
            or ""
        ).strip()
        evidence = str(signal.get("evidence_text") or signal.get("evidence") or summary).strip()
        if not summary and not evidence:
            continue
        section_key = str(signal.get("section_key") or "").strip()
        facts.append(
            _fact(
                fact_id=f"raw_article:{article_id}:topic_signal:{index}",
                article_id=article_id,
                fact=summary or evidence,
                evidence_text=evidence or summary,
                source_type=source_type,
                fact_type=_fact_type_for_text(" ".join([section_key, summary, evidence])),
                derived_from="business_signal",
                extra={
                    "section_key": section_key,
                    "source_chunk_uid": signal.get("chunk_id") or signal.get("source_chunk_uid"),
                },
            )
        )
    for index, chunk in enumerate(_as_dict_list(parsed.get("document_chunks")), start=1):
        text = _compact_text(chunk.get("text"), policy=policy)
        if not text:
            continue
        section_key = str(chunk.get("section_key") or "").strip()
        section_title = str(chunk.get("section_title") or "").strip()
        fact_text = f"{section_title}: {text}" if section_title else text
        facts.append(
            _fact(
                fact_id=f"raw_article:{article_id}:chunk:{chunk.get('chunk_id') or index}",
                article_id=article_id,
                fact=fact_text,
                evidence_text=text,
                source_type=source_type,
                fact_type=_fact_type_for_text(" ".join([section_key, section_title, text])),
                derived_from="parser_chunk",
                extra={
                    "section_key": section_key,
                    "section_title": section_title,
                    "source_chunk_uid": chunk.get("chunk_id"),
                },
            )
        )
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        for section_key, section_value in sections.items():
            text = _compact_text(section_value, policy=policy)
            if not text:
                continue
            facts.append(
                _fact(
                    fact_id=f"raw_article:{article_id}:section:{section_key}",
                    article_id=article_id,
                    fact=text,
                    evidence_text=text,
                    source_type=source_type,
                    fact_type=_fact_type_for_text(f"{section_key} {text}"),
                    derived_from="parser_section",
                    extra={"section_key": str(section_key)},
                )
            )
    return facts


def _facts_from_content_sentences(
    article: dict[str, Any],
    article_id: int,
    *,
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    sentence = _best_content_sentence(article, policy=policy)
    if not sentence:
        return []
    return [
        _fact(
            fact_id=f"raw_article:{article_id}:content_sentence",
            article_id=article_id,
            fact=sentence,
            evidence_text=sentence,
            source_type=article.get("source_type"),
            fact_type=_fact_type_for_text(sentence),
            derived_from="content_sentence",
        )
    ]


def _best_content_sentence(
    article: dict[str, Any],
    *,
    policy: IntegrationPolicy,
) -> str:
    content = " ".join(str(article.get("content") or "").split())
    if not content:
        return ""
    sentences = [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(content) if part.strip()]
    if not sentences:
        return _compact_text(content, policy=policy)
    scored = sorted(
        ((_sentence_score(sentence, policy=policy), sentence) for sentence in sentences),
        key=lambda item: (item[0], len(item[1])),
        reverse=True,
    )
    return _compact_text(scored[0][1], policy=policy)


def _sentence_score(sentence: str, *, policy: IntegrationPolicy) -> float:
    text = sentence.lower()
    score = 0.0
    score += len(_NUMBER_PATTERN.findall(sentence))
    score += sum(1.0 for term in policy.materiality_terms if term.lower() in text)
    score += 0.5 if any(marker.lower() in text for marker in policy.uncertainty_markers) else 0.0
    return score


def _fact(
    *,
    fact_id: str,
    article_id: int,
    fact: str,
    evidence_text: str,
    source_type: Any,
    fact_type: str,
    derived_from: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(fact or "").strip()
    evidence = str(evidence_text or text).strip()
    payload: dict[str, Any] = {
        "fact_id": fact_id,
        "article_id": article_id,
        "raw_article_id": article_id,
        "fact": text or evidence,
        "evidence_text": evidence or text,
        "source_type": source_type,
        "fact_type": fact_type or "general_fact",
        "derived_from": derived_from,
        "numbers_and_dates": _NUMBER_PATTERN.findall(f"{text} {evidence}"),
    }
    if extra:
        payload.update({key: value for key, value in extra.items() if value not in (None, "", [])})
    return payload


def _fact_type_for_text(value: str) -> str:
    text = value.lower()
    scores = {
        fact_type: sum(1 for token in tokens if token.lower() in text)
        for fact_type, tokens in FACT_TYPE_INDICATORS.items()
    }
    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        return "general_fact"
    winners = sorted(fact_type for fact_type, score in scores.items() if score == best_score)
    return winners[0]


def _article_has_structured_or_parser_facts(article: dict[str, Any]) -> bool:
    return bool(
        _as_dict_list(article.get("financial_metrics"))
        or _as_dict_list(article.get("business_signals"))
        or parser_result(article)
    )


def _compact_text(value: Any, *, policy: IntegrationPolicy) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= policy.sentence_window_chars:
        return text
    boundary = text.rfind(" ", 0, policy.sentence_window_chars)
    if boundary <= 0:
        boundary = policy.sentence_window_chars
    return text[:boundary].rstrip()


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    out: list[dict[str, Any]] = []
    for fact in facts:
        article_id = _safe_int(fact.get("article_id"))
        key = (article_id, _normalize_for_dedupe(fact.get("fact") or fact.get("evidence_text")))
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append(fact)
    return out


def _normalize_for_dedupe(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _metadata(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _link_check(article: dict[str, Any]) -> dict[str, Any]:
    link_check = _metadata(article).get("link_check")
    return link_check if isinstance(link_check, dict) else {}


def is_analysis_eligible_row(
    article: dict[str, Any],
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> bool:
    processing_status = str(article.get("processing_status") or "").strip().lower()
    relevance_label = str(article.get("relevance_label") or "").strip().lower()
    crawl_status = str(article.get("crawl_status") or "").strip().lower()
    if processing_status in policy.ineligible_processing_statuses:
        return False
    if relevance_label in policy.ineligible_relevance_labels:
        return False
    if crawl_status and crawl_status not in policy.eligible_crawl_statuses:
        return False
    if article.get("error_message"):
        return False
    return True


def _normalize_string_list(value: Any) -> list[str]:
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


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raw_article_id(article: dict[str, Any]) -> int:
    return _safe_int(
        article.get("raw_article_id")
        or article.get("id")
        or article.get("preprocess_id")
        or article.get("article_id")
    )


__all__ = [
    "companies_from_articles",
    "dominant_source_type",
    "evidence_snippets",
    "extract_facts_from_articles",
    "is_analysis_eligible_row",
    "parser_result",
    "sectors_from_articles",
    "sources_from_articles",
]
