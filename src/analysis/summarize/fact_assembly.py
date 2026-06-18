"""summarize fact_assembly — extracted from facade (move-only)."""

from __future__ import annotations

import json
import re
from typing import Any

from src.analysis.summarize.article_selection import (  # noqa: F401
    _analysis_article_limit,
    _analysis_article_score,
    _article_dedupe_text,
    _article_evidence_score,
    _article_prompt_snippets,
    _build_fetch_ids,
    _company_alias_title_tokens,
    _diverse_articles_from_same_event_group,
    _format_articles,
    _full_text_article_ids,
    _is_near_duplicate_article,
    _is_near_duplicate_snippet,
    _normalize_title_event_token,
    _same_event_title_groups,
    _same_title_event,
    _select_analysis_articles,
    _snippet_score,
    _title_event_tokens,
    _title_group_features,
    _useful_title_event_token,
)
from src.analysis.summarize.config import (  # noqa: F401
    _ALLOWED_EVIDENCE_TYPES,
    _ARTICLE_CONTENT_CHARS,
    _ARTICLE_FACT_EXTRACTION_PROMPT,
    _ARTICLE_UI_BOILERPLATE_MARKERS,
    _COMPACT_ARTICLE_CONTENT_CHARS,
    _EVENT_TYPE_VALUES,
    _EVENT_TYPES,
    _FACT_EXTRACTION_BATCH_SIZE,
    _FACT_EXTRACTION_MAX_TOKENS,
    _FACT_ID_SUMMARY_PROMPT,
    _FACT_TYPES,
    _FULL_TEXT_ARTICLE_LIMIT,
    _INDUSTRY_TREND_ALIASES,
    _INDUSTRY_TREND_COMPANY_ID,
    _KNOWN_COMPANY_ALIASES,
    _LLM_MODEL,
    _MAJORITY_THRESHOLD,
    _MAX_ANALYZED_ARTICLES,
    _MIN_ANALYZED_ARTICLES,
    _MIXED_THRESHOLD,
    _NEAR_DUPLICATE_SIMILARITY,
    _NUMBER_TOKEN_PATTERN,
    _PEER_ALIASES,
    _PROMPT_VERSION,
    _SNIPPET_CANDIDATE_SENTENCES,
    _SNIPPET_DEDUP_SIMILARITY,
    _SNIPPETS_PER_ARTICLE,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _SUMMARY_MAX_TOKENS,
    _SUMMARY_ROLES,
    _SUPPORTING_ARTICLE_CONTENT_CHARS,
    _VALIDATION_MAX_TOKENS,
    _env_float,
    _env_int,
    _get_llm,
    _llm,
)
from src.analysis.summarize.fact_extraction import (  # noqa: F401
    _coerce_summary_role,
    _combined_evidence_type,
    _default_summary_role,
    _evidence_type_from_fact_type,
    _extract_article_fact_notes,
    _extract_article_fact_notes_batch,
    _important_single_core_facts,
    _invoke_fact_extraction_llm,
    _is_financial_only_fact,
    _is_length_limit_error,
    _is_market_data_fact,
    _LengthLimitError,
    _merge_article_facts,
    _normalize_article_fact_note,
    _normalize_fact_object,
    _normalize_fact_type,
    _normalize_fact_type_value,
    _normalize_summary_role,
    _normalize_uncertain_fact,
    _normalize_unique_fact,
    _response_finish_reason,
    _response_hit_length_limit,
    _response_token_usage,
    _summary_role_priority,
)
from src.analysis.summarize.rule_based_facts import (  # noqa: F401
    _contract_candidate_sentences,
    _contract_detail_facts_from_article,
    _contract_entities,
    _contract_fact_sentence,
    _dedupe_contract_facts,
    _first_sentence_matching,
    _rule_based_article_fact_notes,
    _rule_based_entities,
    _rule_based_event_type,
    _rule_based_fact_type_and_role,
    _scope_fact_sentence,
    _select_rule_based_sentences,
)
from src.analysis.summarize.text_utils import (  # noqa: F401
    _append_reason,
    _article_ids,
    _article_numeric_id,
    _as_int_list,
    _as_list,
    _chunked,
    _clamp_float,
    _clean_domain_term,
    _clean_json_response,
    _compact,
    _company_display_name,
    _company_list,
    _coverage_info,
    _date_tokens,
    _dedupe_ints,
    _dedupe_keep_order,
    _dedupe_similar_texts,
    _detect_conflict_notes,
    _empty_summary,
    _ensure_sentence,
    _escape_json_string_newlines,
    _event_verbs_in_text,
    _extract_json_object_text,
    _fact_key,
    _has_bad_korean_join,
    _has_business_scope_terms,
    _has_detail_preservation_terms,
    _has_uncertain_fact_marker,
    _has_unique_fact_importance,
    _is_article_ui_boilerplate,
    _join_warnings,
    _matched_companies,
    _metadata,
    _normalize_content,
    _normalize_event_type,
    _normalize_number_token,
    _normalize_string_list,
    _number_token_covered,
    _number_tokens,
    _parse_json,
    _render_prompt,
    _repair_json_text,
    _safe_int,
    _safe_json_loads,
    _split_evidence_sentences,
    _strip_article_ui_boilerplate,
    _summary_mentions_company,
    _summary_metadata,
    _target_company_aliases,
    _text_similarity,
    normalize_korean_spacing,
)
from src.config.companies import COMPANY_ALIASES


def _build_cluster_fact_intelligence(merged_facts: dict[str, Any]) -> dict[str, Any]:
    all_facts = [
        *merged_facts.get("common_facts", []),
        *merged_facts.get("unique_facts", []),
        *merged_facts.get("uncertain_facts", []),
    ]
    return {
        "common_facts": merged_facts.get("common_facts", []),
        "unique_facts": merged_facts.get("unique_facts", []),
        "uncertain_facts": merged_facts.get("uncertain_facts", []),
        "conflict_notes": merged_facts.get("conflict_notes", []),
        "numbers_and_dates": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("numbers_and_dates"))
            ]
        ),
        "customers_or_industries": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("customers_or_industries"))
            ]
        ),
        "products_or_services": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("products_or_services"))
            ]
        ),
        "activity_types": _dedupe_keep_order(
            [
                value
                for fact in all_facts
                for value in _normalize_string_list(fact.get("activity_types"))
            ]
        ),
    }


def _classify_cluster_event_type(
    cluster_fact_intelligence: dict[str, Any],
    articles: list[dict[str, Any]],
) -> str:
    activity_types = _normalize_string_list(cluster_fact_intelligence.get("activity_types"))
    for event_type in _EVENT_TYPES:
        if event_type in activity_types and event_type != "unknown":
            return event_type
    text = " ".join(
        [
            json.dumps(cluster_fact_intelligence, ensure_ascii=False),
            *[
                f"{article.get('title') or ''} {article.get('content') or ''}"
                for article in articles
            ],
        ]
    )
    return _classify_event_type_from_text(text)


def _classify_event_type_from_text(text: str) -> str:
    return "general_update" if text.strip() else "unknown"


def _build_extracted_facts(
    *,
    cluster_id: int,
    article_fact_notes: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    cluster_event_type: str,
) -> list[dict[str, Any]]:
    """기사 fact note에 안정적인 fact_id를 붙여 요약 가능한 fact 목록으로 변환한다."""
    facts: list[dict[str, Any]] = []
    counters: dict[int, int] = {}
    article_by_id = {
        _article_numeric_id(article): article
        for article in articles
        if _article_numeric_id(article) > 0
    }

    def add_fact(
        *,
        article_id: int,
        raw_fact: str,
        evidence_text: str,
        fact_type: str,
        summary_role: str | None = None,
        numbers: list[str] | None = None,
        entities: list[str] | None = None,
        activity_type: str | None = None,
        confidence: str = "medium",
    ) -> None:
        text = normalize_korean_spacing(raw_fact)
        evidence = normalize_korean_spacing(evidence_text or raw_fact)
        if not text or not evidence:
            return
        if _fact_is_off_topic_for_article(
            f"{text} {evidence}",
            article=article_by_id.get(article_id) or {},
        ):
            return
        activity = activity_type or cluster_event_type
        if _is_market_data_fact(f"{text} {evidence}") and _normalize_event_type(activity) not in {
            "stock_market",
            "analyst_report",
        }:
            return
        if _is_duplicate_extracted_fact(facts, article_id, text, evidence):
            return
        counters[article_id] = counters.get(article_id, 0) + 1
        inferred_type = _normalize_fact_type(
            fact_type=fact_type,
            text=f"{text} {evidence}",
            activity_type=activity,
        )
        normalized_role = _normalize_summary_role(summary_role, fact_type=inferred_type)
        normalized_role = _coerce_summary_role(
            role=normalized_role,
            fact_type=inferred_type,
            text=f"{text} {evidence}",
            activity_type=activity,
        )
        facts.append(
            {
                "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                "article_id": article_id,
                "fact_type": inferred_type,
                "summary_role": normalized_role,
                "role_priority": _summary_role_priority(normalized_role),
                "evidence_text": evidence,
                "normalized_fact": text,
                "entities": _dedupe_keep_order(_normalize_string_list(entities)),
                "numbers": _dedupe_keep_order(
                    [*_normalize_string_list(numbers), *_number_tokens(evidence)]
                ),
                "dates": _date_tokens(evidence),
                "event_verbs": _event_verbs_in_text(f"{text} {evidence}"),
                "confidence": confidence if confidence in {"high", "medium", "low"} else "medium",
            }
        )

    for note in article_fact_notes:
        article_id = _safe_int(note.get("article_id"))
        if article_id <= 0:
            continue
        for fact in note.get("core_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            entities = [
                *_normalize_string_list(fact.get("products_or_services")),
                *_normalize_string_list(fact.get("customers_or_industries")),
            ]
            add_fact(
                article_id=article_id,
                raw_fact=text,
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="general_fact",
                summary_role=str(fact.get("summary_role") or ""),
                numbers=_normalize_string_list(fact.get("numbers_and_dates")),
                entities=entities,
                activity_type=str(fact.get("activity_type") or ""),
                confidence="high",
            )
        for fact in note.get("unique_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            add_fact(
                article_id=article_id,
                raw_fact=text,
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="general_fact",
                summary_role=str(fact.get("summary_role") or ""),
                activity_type=cluster_event_type,
                confidence="medium",
            )
        for fact in note.get("uncertain_facts", []):
            text = str(fact.get("fact") or "").strip()
            if not text:
                continue
            add_fact(
                article_id=article_id,
                raw_fact=_soften_uncertain_sentence(text),
                evidence_text=str(fact.get("evidence_text") or text),
                fact_type="uncertain_fact",
                summary_role="uncertainty_detail",
                activity_type=cluster_event_type,
                confidence="medium",
            )

    for article in articles:
        article_id = _safe_int(article.get("id") or article.get("raw_article_id"))
        if article_id <= 0:
            continue
        for fact in _contract_detail_facts_from_article(article):
            add_fact(
                article_id=article_id,
                raw_fact=fact["fact"],
                evidence_text=fact["evidence_text"],
                fact_type=fact["fact_type"],
                summary_role=fact["summary_role"],
                numbers=fact.get("numbers"),
                entities=fact.get("entities"),
                activity_type="contract",
                confidence="high",
            )

    if len(facts) < 3:
        _add_article_fallback_facts(
            facts=facts,
            counters=counters,
            cluster_id=cluster_id,
            articles=articles,
            cluster_event_type=cluster_event_type,
        )

    return facts


def _is_duplicate_extracted_fact(
    facts: list[dict[str, Any]],
    article_id: int,
    normalized_fact: str,
    evidence_text: str,
) -> bool:
    for fact in facts:
        if _safe_int(fact.get("article_id")) != article_id:
            continue
        if _text_similarity(str(fact.get("normalized_fact") or ""), normalized_fact) >= 0.9:
            return True
        if _text_similarity(str(fact.get("evidence_text") or ""), evidence_text) >= 0.9:
            return True
    return False


def _add_article_fallback_facts(
    *,
    facts: list[dict[str, Any]],
    counters: dict[int, int],
    cluster_id: int,
    articles: list[dict[str, Any]],
    cluster_event_type: str,
) -> None:
    for article in articles:
        article_id = _article_numeric_id(article)
        if article_id <= 0:
            continue
        title = normalize_korean_spacing(article.get("title") or "")
        if title and not _is_duplicate_extracted_fact(facts, article_id, title, title):
            counters[article_id] = counters.get(article_id, 0) + 1
            facts.append(
                {
                    "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                    "article_id": article_id,
                    "fact_type": _normalize_fact_type(
                        fact_type="general_fact",
                        text=title,
                        activity_type=cluster_event_type,
                    ),
                    "summary_role": "main_event",
                    "role_priority": _summary_role_priority("main_event"),
                    "evidence_text": title,
                    "normalized_fact": _title_to_fact_sentence(title),
                    "entities": [],
                    "numbers": _number_tokens(title),
                    "dates": _date_tokens(title),
                    "event_verbs": _event_verbs_in_text(title),
                    "confidence": "medium",
                }
            )
        for sentence in _split_evidence_sentences(article.get("content") or "", limit=8):
            if len(facts) >= 6:
                return
            if _is_duplicate_extracted_fact(facts, article_id, sentence, sentence):
                continue
            counters[article_id] = counters.get(article_id, 0) + 1
            facts.append(
                {
                    "fact_id": f"c{cluster_id}_a{article_id}_f{counters[article_id]}",
                    "article_id": article_id,
                    "fact_type": _normalize_fact_type(
                        fact_type="general_fact",
                        text=sentence,
                        activity_type=cluster_event_type,
                    ),
                    "summary_role": "main_event",
                    "role_priority": _summary_role_priority("main_event"),
                    "evidence_text": sentence,
                    "normalized_fact": normalize_korean_spacing(sentence),
                    "entities": [],
                    "numbers": _number_tokens(sentence),
                    "dates": _date_tokens(sentence),
                    "event_verbs": _event_verbs_in_text(sentence),
                    "confidence": "low",
                }
            )


def _fact_is_off_topic_for_article(text: str, *, article: dict[str, Any]) -> bool:
    title = str(article.get("title") or "").strip()
    if not title:
        return False
    title_tokens = _article_topic_tokens(title)
    if len(title_tokens) < 2:
        return False
    value = str(text or "")
    fact_tokens = _article_topic_tokens(value)
    if title_tokens & fact_tokens:
        return False
    if _article_company_alias_mentioned(value, article):
        return False
    return True


def _article_topic_tokens(text: str) -> set[str]:
    stopwords = {
        "속보",
        "단독",
        "특징주",
        "정부",
        "사업",
        "참여",
        "선정",
        "체결",
        "규모",
        "지원",
        "구축",
        "확보",
        "운용",
        "관련",
        "오늘",
        "이번",
    }
    return {
        token
        for token in _article_similarity_tokens(text)
        if len(token) >= 2 and token not in stopwords and not token.isdigit()
    }


def _article_similarity_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", str(text or ""))
        if len(token) >= 2
    }


def _article_company_alias_mentioned(text: str, article: dict[str, Any]) -> bool:
    companies = [
        *_normalize_string_list(article.get("company")),
        *_normalize_string_list(article.get("matched_companies")),
        *_normalize_string_list(article.get("matched_company")),
    ]
    value = str(text or "")
    for company_id in companies:
        aliases = _PEER_ALIASES.get(company_id) or COMPANY_ALIASES.get(company_id) or []
        if any(
            alias and re.search(re.escape(str(alias)), value, re.IGNORECASE) for alias in aliases
        ):
            return True
    return False


def _soften_uncertain_sentence(text: str) -> str:
    sentence = normalize_korean_spacing(text).rstrip()
    if _has_uncertain_fact_marker(sentence):
        return sentence
    if sentence.endswith("했다."):
        return sentence[:-3] + "한 것으로 소개됐다."
    if sentence.endswith("한다."):
        return sentence[:-3] + "하는 방향으로 제시됐다."
    if sentence.endswith("있다."):
        return sentence[:-3] + "있는 것으로 설명됐다."
    return sentence


def _title_to_fact_sentence(title: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", str(title or "")).strip()
    text = normalize_korean_spacing(text)
    return text if text.endswith((".", "다", "요", "죠")) else f"{text}."
