"""summarize rule_based_facts — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any

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


def _rule_based_article_fact_notes(
    articles: list[dict[str, Any]],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for article in articles:
        article_id = _article_numeric_id(article)
        title = normalize_korean_spacing(article.get("title") or "")
        sentences = [
            sentence
            for sentence in _dedupe_keep_order(
                [
                    title,
                    *_split_evidence_sentences(article.get("content") or "", limit=8),
                ]
            )
            if sentence and not _is_article_ui_boilerplate(sentence)
        ]
        if not article_id or not sentences:
            continue

        event_type = _rule_based_event_type(sentences)
        entities = _rule_based_entities(sentences)
        selected = _select_rule_based_sentences(sentences)
        core_facts: list[dict[str, Any]] = []
        for index, sentence in enumerate(selected, start=1):
            fact_type, summary_role = _rule_based_fact_type_and_role(index, sentence, event_type)
            core_facts.append(
                {
                    "fact": sentence,
                    "evidence_text": sentence,
                    "activity_type": event_type,
                    "fact_type": fact_type,
                    "summary_role": summary_role,
                    "numbers_and_dates": _dedupe_keep_order(
                        [*_number_tokens(sentence), *_date_tokens(sentence)]
                    ),
                    "customers_or_industries": [],
                    "products_or_services": [
                        entity for entity in entities if _compact(entity) in _compact(sentence)
                    ][:5],
                }
            )
        notes.append(
            {
                "article_id": article_id,
                "core_facts": core_facts,
                "unique_facts": [],
                "uncertain_facts": [],
                "extraction_warning": reason,
            }
        )
    return notes


def _rule_based_event_type(sentences: list[str]) -> str:
    text = " ".join(sentences).lower()
    event_markers = (
        ("contract", ("수주", "계약", "사업자 선정", "우선협상")),
        ("partnership", ("mou", "협약", "제휴", "협력")),
        ("launch", ("출시", "공개", "선보", "론칭")),
        ("earnings", ("매출", "영업이익", "실적", "순이익")),
        ("stock_market", ("주가", "거래량", "시가총액", "목표주가")),
        ("analyst_report", ("증권사", "리포트", "투자의견", "전망")),
        ("risk", ("장애", "소송", "침해", "해킹", "리스크")),
        ("technology_update", ("기술", "플랫폼", "서비스", "솔루션", "시스템")),
    )
    for event_type, markers in event_markers:
        if any(marker in text for marker in markers):
            return event_type
    return "general_update"


def _rule_based_entities(sentences: list[str]) -> list[str]:
    text = " ".join(sentences)
    quoted = re.findall(r"['\"‘’“”]([^'\"‘’“”]{2,40})['\"‘’“”]", text)
    acronym_like = re.findall(r"\b[A-Z][A-Za-z0-9+\-/]{1,20}\b", text)
    return _dedupe_keep_order(
        [_clean_domain_term(term) for term in [*quoted, *acronym_like] if _clean_domain_term(term)]
    )[:12]


def _select_rule_based_sentences(sentences: list[str]) -> list[str]:
    selected: list[str] = []
    for sentence in sentences:
        if sentence and sentence not in selected:
            selected.append(sentence)
        if len(selected) >= 3:
            break
    while selected and len(selected) < 3:
        selected.append(selected[-1])
    return selected[:3]


def _rule_based_fact_type_and_role(
    index: int,
    sentence: str,
    event_type: str,
) -> tuple[str, str]:
    if index == 1:
        return ("launch_fact" if event_type == "launch" else "general_fact", "main_event")
    if index == 2:
        if event_type == "launch":
            return "platform_definition_fact", "product_definition"
        return "general_fact", "service_function"
    if _number_tokens(sentence):
        return "numeric_fact", "numeric_effect"
    return "application_fact", "application_case"


def _contract_detail_facts_from_article(article: dict[str, Any]) -> list[dict[str, Any]]:
    """계약/수주 기사에서 사업명, 금액, 기간, 매출 대비 비율을 보강 추출한다.

    LLM 추출이 계약 범위를 "공급" 정도로 약화할 때를 막기 위한 일반 보조 규칙이다.
    특정 회사나 사업명을 박지 않고, 기사 문장에 이미 있는 계약 관련 문장만 사용한다.
    """
    title = str(article.get("title") or "").strip()
    body = " ".join(
        str(article.get(key) or "").strip()
        for key in ("content", "body", "summary", "description")
        if str(article.get(key) or "").strip()
    )
    text = normalize_korean_spacing(f"{title}. {body}")
    if not re.search(r"계약|수주|공급\s*계약|공급계약", text):
        return []

    sentences = _contract_candidate_sentences(text)
    facts: list[dict[str, Any]] = []

    main_sentence = _first_sentence_matching(
        sentences,
        include=(r"계약|수주|공급\s*계약|공급계약", r"억|원|규모|사업|프로젝트|전환|구축|공급"),
    )
    if main_sentence:
        facts.append(
            {
                "fact": _contract_fact_sentence(main_sentence),
                "evidence_text": main_sentence,
                "fact_type": "general_fact",
                "summary_role": "main_event",
                "numbers": _number_tokens(main_sentence),
                "entities": _contract_entities(main_sentence),
            }
        )

    scope_sentence = _first_sentence_matching(
        sentences,
        include=(r"사업|프로젝트|전환|구축|공급|시스템|단말|플랫폼|업무",),
        exclude=[main_sentence] if main_sentence else None,
    )
    if scope_sentence:
        facts.append(
            {
                "fact": _scope_fact_sentence(scope_sentence),
                "evidence_text": scope_sentence,
                "fact_type": "application_fact",
                "summary_role": "service_function",
                "numbers": _number_tokens(scope_sentence),
                "entities": _contract_entities(scope_sentence),
            }
        )

    period_sentence = _first_sentence_matching(
        sentences,
        include=(r"계약\s*기간|기간은|20\d{2}년\s*\d{1,2}월\s*\d{1,2}일",),
    )
    if period_sentence:
        facts.append(
            {
                "fact": _ensure_sentence(period_sentence),
                "evidence_text": period_sentence,
                "fact_type": "numeric_fact",
                "summary_role": "numeric_effect",
                "numbers": [*_number_tokens(period_sentence), *_date_tokens(period_sentence)],
                "entities": _contract_entities(period_sentence),
            }
        )

    ratio_sentence = _first_sentence_matching(
        sentences,
        include=(r"최근\s*매출|매출액\s*대비|매출\s*대비|%",),
    )
    if ratio_sentence:
        facts.append(
            {
                "fact": _ensure_sentence(ratio_sentence),
                "evidence_text": ratio_sentence,
                "fact_type": "numeric_fact",
                "summary_role": "numeric_effect",
                "numbers": _number_tokens(ratio_sentence),
                "entities": _contract_entities(ratio_sentence),
            }
        )

    return _dedupe_contract_facts(facts)


def _contract_candidate_sentences(text: str) -> list[str]:
    cleaned = normalize_korean_spacing(text)
    parts = [
        re.sub(r"\s+", " ", item).strip(" -·")
        for item in re.split(r"(?<=[.!?。！？])\s+|(?<=다)\.\s*|(?<=다)\s+", cleaned)
        if re.sub(r"\s+", " ", item).strip(" -·")
    ]
    return [item if item.endswith((".", "다")) else _ensure_sentence(item) for item in parts]


def _first_sentence_matching(
    sentences: list[str],
    *,
    include: tuple[str, ...],
    exclude: list[str | None] | None = None,
) -> str:
    excluded = {re.sub(r"\s+", "", str(item or "")) for item in (exclude or []) if item}
    for sentence in sentences:
        key = re.sub(r"\s+", "", sentence)
        if key in excluded:
            continue
        if all(re.search(pattern, sentence) for pattern in include):
            return sentence
    return ""


def _contract_fact_sentence(sentence: str) -> str:
    return _ensure_sentence(sentence)


def _scope_fact_sentence(sentence: str) -> str:
    return _ensure_sentence(sentence)


def _contract_entities(sentence: str) -> list[str]:
    entities = re.findall(
        r"[가-힣A-Za-z0-9&·+_-]{2,}(?:\s+[가-힣A-Za-z0-9&·+_-]{2,}){0,5}"
        r"(?:사업|프로젝트|계약|시스템|플랫폼|단말|서비스|솔루션|업무|인프라)",
        sentence,
    )
    return _dedupe_keep_order([re.sub(r"\s+", " ", item).strip() for item in entities])


def _dedupe_contract_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in facts:
        key = re.sub(r"[\s.。!?！？,，]+", "", str(fact.get("fact") or ""))
        if key and key not in seen:
            result.append(fact)
            seen.add(key)
    return result
