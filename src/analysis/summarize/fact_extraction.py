"""summarize fact_extraction — extracted from facade (move-only)."""

# ruff: noqa: E501  — long lines inherited from E501-exempt facade

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

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

log = logging.getLogger(__name__)


def _extract_article_fact_notes_batch(
    *,
    cluster_id: int,
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    notes: list[dict[str, Any]] = []
    warnings: list[str] = []
    extraction_failed = False
    for batch in _chunked(articles, _FACT_EXTRACTION_BATCH_SIZE):
        batch_article_ids = [_article_numeric_id(article) for article in batch]
        articles_text = _format_articles(
            articles=batch,
            target_companies=target_companies,
            representative_id=representative_id,
        )
        compact_articles_text = _format_articles(
            articles=batch,
            target_companies=target_companies,
            representative_id=representative_id,
            compact=True,
        )
        parsed, statuses = _extract_article_fact_notes(
            articles_text,
            compact_articles_text=compact_articles_text,
            cluster_id=cluster_id,
            article_ids=batch_article_ids,
        )
        warnings.extend(statuses)
        if "fact_extraction_rule_based_fallback" in statuses:
            extraction_failed = True
        values = parsed.get("article_facts", []) if isinstance(parsed, dict) else []
        if not values:
            values = _rule_based_article_fact_notes(batch, reason="empty_fact_extraction_result")
            warnings.append("fact_extraction_rule_based_candidates_created")
            extraction_failed = True
        for index, item in enumerate(values):
            if isinstance(item, dict):
                note = _normalize_article_fact_note(item)
                if note["article_id"] <= 0 and index < len(batch_article_ids):
                    note["article_id"] = batch_article_ids[index]
                notes.append(note)
    return notes, _dedupe_keep_order(warnings), extraction_failed


def _extract_article_fact_notes(
    articles_text: str,
    *,
    compact_articles_text: str,
    cluster_id: int,
    article_ids: list[int],
) -> tuple[dict[str, Any], list[str]]:
    from src.observability import tracing_config

    def build_prompt(text: str, *, compact_retry: bool = False) -> str:
        prompt_text = _render_prompt(_ARTICLE_FACT_EXTRACTION_PROMPT).replace(
            "{articles_text}",
            text,
        )
        if compact_retry:
            prompt_text += (
                "\n\n추가 지시: 이전 응답이 length limit에 걸렸습니다. "
                "각 article_id마다 core_facts는 최대 3개로 제한하고, evidence_text는 한 문장으로 짧게 유지하세요. "
                "그래도 제품/서비스/플랫폼 공개 fact, 정의/기능 fact, 시연/적용/수치 fact는 우선 보존하세요."
            )
        return prompt_text

    try:
        response = _invoke_fact_extraction_llm(
            build_prompt(articles_text),
            config=tracing_config(
                agent="SourceSummarizer",
                phase="extract_facts",
                prompt_version=_PROMPT_VERSION,
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        if _response_hit_length_limit(response):
            usage = _response_token_usage(response)
            log.warning(
                "기사별 팩트 추출 응답 length finish_reason 감지 | cluster_id=%s article_ids=%s prompt_tokens=%s completion_tokens=%s max_completion_tokens=%s finish_reason=%s retry=%s",
                cluster_id,
                article_ids,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                _FACT_EXTRACTION_MAX_TOKENS,
                _response_finish_reason(response),
                False,
            )
            raise _LengthLimitError("fact extraction response reached length limit")
        parsed, status = _safe_json_loads(content)
        return parsed, [] if status == "parsed" else ["fact_extraction_json_repaired"]
    except Exception as exc:
        if _is_length_limit_error(exc):
            log.warning(
                "기사별 팩트 추출 length limit, compact retry 수행 | cluster_id=%s article_ids=%s max_completion_tokens=%s error=%s",
                cluster_id,
                article_ids,
                _FACT_EXTRACTION_MAX_TOKENS,
                exc,
            )
            try:
                response = _invoke_fact_extraction_llm(
                    build_prompt(compact_articles_text, compact_retry=True),
                    config=tracing_config(
                        agent="SourceSummarizer",
                        phase="extract_facts_compact_retry",
                        prompt_version=_PROMPT_VERSION,
                    ),
                )
                content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                if _response_hit_length_limit(response):
                    usage = _response_token_usage(response)
                    log.warning(
                        "기사별 팩트 추출 compact retry length finish_reason 감지 | cluster_id=%s article_ids=%s prompt_tokens=%s completion_tokens=%s max_completion_tokens=%s finish_reason=%s retry=%s",
                        cluster_id,
                        article_ids,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        _FACT_EXTRACTION_MAX_TOKENS,
                        _response_finish_reason(response),
                        True,
                    )
                    raise _LengthLimitError("compact retry response reached length limit")
                parsed, status = _safe_json_loads(content)
                statuses = ["fact_extraction_length_limit", "fact_extraction_compact_retry"]
                if status != "parsed":
                    statuses.append("fact_extraction_json_repaired")
                return parsed, statuses
            except Exception as retry_exc:
                log.warning(
                    "기사별 팩트 추출 compact retry 실패 | cluster_id=%s article_ids=%s max_completion_tokens=%s error=%s",
                    cluster_id,
                    article_ids,
                    _FACT_EXTRACTION_MAX_TOKENS,
                    retry_exc,
                )
                return {}, [
                    "fact_extraction_length_limit",
                    "fact_extraction_compact_retry_failed",
                    "fact_extraction_rule_based_fallback",
                ]
        log.warning("기사별 팩트 추출 실패 | error=%s", exc)
        return {}, ["fact_extraction_rule_based_fallback"]


def _invoke_fact_extraction_llm(prompt: str, *, config: RunnableConfig | None) -> Any:
    return (
        _get_llm()
        .bind(
            response_format={"type": "json_object"},
            max_completion_tokens=_FACT_EXTRACTION_MAX_TOKENS,
        )
        .invoke(prompt, config=config)
    )


class _LengthLimitError(RuntimeError):
    pass


def _is_length_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "length limit" in text or "finish_reason" in text and "length" in text


def _response_hit_length_limit(response: Any) -> bool:
    return _response_finish_reason(response) == "length"


def _response_finish_reason(response: Any) -> str:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        reason = metadata.get("finish_reason")
        if reason:
            return str(reason)
        generations = metadata.get("generations")
        if isinstance(generations, list) and generations:
            first = generations[0]
            if isinstance(first, dict) and first.get("finish_reason"):
                return str(first["finish_reason"])
    return ""


def _response_token_usage(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(usage, dict):
            return usage
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {}


def _normalize_article_fact_note(item: dict[str, Any]) -> dict[str, Any]:
    article_id = _safe_int(item.get("article_id"))
    return {
        "article_id": article_id,
        "core_facts": [
            _normalize_fact_object(fact, default_type="general_update")
            for fact in _as_list(item.get("core_facts"))
        ],
        "unique_facts": [
            _normalize_unique_fact(fact) for fact in _as_list(item.get("unique_facts"))
        ],
        "uncertain_facts": [
            _normalize_uncertain_fact(fact) for fact in _as_list(item.get("uncertain_facts"))
        ],
    }


def _normalize_fact_object(value: Any, *, default_type: str) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        activity_type = _normalize_event_type(value.get("activity_type") or default_type)
        fact_type = _normalize_fact_type_value(value.get("fact_type"))
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "activity_type": activity_type,
            "fact_type": fact_type,
            "summary_role": _normalize_summary_role(value.get("summary_role"), fact_type=fact_type),
            "numbers_and_dates": _normalize_string_list(value.get("numbers_and_dates")),
            "customers_or_industries": _normalize_string_list(value.get("customers_or_industries")),
            "products_or_services": _normalize_string_list(value.get("products_or_services")),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "activity_type": default_type,
        "fact_type": "general_fact",
        "summary_role": "main_event",
        "numbers_and_dates": [],
        "customers_or_industries": [],
        "products_or_services": [],
    }


def _normalize_unique_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        fact_type = _normalize_fact_type_value(value.get("fact_type"))
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "fact_type": fact_type,
            "summary_role": _normalize_summary_role(value.get("summary_role"), fact_type=fact_type),
            "importance_reason": str(value.get("importance_reason") or "").strip(),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "fact_type": "general_fact",
        "summary_role": "main_event",
        "importance_reason": "",
    }


def _normalize_uncertain_fact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        fact = str(value.get("fact") or "").strip()
        return {
            "fact": fact,
            "evidence_text": str(value.get("evidence_text") or fact).strip(),
            "fact_type": "uncertain_fact",
            "summary_role": "uncertainty_detail",
            "caution": str(value.get("caution") or "전망/예정/가능성 표현").strip(),
        }
    fact = str(value or "").strip()
    return {
        "fact": fact,
        "evidence_text": fact,
        "fact_type": "uncertain_fact",
        "summary_role": "uncertainty_detail",
        "caution": "전망/예정/가능성 표현",
    }


def _merge_article_facts(article_fact_notes: list[dict[str, Any]]) -> dict[str, Any]:
    fact_map: dict[str, dict[str, Any]] = {}
    unique_facts: list[dict[str, Any]] = []
    uncertain_facts: list[dict[str, Any]] = []
    for note in article_fact_notes:
        article_id = _safe_int(note.get("article_id"))
        for fact in note.get("core_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            key = _fact_key(fact_text)
            entry = fact_map.setdefault(
                key,
                {
                    "fact": fact_text,
                    "source_article_ids": [],
                    "evidence_texts": [],
                    "activity_types": [],
                    "numbers_and_dates": [],
                    "customers_or_industries": [],
                    "products_or_services": [],
                },
            )
            if article_id and article_id not in entry["source_article_ids"]:
                entry["source_article_ids"].append(article_id)
            if fact.get("evidence_text"):
                entry["evidence_texts"].append(str(fact.get("evidence_text")))
            if fact.get("activity_type"):
                entry["activity_types"].append(str(fact.get("activity_type")))
            if fact.get("fact_type"):
                entry.setdefault("fact_types", []).append(str(fact.get("fact_type")))
            if fact.get("summary_role"):
                entry.setdefault("summary_roles", []).append(str(fact.get("summary_role")))
            entry["numbers_and_dates"].extend(_normalize_string_list(fact.get("numbers_and_dates")))
            entry["customers_or_industries"].extend(
                _normalize_string_list(fact.get("customers_or_industries"))
            )
            entry["products_or_services"].extend(
                _normalize_string_list(fact.get("products_or_services"))
            )

        for fact in note.get("unique_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            evidence_text = str(fact.get("evidence_text") or fact_text)
            unique_facts.append(
                {
                    "fact": fact_text,
                    "source_article_ids": [article_id] if article_id else [],
                    "evidence_count": 1,
                    "evidence_texts": [evidence_text],
                    "fact_type": fact.get("fact_type") or "general_fact",
                    "summary_role": fact.get("summary_role") or "main_event",
                    "numbers_and_dates": _dedupe_keep_order(
                        [
                            *_normalize_string_list(fact.get("numbers_and_dates")),
                            *_number_tokens(evidence_text),
                            *_date_tokens(evidence_text),
                        ]
                    ),
                    "customers_or_industries": _normalize_string_list(
                        fact.get("customers_or_industries")
                    ),
                    "products_or_services": _normalize_string_list(
                        fact.get("products_or_services")
                    ),
                    "importance_reason": str(fact.get("importance_reason") or ""),
                }
            )

        for fact in note.get("uncertain_facts", []):
            fact_text = str(fact.get("fact") or "").strip()
            if not fact_text:
                continue
            uncertain_facts.append(
                {
                    "fact": fact_text,
                    "source_article_ids": [article_id] if article_id else [],
                    "evidence_count": 1,
                    "evidence_texts": [str(fact.get("evidence_text") or fact_text)],
                    "fact_type": "uncertain_fact",
                    "summary_role": "uncertainty_detail",
                    "caution": str(fact.get("caution") or "확정 사실로 단정하지 않음"),
                }
            )

    common_facts: list[dict[str, Any]] = []
    single_core_facts: list[dict[str, Any]] = []
    for entry in fact_map.values():
        entry["source_article_ids"] = _dedupe_ints(entry["source_article_ids"])
        entry["evidence_texts"] = _dedupe_keep_order(
            [text for text in entry["evidence_texts"] if text]
        )[:5]
        entry["activity_types"] = _dedupe_keep_order(
            [item for item in entry["activity_types"] if item]
        )
        entry["fact_types"] = _dedupe_keep_order(
            [item for item in entry.get("fact_types", []) if item]
        )
        entry["summary_roles"] = _dedupe_keep_order(
            [item for item in entry.get("summary_roles", []) if item]
        )
        entry["numbers_and_dates"] = _dedupe_keep_order(entry["numbers_and_dates"])
        entry["customers_or_industries"] = _dedupe_keep_order(entry["customers_or_industries"])
        entry["products_or_services"] = _dedupe_keep_order(entry["products_or_services"])
        entry["evidence_count"] = len(entry["source_article_ids"])
        if entry["evidence_count"] >= 2:
            common_facts.append(entry)
        else:
            single_core_facts.append(entry)

    unique_facts.extend(_important_single_core_facts(single_core_facts))
    return {
        "common_facts": common_facts,
        "unique_facts": unique_facts,
        "uncertain_facts": uncertain_facts,
        "conflict_notes": _detect_conflict_notes([*common_facts, *unique_facts, *uncertain_facts]),
    }


def _important_single_core_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    important: list[dict[str, Any]] = []
    for fact in facts:
        text = " ".join(
            [
                str(fact.get("fact") or ""),
                " ".join(fact.get("numbers_and_dates", [])),
                " ".join(fact.get("customers_or_industries", [])),
                " ".join(fact.get("products_or_services", [])),
            ]
        )
        if not _has_unique_fact_importance(text):
            continue
        copied = dict(fact)
        copied["importance_reason"] = (
            "단일 기사에만 있지만 수치/일정/고객/서비스/후속 단계 정보가 포함됨"
        )
        important.append(copied)
    return important


def _normalize_fact_type(*, fact_type: str, text: str, activity_type: str) -> str:
    normalized = _normalize_fact_type_value(fact_type)
    if normalized == "numeric_fact" and _has_business_scope_terms(text):
        if _normalize_event_type(activity_type) in {"contract", "partnership"}:
            return "application_fact"
        if _normalize_event_type(activity_type) in {
            "launch",
            "technology_update",
            "general_update",
        }:
            return "application_fact"
    return normalized


def _normalize_fact_type_value(value: Any) -> str:
    fact_type = str(value or "").strip()
    return fact_type if fact_type in _FACT_TYPES else "general_fact"


def _normalize_summary_role(value: Any, *, fact_type: str) -> str:
    role = str(value or "").strip()
    if role in _SUMMARY_ROLES:
        return role
    return _default_summary_role(fact_type)


def _coerce_summary_role(*, role: str, fact_type: str, text: str, activity_type: str) -> str:
    event_type = _normalize_event_type(activity_type)
    if role == "numeric_effect" and _has_business_scope_terms(text):
        if event_type in {"contract", "partnership"} and re.search(
            r"계약|수주|공급\s*계약|공급계약|협약|MOU", text
        ):
            return "main_event"
        if re.search(r"업무|시스템|전환|구축|플랫폼|솔루션|서비스|AI|에이전트", text, re.I):
            return "service_function"
        return "application_case"
    if fact_type == "numeric_fact":
        return role
    if role == "main_event" and re.search(r"기능|역할|지원|자동화|분석|검증|운영|적용|연계", text):
        return "service_function"
    return role


def _is_financial_only_fact(text: str) -> bool:
    value = str(text or "")
    if _has_business_scope_terms(value):
        return False
    return bool(
        re.search(r"매출|영업이익|순이익|주가|시가총액|증가|감소|흑자|적자|억원|조원|%", value)
    )


def _is_market_data_fact(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    strong_terms = (
        "주가",
        "현재가",
        "전일대비",
        "등락률",
        "거래량",
        "시가총액",
        "목표주가",
        "투자의견",
    )
    if any(term in value for term in strong_terms):
        return True
    return bool(
        re.search(
            r"\b(?:KOSPI|KOSDAQ)\b|전\s*거래일|장\s*(초반|마감)|"
            r"(?:상승|하락|급등|급락)\s*(?:마감|출발|전환)",
            value,
            re.I,
        )
    )


def _default_summary_role(fact_type: str) -> str:
    if fact_type == "launch_fact":
        return "main_event"
    if fact_type == "platform_definition_fact":
        return "product_definition"
    if fact_type == "application_fact":
        return "application_case"
    if fact_type == "numeric_fact":
        return "numeric_effect"
    if fact_type == "market_fact":
        return "market_reaction"
    if fact_type == "risk_fact":
        return "risk_detail"
    if fact_type == "uncertain_fact":
        return "uncertainty_detail"
    return "main_event"


def _summary_role_priority(role: str) -> int:
    ordered = [
        "main_event",
        "product_definition",
        "service_function",
        "application_case",
        "numeric_effect",
        "market_reaction",
        "risk_detail",
        "uncertainty_detail",
    ]
    try:
        return ordered.index(role) + 1
    except ValueError:
        return len(ordered)


def _combined_evidence_type(facts: list[dict[str, Any]]) -> str:
    evidence_types = [
        _evidence_type_from_fact_type(str(fact.get("fact_type") or "")) for fact in facts
    ]
    for preferred in (
        "risk_fact",
        "market_reaction_fact",
        "uncertain_fact",
        "core_fact",
        "unique_fact",
        "numeric_fact",
    ):
        if preferred in evidence_types:
            return preferred
    return "reported_fact"


def _evidence_type_from_fact_type(fact_type: str) -> str:
    if fact_type == "market_fact":
        return "market_reaction_fact"
    if fact_type == "risk_fact":
        return "risk_fact"
    if fact_type == "uncertain_fact":
        return "uncertain_fact"
    if fact_type in {"launch_fact", "platform_definition_fact"}:
        return "core_fact"
    if fact_type == "application_fact":
        return "unique_fact"
    if fact_type == "numeric_fact":
        return "numeric_fact"
    return "reported_fact"
