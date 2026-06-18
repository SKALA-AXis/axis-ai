"""summarize summary_lines — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
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
    _FACT_EXTRACTION_MODE,
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
    _USE_FACT_EXTRACTION_LLM,
    _VALIDATION_MAX_TOKENS,
    _env_bool,
    _env_float,
    _env_int,
    _get_llm,
    _llm,
)
from src.analysis.summarize.fact_assembly import (  # noqa: F401
    _add_article_fallback_facts,
    _build_cluster_fact_intelligence,
    _build_extracted_facts,
    _classify_cluster_event_type,
    _classify_event_type_from_text,
    _is_duplicate_extracted_fact,
    _soften_uncertain_sentence,
    _title_to_fact_sentence,
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
    _is_article_context_detail_snippet,
    _is_article_relevant_snippet,
    _rule_based_article_fact_notes,
    _rule_based_entities,
    _rule_based_event_type,
    _rule_based_fact_notes_need_llm,
    _rule_based_fact_type_and_role,
    _scope_fact_sentence,
    _select_rule_based_sentences,
    _snippet_score,
)
from src.analysis.summarize.text_utils import (  # noqa: F401
    _append_reason,
    _article_company_alias_mentioned,
    _article_ids,
    _article_numeric_id,
    _article_similarity_tokens,
    _article_target_company_alias_mentioned,
    _article_topic_tokens,
    _articles_text,
    _as_int_list,
    _as_list,
    _body_peer_companies,
    _candidate_peer_companies,
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
    _fact_is_off_topic_for_article,
    _fact_key,
    _has_bad_korean_join,
    _has_business_scope_terms,
    _has_detail_preservation_terms,
    _has_uncertain_fact_marker,
    _has_unique_fact_importance,
    _is_article_ui_boilerplate,
    _is_company_neutral_context_detail,
    _is_industry_trend_cluster,
    _is_peer_comparison_issue,
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


def _select_fact_ids_for_summary_lines(
    *,
    extracted_facts: list[dict[str, Any]],
    cluster_event_type: str,
) -> dict[str, list[str]]:
    available = [fact for fact in extracted_facts if fact.get("fact_id")]
    used: set[str] = set()
    selected_facts: list[dict[str, Any]] = []

    def choose(index: int, preferred_roles: tuple[str, ...]) -> list[str]:
        candidates = [
            fact
            for fact in available
            if fact.get("fact_id") not in used and fact.get("summary_role") in preferred_roles
        ]
        if not candidates:
            candidates = [fact for fact in available if fact.get("fact_id") not in used]
        if not candidates:
            candidates = available
        if not candidates:
            return []
        selected = sorted(
            candidates,
            key=lambda fact: (
                _fact_selection_score(fact) - _similar_selected_fact_penalty(fact, selected_facts)
            ),
            reverse=True,
        )[0]
        fact_id = str(selected.get("fact_id"))
        used.add(fact_id)
        selected_facts.append(selected)
        return [fact_id]

    preferences = _line_summary_role_preferences(cluster_event_type)
    desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(available)))
    return {
        str(index): choose(index, preferences[min(index - 1, len(preferences) - 1)])
        for index in range(1, desired_count + 1)
    }


def _line_summary_role_preferences(
    event_type: str,
) -> tuple[tuple[str, ...], ...]:
    event_type = _normalize_event_type(event_type)
    if event_type == "launch":
        return (
            ("main_event",),
            ("product_definition", "service_function"),
            ("application_case", "numeric_effect", "uncertainty_detail"),
            ("application_case", "service_function", "numeric_effect"),
            ("uncertainty_detail", "numeric_effect", "market_reaction"),
        )
    if event_type in {"technology_update", "general_update", "unknown"}:
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("uncertainty_detail", "application_case", "numeric_effect"),
            ("application_case", "service_function", "numeric_effect"),
            ("uncertainty_detail", "risk_detail", "market_reaction"),
        )
    if event_type == "contract":
        return (
            ("main_event",),
            ("service_function", "product_definition", "application_case"),
            ("application_case", "service_function", "product_definition"),
            ("service_function", "application_case", "numeric_effect"),
            ("service_function", "application_case", "uncertainty_detail", "numeric_effect"),
        )
    if event_type == "earnings":
        return (
            ("main_event", "numeric_effect"),
            ("service_function", "product_definition"),
            ("numeric_effect", "uncertainty_detail"),
            ("market_reaction", "numeric_effect"),
            ("uncertainty_detail", "risk_detail"),
        )
    if event_type == "stock_market":
        return (
            ("market_reaction", "main_event"),
            ("service_function", "product_definition"),
            ("numeric_effect", "market_reaction", "uncertainty_detail"),
            ("market_reaction", "risk_detail"),
            ("uncertainty_detail", "numeric_effect"),
        )
    if event_type == "risk":
        return (
            ("risk_detail", "main_event"),
            ("service_function", "product_definition", "application_case"),
            ("risk_detail", "uncertainty_detail", "numeric_effect"),
            ("application_case", "risk_detail"),
            ("uncertainty_detail", "market_reaction"),
        )
    return (
        ("main_event",),
        ("product_definition", "service_function", "application_case"),
        ("numeric_effect", "application_case", "uncertainty_detail"),
        ("application_case", "service_function", "market_reaction"),
        ("uncertainty_detail", "risk_detail", "numeric_effect"),
    )


def _fact_selection_score(fact: dict[str, Any]) -> int:
    score = 0
    role = str(fact.get("summary_role") or "")
    score += {
        "main_event": 8,
        "product_definition": 7,
        "service_function": 7,
        "application_case": 6,
        "uncertainty_detail": 2,
        "risk_detail": 2,
        "numeric_effect": 1,
        "market_reaction": 0,
    }.get(role, 0)
    score += (
        3 if fact.get("confidence") == "high" else 2 if fact.get("confidence") == "medium" else 1
    )
    if fact.get("entities"):
        score += 2
    if fact.get("numbers") or fact.get("dates"):
        score += 2
    if fact.get("event_verbs"):
        score += 1
    text = f"{fact.get('normalized_fact') or ''} {fact.get('evidence_text') or ''}"
    if re.search(r"업무|시스템|고객|서비스|솔루션|플랫폼|에이전트|코딩|협업|문서", text):
        score += 3
    if re.search(r"외부|확대|고도화|제공|지원|활용|적용|연계", text):
        score += 2
    if re.search(r"외부\s*기업|기업\s*고객|사업\s*영역|사업\s*확장|고객으로|고객에게", text):
        score += 5
    if role == "numeric_effect" and _is_financial_only_fact(text):
        score -= 8
    score += min(len(str(fact.get("normalized_fact") or "")) // 30, 3)
    return score


def _similar_selected_fact_penalty(
    fact: dict[str, Any],
    selected_facts: list[dict[str, Any]],
) -> int:
    text = _fact_similarity_text(fact)
    if not text or not selected_facts:
        return 0
    max_similarity = max(
        _text_similarity(text, _fact_similarity_text(selected)) for selected in selected_facts
    )
    if max_similarity >= 0.82:
        return 8
    if max_similarity >= 0.68:
        return 4
    return 0


def _fact_similarity_text(fact: dict[str, Any]) -> str:
    return str(fact.get("normalized_fact") or fact.get("evidence_text") or "").strip()


def _summarize_from_fact_ids(
    *,
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
    selected_fact_ids: dict[str, list[str]],
    target_companies: list[str],
    main_company: str,
) -> dict[str, Any]:
    if not extracted_facts:
        return _fact_id_empty_result(
            main_company=main_company,
            target_companies=target_companies,
            cluster_event_type=cluster_event_type,
            reason="추출된 fact가 없어 요약할 수 없음",
        )
    try:
        from src.observability import tracing_config

        selected_prompt_facts = _selected_facts_by_line(selected_fact_ids, extracted_facts)
        additional_prompt_facts = _additional_facts_for_prompt(
            extracted_facts,
            selected_fact_ids=selected_fact_ids,
        )
        prompt = _render_prompt(_FACT_ID_SUMMARY_PROMPT)
        prompt = (
            prompt.replace("{main_company}", main_company)
            .replace("{target_companies_json}", json.dumps(target_companies, ensure_ascii=False))
            .replace("{cluster_event_type}", cluster_event_type)
            .replace(
                "{selected_facts_json}",
                json.dumps(selected_prompt_facts, ensure_ascii=False, separators=(",", ":")),
            )
            .replace(
                "{all_facts_json}",
                json.dumps(additional_prompt_facts, ensure_ascii=False, separators=(",", ":")),
            )
        )
        response = (
            _get_llm()
            .bind(max_completion_tokens=_SUMMARY_MAX_TOKENS)
            .invoke(
                prompt,
                config=tracing_config(
                    agent="SourceSummarizer",
                    phase="fact_id_summary",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        result = _normalize_fact_id_summary_result(
            _parse_json(content),
            target_companies=target_companies,
            fallback_company=main_company,
            cluster_event_type=cluster_event_type,
            extracted_facts=extracted_facts,
        )
    except Exception as exc:
        log.warning("fact_id 기반 요약 LLM 실패, fallback 사용 | error=%s", exc)
        result = _fallback_fact_id_summary(
            main_company=main_company,
            target_companies=target_companies,
            cluster_event_type=cluster_event_type,
            extracted_facts=extracted_facts,
            selected_fact_ids=selected_fact_ids,
            reason=f"fact_id 요약 LLM 실패: {type(exc).__name__}",
        )

    checked = _validate_fact_id_summary(
        result=result,
        extracted_facts=extracted_facts,
        source_article_ids=[],
        main_company=result.get("main_company") or main_company,
    )
    if checked.get("is_valid_summary"):
        return checked

    fallback = _fallback_fact_id_summary(
        main_company=main_company,
        target_companies=target_companies,
        cluster_event_type=cluster_event_type,
        extracted_facts=extracted_facts,
        selected_fact_ids=selected_fact_ids,
        reason=_append_reason(
            checked.get("reason"), "LLM 결과 검증 실패 후 fallback template 사용"
        ),
    )
    fallback["repair_actions"] = _dedupe_keep_order(
        [
            *_normalize_string_list(checked.get("repair_actions")),
            "fallback_template_from_selected_fact_ids",
        ]
    )
    return fallback


def _normalize_fact_id_summary_result(
    data: dict[str, Any],
    *,
    target_companies: list[str],
    fallback_company: str,
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    mentioned = [
        company_id
        for company_id in _normalize_string_list(data.get("mentioned_peer_companies"))
        if company_id in target_companies
    ]
    main_company = str(data.get("main_company") or fallback_company).strip()
    if main_company not in target_companies:
        main_company = mentioned[0] if mentioned else fallback_company
    line_items = _normalize_summary_line_items(data.get("summary_lines"))
    if not line_items and data.get("fact_summary"):
        line_items = [
            {"line_index": index, "text": text, "fact_ids": []}
            for index, text in enumerate(
                _normalize_string_list(data.get("fact_summary"))[:_SUMMARY_LINE_MAX], start=1
            )
        ]
    line_items = _ensure_fact_summary_lines(line_items, extracted_facts)
    fact_summary = [str(item.get("text") or "").strip() for item in line_items]
    is_valid_summary = (
        bool(data.get("is_valid_summary", True))
        and _SUMMARY_LINE_MIN <= len(fact_summary) <= _SUMMARY_LINE_MAX
        and all(fact_summary)
    )
    result = {
        "is_valid_summary": is_valid_summary,
        "main_company": main_company,
        "mentioned_peer_companies": mentioned or [main_company],
        "cluster_event_type": _normalize_event_type(
            data.get("cluster_event_type") or cluster_event_type
        ),
        "headline": str(data.get("headline") or fact_summary[0] if fact_summary else "").strip(),
        "one_line_summary": str(
            data.get("one_line_summary") or fact_summary[0] if fact_summary else ""
        ).strip(),
        "fact_summary": fact_summary,
        "summary_lines_with_fact_ids": line_items,
        "main_event": str(
            data.get("main_event") or fact_summary[0] if fact_summary else ""
        ).strip(),
        "confidence": _clamp_float(data.get("confidence"), default=0.0),
        "reason": str(data.get("reason") or "").strip(),
    }
    result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)
    return result


def _normalize_summary_line_items(value: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(_as_list(value), start=1):
        if isinstance(item, dict):
            index = _safe_int(item.get("line_index")) or fallback_index
            text = normalize_korean_spacing(item.get("text") or item.get("summary_line") or "")
            fact_ids = [
                str(fact_id) for fact_id in _normalize_string_list(item.get("fact_ids")) if fact_id
            ]
        else:
            index = fallback_index
            text = normalize_korean_spacing(item)
            fact_ids = []
        if 1 <= index <= _SUMMARY_LINE_MAX and text:
            lines.append({"line_index": index, "text": text, "fact_ids": fact_ids})
    by_index: dict[int, dict[str, Any]] = {}
    for item in lines:
        by_index[_safe_int(item.get("line_index"))] = item
    return [by_index[index] for index in range(1, _SUMMARY_LINE_MAX + 1) if index in by_index]


def _ensure_fact_summary_lines(
    line_items: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    line_items = [
        item for item in line_items if 1 <= _safe_int(item.get("line_index")) <= _SUMMARY_LINE_MAX
    ]
    desired_count = len(line_items)
    if desired_count <= 0:
        desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(extracted_facts)))
    result: dict[int, dict[str, Any]] = {
        _safe_int(item.get("line_index")): item
        for item in line_items
        if 1 <= _safe_int(item.get("line_index")) <= _SUMMARY_LINE_MAX
    }
    unused_facts = [fact for fact in extracted_facts if str(fact.get("fact_id"))]
    for index in range(1, desired_count + 1):
        item = result.get(index)
        if item and item.get("fact_ids"):
            item["fact_ids"] = [fact_id for fact_id in item["fact_ids"] if fact_id in fact_by_id]
        if item and item.get("fact_ids"):
            continue
        fact = unused_facts[min(index - 1, len(unused_facts) - 1)] if unused_facts else None
        if fact is None:
            result[index] = {"line_index": index, "text": "", "fact_ids": []}
            continue
        result[index] = {
            "line_index": index,
            "text": _fact_text_for_summary_line(fact),
            "fact_ids": [str(fact.get("fact_id"))],
        }
    return [result[index] for index in range(1, desired_count + 1)]


def _fact_basis_from_summary_line_fact_ids(
    line_items: list[dict[str, Any]],
    extracted_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    basis: list[dict[str, Any]] = []
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        fact_ids = [
            fact_id
            for fact_id in _normalize_string_list(item.get("fact_ids"))
            if fact_id in fact_by_id
        ]
        if not index or not fact_ids:
            continue
        facts = [fact_by_id[fact_id] for fact_id in fact_ids]
        evidence_texts = _dedupe_similar_texts(
            [str(fact.get("evidence_text") or "") for fact in facts]
        )
        basis.append(
            {
                "summary_sentence_index": index,
                "summary_line_index": index,
                "fact": str(item.get("text") or ""),
                "source_article_ids": _dedupe_ints(
                    [_safe_int(fact.get("article_id")) for fact in facts]
                ),
                "fact_ids": fact_ids,
                "evidence_count": len(fact_ids),
                "evidence_type": _combined_evidence_type(facts),
                "evidence_texts": evidence_texts[:3],
            }
        )
    return basis


def _fallback_fact_id_summary(
    *,
    main_company: str,
    target_companies: list[str],
    cluster_event_type: str,
    extracted_facts: list[dict[str, Any]],
    selected_fact_ids: dict[str, list[str]],
    reason: str,
) -> dict[str, Any]:
    line_items: list[dict[str, Any]] = []
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    selected_fact_ids = _select_fact_ids_for_summary_lines(
        extracted_facts=extracted_facts,
        cluster_event_type=cluster_event_type,
    )
    desired_count = min(_SUMMARY_LINE_MAX, max(_SUMMARY_LINE_MIN, len(extracted_facts)))
    for index in range(1, desired_count + 1):
        ids = [
            fact_id for fact_id in selected_fact_ids.get(str(index), []) if fact_id in fact_by_id
        ]
        if not ids and extracted_facts:
            fallback_fact = extracted_facts[min(index - 1, len(extracted_facts) - 1)]
            ids = [str(fallback_fact.get("fact_id"))]
        facts = [fact_by_id[fact_id] for fact_id in ids if fact_id in fact_by_id]
        text = _compose_fallback_line(index=index, facts=facts)
        line_items.append({"line_index": index, "text": text, "fact_ids": ids})
    fact_summary = [item["text"] for item in line_items]
    result = {
        "is_valid_summary": bool(extracted_facts),
        "main_company": main_company,
        "mentioned_peer_companies": [main_company]
        if main_company in target_companies
        else target_companies[:1],
        "cluster_event_type": _normalize_event_type(cluster_event_type),
        "headline": fact_summary[0] if fact_summary else "",
        "one_line_summary": fact_summary[0] if fact_summary else "",
        "fact_summary": fact_summary,
        "summary_lines_with_fact_ids": line_items,
        "main_event": fact_summary[0] if fact_summary else "",
        "fact_basis": _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts),
        "confidence": 0.65 if extracted_facts else 0.0,
        "reason": reason,
    }
    return result


def _fact_id_empty_result(
    *,
    main_company: str,
    target_companies: list[str],
    cluster_event_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "is_valid_summary": False,
        "main_company": main_company,
        "mentioned_peer_companies": [main_company] if main_company in target_companies else [],
        "cluster_event_type": _normalize_event_type(cluster_event_type),
        "headline": "",
        "one_line_summary": "",
        "fact_summary": [],
        "summary_lines_with_fact_ids": [],
        "main_event": "",
        "fact_basis": [],
        "confidence": 0.0,
        "reason": reason,
    }


def _validate_fact_id_summary(
    *,
    result: dict[str, Any],
    extracted_facts: list[dict[str, Any]],
    source_article_ids: list[int],
    main_company: str,
) -> dict[str, Any]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    lines = [
        _clean_summary_line(normalize_korean_spacing(line))
        for line in _normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX]
    ]
    result["fact_summary"] = lines
    line_items = _normalize_summary_line_items(result.get("summary_lines_with_fact_ids"))
    if not line_items:
        line_items = [
            {"line_index": index, "text": line, "fact_ids": []}
            for index, line in enumerate(lines, start=1)
        ]
    line_items = _ensure_fact_summary_lines(line_items, extracted_facts)
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        if 1 <= index <= len(lines):
            item["text"] = lines[index - 1]
    result["summary_lines_with_fact_ids"] = line_items
    result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)

    warnings: list[str] = []
    actions: list[str] = []
    if not (_SUMMARY_LINE_MIN <= len(lines) <= _SUMMARY_LINE_MAX) or any(
        not line for line in lines
    ):
        warnings.append("summary_lines가 3~5개 범위를 벗어남")
    basis_indexes = {
        _safe_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        for item in result.get("fact_basis", [])
    }
    expected_indexes = [
        _safe_int(item.get("line_index"))
        for item in line_items
        if _safe_int(item.get("line_index")) > 0
    ]
    missing_indexes = [index for index in expected_indexes if index not in basis_indexes]
    if missing_indexes:
        warnings.append(f"fact_basis 누락 summary_line_index: {missing_indexes}")
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        fact_ids = _normalize_string_list(item.get("fact_ids"))
        if not fact_ids:
            warnings.append(f"{index}번 문장 fact_id 없음")
        invalid_ids = [fact_id for fact_id in fact_ids if fact_id not in fact_by_id]
        if invalid_ids:
            warnings.append(f"{index}번 문장에 존재하지 않는 fact_id: {invalid_ids}")
    for index, line in enumerate(lines, start=1):
        related_facts = _facts_for_line(index, line_items, fact_by_id)
        related_evidence = " ".join(
            " ".join(
                [
                    str(fact.get("fact") or ""),
                    str(fact.get("evidence_text") or ""),
                    " ".join(_normalize_string_list(fact.get("evidence_texts"))),
                    " ".join(_normalize_string_list(fact.get("numbers_and_dates"))),
                    " ".join(_normalize_string_list(fact.get("numbers"))),
                ]
            )
            for fact in related_facts
        )
        missing_numbers = [
            number
            for number in _number_tokens(line)
            if not _number_token_covered(number, _number_tokens(related_evidence))
        ]
        if missing_numbers:
            warnings.append(f"{index}번 문장 수치 근거 부족: {', '.join(missing_numbers)}")
        attribution_warning = _summary_line_company_attribution_warning(
            line=line,
            evidence=related_evidence,
            main_company=main_company,
        )
        if attribution_warning:
            warnings.append(f"{index}번 문장 {attribution_warning}")
    cleaned_lines = [normalize_korean_spacing(line) for line in lines]
    if cleaned_lines != lines:
        result["fact_summary"] = cleaned_lines
        actions.append("normalize_korean_spacing")
        line_items = _sync_summary_line_item_texts(line_items, cleaned_lines)

    reduced_lines, repetition_actions = normalize_subject_predicate_consistency(
        lines=_normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX],
        line_items=line_items,
        fact_by_id=fact_by_id,
        main_company=main_company,
    )
    if repetition_actions:
        result["fact_summary"] = reduced_lines
        line_items = _sync_summary_line_item_texts(line_items, reduced_lines)
        result["summary_lines_with_fact_ids"] = line_items
        result["fact_basis"] = _fact_basis_from_summary_line_fact_ids(line_items, extracted_facts)
        actions.extend(repetition_actions)

    company_start_count = _company_name_start_count(
        _normalize_string_list(result.get("fact_summary"))[:_SUMMARY_LINE_MAX],
        main_company,
    )
    if company_start_count >= 2:
        warnings.append(f"company_name_start_count={company_start_count}")
        actions.append("company_name_repetition_detected")
    if lines and company_start_count == len(lines):
        warnings.append("summary_lines 모든 문장이 company_name으로 시작함")

    role_counts = _summary_line_role_counts(line_items, fact_by_id)
    cluster_event_type = _normalize_event_type(result.get("cluster_event_type"))
    if cluster_event_type not in {"earnings", "stock_market", "analyst_report"}:
        business_role_count = sum(
            role_counts.get(role, 0)
            for role in ("main_event", "product_definition", "service_function", "application_case")
        )
        numeric_role_count = role_counts.get("numeric_effect", 0) + role_counts.get(
            "market_reaction", 0
        )
        if numeric_role_count >= 2 and business_role_count < 2:
            warnings.append("비실적 이슈 요약이 수치/시장반응 중심으로 치우침")
            actions.append("numeric_heavy_summary_detected")

    bad_korean = [line for line in result["fact_summary"] if _has_bad_korean_join(line)]
    if bad_korean:
        warnings.append("한국어 조사/띄어쓰기 오류가 남아 있음")
    if (
        main_company
        and main_company != _INDUSTRY_TREND_COMPANY_ID
        and result.get("is_valid_summary")
        and not _summary_mentions_company(result, main_company)
    ):
        warnings.append("요약 문장에 main_company alias가 없음")
    if source_article_ids:
        basis_source_ids = _dedupe_ints(
            [
                article_id
                for item in result.get("fact_basis", [])
                for article_id in _as_int_list(item.get("source_article_ids"))
            ]
        )
        if not basis_source_ids:
            warnings.append("fact_basis source_article_ids가 비어 있음")
    result["is_valid_summary"] = bool(result.get("is_valid_summary", True)) and not warnings
    if warnings:
        result["validation_warnings"] = _dedupe_keep_order(
            [*_normalize_string_list(result.get("validation_warnings")), *warnings]
        )
        result["reason"] = _append_reason(result.get("reason"), "; ".join(warnings))
    if actions:
        result["repair_actions"] = _dedupe_keep_order(
            [*_normalize_string_list(result.get("repair_actions")), *actions]
        )
    return result


def _summary_line_company_attribution_warning(
    *,
    line: str,
    evidence: str,
    main_company: str,
) -> str:
    if not main_company:
        return ""
    line_text = str(line or "")
    evidence_text = str(evidence or "")
    if not line_text.strip() or not evidence_text.strip():
        return ""
    main_aliases = _company_aliases_for_detection(main_company)
    if not _text_mentions_any_alias(line_text, main_aliases):
        return ""
    if _text_mentions_any_alias(evidence_text, main_aliases):
        return ""
    other_hits: list[str] = []
    for company_id, aliases in _KNOWN_COMPANY_ALIASES.items():
        if company_id == main_company:
            continue
        if _text_mentions_any_alias(evidence_text, _company_aliases_for_detection(company_id)):
            other_hits.append(company_id)
    if other_hits or _company_like_mentions(evidence_text):
        return "회사 주체 귀속 불일치: related fact evidence가 다른 피어사를 가리킴"
    return ""


def _company_aliases_for_detection(company_id: str) -> list[str]:
    if company_id == _INDUSTRY_TREND_COMPANY_ID:
        return _INDUSTRY_TREND_ALIASES
    aliases = [
        str(alias) for alias in _KNOWN_COMPANY_ALIASES.get(company_id, []) if str(alias).strip()
    ]
    aliases.append(str(company_id or ""))
    return _dedupe_keep_order(aliases)


def _text_mentions_any_alias(text: str, aliases: list[str]) -> bool:
    compact_text = _compact(text)
    for alias in aliases:
        compact_alias = _compact(alias)
        if compact_alias and compact_alias in compact_text:
            return True
    return False


def _company_like_mentions(text: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or ""))
    patterns = (
        r"[가-힣A-Z]+(?:전자|SDS|CNS|하이닉스|클라우드|오토에버|DX|테크윈|엔솔)",
        r"(?:네이버|카카오|포스코|현대|삼성|SK|LG)[가-힣A-Z]*",
    )
    mentions: list[str] = []
    for pattern in patterns:
        mentions.extend(match.group(0) for match in re.finditer(pattern, value))
    return _dedupe_keep_order([mention for mention in mentions if len(mention) >= 2])


def _summary_line_role_counts(
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in line_items:
        for fact_id in _normalize_string_list(item.get("fact_ids")):
            role = str((fact_by_id.get(fact_id) or {}).get("summary_role") or "")
            if role:
                counts[role] = counts.get(role, 0) + 1
    return counts


def _sync_summary_line_item_texts(
    line_items: list[dict[str, Any]],
    lines: list[str],
) -> list[dict[str, Any]]:
    synced: list[dict[str, Any]] = []
    for item in line_items:
        index = _safe_int(item.get("line_index"))
        updated = dict(item)
        if 1 <= index <= len(lines):
            updated["text"] = lines[index - 1]
        synced.append(updated)
    return synced


def normalize_subject_predicate_consistency(
    *,
    lines: list[str],
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
    main_company: str,
) -> tuple[list[str], list[str]]:
    aliases = sorted(_PEER_ALIASES.get(main_company, [main_company]), key=len, reverse=True)
    updated = list(lines)
    actions: list[str] = []
    company_started_indexes: list[int] = []

    for line_index, line in enumerate(lines):
        alias = _starting_company_alias(line, aliases)
        if not alias:
            continue
        company_started_indexes.append(line_index)
        if len(company_started_indexes) == 1:
            continue

        facts = _facts_for_line(line_index + 1, line_items, fact_by_id)
        entity = _primary_entity_for_summary_line(line_index + 1, line_items, fact_by_id)
        replacement = _rewrite_repeated_company_sentence(
            updated[line_index],
            alias=alias,
            entity=entity,
            facts=facts,
            line_index=line_index + 1,
        )
        if replacement != updated[line_index]:
            updated[line_index] = replacement
            actions.append(f"summary_line_{line_index + 1}_subject_replaced")
            actions.append("subject_predicate_consistency_normalized")
            actions.append("company_name_repetition_reduced")

    if len(company_started_indexes) >= 2:
        actions.insert(0, "company_name_repetition_detected")
    if actions:
        updated = [normalize_korean_spacing(line) for line in updated]
    return updated, _dedupe_keep_order(actions)


def _starting_company_alias(line: str, aliases: list[str]) -> str:
    for alias in aliases:
        if not alias:
            continue
        if re.match(rf"^\s*{re.escape(alias)}\s*(은|는|이|가)\s+", str(line or "")):
            return alias
    return ""


def _company_name_start_count(lines: list[str], main_company: str) -> int:
    aliases = sorted(_PEER_ALIASES.get(main_company, [main_company]), key=len, reverse=True)
    return sum(1 for line in lines if _starting_company_alias(line, aliases))


def _primary_entity_for_summary_line(
    line_index: int,
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> str:
    fact_ids: list[str] = []
    for item in line_items:
        if _safe_int(item.get("line_index")) == line_index:
            fact_ids = _normalize_string_list(item.get("fact_ids"))
            break
    for fact_id in fact_ids:
        fact = fact_by_id.get(fact_id)
        if not fact:
            continue
        for entity in _normalize_string_list(fact.get("entities")):
            if len(_compact(entity)) >= 2:
                return entity
    return ""


def _rewrite_repeated_company_sentence(
    line: str,
    *,
    alias: str,
    entity: str,
    facts: list[dict[str, Any]],
    line_index: int,
) -> str:
    match = re.match(
        rf"^\s*{re.escape(alias)}\s*(?:은|는|이|가)\s+(.+)$",
        str(line or "").strip(),
    )
    if not match:
        return line
    rest = match.group(1).strip()
    subject_type = _summary_subject_type(facts)
    if entity and _compact(entity) in _compact(rest[: max(len(entity) + 12, 24)]):
        if subject_type == "product_context":
            return _entity_context_sentence(entity, rest)
        return rest
    if subject_type == "reported_context":
        return _reported_context_sentence(rest)
    if subject_type == "product_context":
        return f"{_product_context_subject(facts)} {rest}"
    if subject_type == "market_context":
        return f"시장 반응은 {rest}"
    if subject_type == "metric_context":
        return _reported_context_sentence(rest)
    if subject_type == "risk_context":
        return f"해당 이슈는 {rest}"
    if line_index == 3:
        return _reported_context_sentence(rest)
    return line


def _summary_subject_type(facts: list[dict[str, Any]]) -> str:
    roles = {str(fact.get("summary_role") or "") for fact in facts}
    fact_types = {str(fact.get("fact_type") or "") for fact in facts}
    if "market_reaction" in roles or "market_fact" in fact_types:
        return "market_context"
    if "numeric_effect" in roles or "numeric_fact" in fact_types:
        return "metric_context"
    if "risk_detail" in roles or "risk_fact" in fact_types:
        return "risk_context"
    if "uncertainty_detail" in roles or "uncertain_fact" in fact_types:
        return "reported_context"
    if "application_case" in roles or "application_fact" in fact_types:
        return "reported_context"
    if (
        "product_definition" in roles
        or "service_function" in roles
        or "platform_definition_fact" in fact_types
    ):
        return "product_context"
    return "company_context"


def _product_context_subject(facts: list[dict[str, Any]]) -> str:
    roles = {str(fact.get("summary_role") or "") for fact in facts}
    if "service_function" in roles:
        return "해당 서비스는"
    return "해당 플랫폼은"


def _entity_context_sentence(entity: str, rest: str) -> str:
    body = _strip_leading_entity_subject(rest, entity)
    body = _extract_reported_clause(body)
    body = body.rstrip(".")
    if not body:
        return rest
    return normalize_korean_spacing(f"{_topic_subject(entity)} {body}.")


def _strip_leading_entity_subject(rest: str, entity: str) -> str:
    match = re.match(
        rf"^\s*{re.escape(entity)}\s*(?:은|는|이|가)\s+(.+)$",
        str(rest or "").strip(),
    )
    return match.group(1).strip() if match else str(rest or "").strip()


def _extract_reported_clause(text: str) -> str:
    value = str(text or "").strip().rstrip(".")
    match = re.match(r"^(.+?고)\s+\S+다$", value)
    if match:
        clause = match.group(1).strip()
        return clause[:-1].strip() if clause.endswith("고") else clause
    return value


def _topic_subject(entity: str) -> str:
    value = str(entity or "").strip()
    if not value:
        return "해당 항목은"
    last = value[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28:
        return f"{value}은"
    return f"{value}는"


def _reported_context_sentence(rest: str) -> str:
    value = _clean_summary_line(rest)
    return value if value else "관련 사실이 확인됐다."


def _clean_summary_line(line: str) -> str:
    value = normalize_korean_spacing(line).strip()
    value = re.sub(r"^기사에서는\s+", "", value)
    value = re.sub(r"\s*사실이\s+확인됐다\.?$", ".", value)
    value = re.sub(r"\s*사실이\s+확인됐습니다\.?$", ".", value)
    return normalize_korean_spacing(value)


def _nominalize_korean_predicate(text: str) -> str:
    value = str(text or "").strip().rstrip(".")
    suffix_map = (
        ("했다", "한"),
        ("한다", "하는"),
        ("됐다", "된"),
        ("된다", "되는"),
        ("있다", "있는"),
        ("이었다", "이었던"),
        ("이다", "인"),
    )
    for suffix, replacement in suffix_map:
        if value.endswith(suffix):
            return value[: -len(suffix)] + replacement
    return value


def _selected_facts_by_line(
    selected_fact_ids: dict[str, list[str]],
    extracted_facts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    fact_by_id = {str(fact.get("fact_id")): fact for fact in extracted_facts}
    result: dict[str, list[dict[str, Any]]] = {}
    for index in range(1, _SUMMARY_LINE_MAX + 1):
        facts = [
            _compact_fact_for_prompt(fact_by_id[fact_id], include_evidence=True)
            for fact_id in selected_fact_ids.get(str(index), [])
            if fact_id in fact_by_id
        ]
        if facts:
            result[str(index)] = facts
    return result


def _additional_facts_for_prompt(
    facts: list[dict[str, Any]],
    *,
    selected_fact_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    selected_ids = {
        fact_id
        for values in selected_fact_ids.values()
        for fact_id in _normalize_string_list(values)
        if fact_id
    }
    selected_texts = [
        _fact_similarity_text(fact)
        for fact in facts
        if str(fact.get("fact_id") or "") in selected_ids and _fact_similarity_text(fact)
    ]
    remaining = [fact for fact in facts if str(fact.get("fact_id") or "") not in selected_ids]
    ranked = sorted(remaining, key=_fact_selection_score, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_texts = list(selected_texts)
    for fact in ranked:
        text = _fact_similarity_text(fact)
        if text and any(_text_similarity(text, existing) >= 0.88 for existing in seen_texts):
            continue
        selected.append(fact)
        if text:
            seen_texts.append(text)
        if len(selected) >= 12:
            break
    return [_compact_fact_for_prompt(fact, include_evidence=False) for fact in selected]


def _compact_facts_for_prompt(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_fact_for_prompt(fact, include_evidence=True) for fact in facts]


def _compact_fact_for_prompt(
    fact: dict[str, Any],
    *,
    include_evidence: bool,
) -> dict[str, Any]:
    item = {
        "fact_id": fact.get("fact_id"),
        "fact_type": fact.get("fact_type"),
        "summary_role": fact.get("summary_role"),
        "normalized_fact": fact.get("normalized_fact"),
        "confidence": fact.get("confidence"),
    }
    if include_evidence:
        item["evidence_text"] = fact.get("evidence_text")
    for key in ("entities", "numbers", "dates"):
        values = fact.get(key, [])
        if values:
            item[key] = values
    return {key: value for key, value in item.items() if value not in (None, "", [])}


def _facts_for_line(
    index: int,
    line_items: list[dict[str, Any]],
    fact_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fact_ids: list[str] = []
    for item in line_items:
        if _safe_int(item.get("line_index")) == index:
            fact_ids = _normalize_string_list(item.get("fact_ids"))
            break
    return [fact_by_id[fact_id] for fact_id in fact_ids if fact_id in fact_by_id]


def _compose_fallback_line(index: int, facts: list[dict[str, Any]]) -> str:
    if not facts:
        return ""
    if len(facts) == 1:
        return _fact_text_for_summary_line(facts[0])
    texts = [_fact_text_for_summary_line(fact).rstrip(".") for fact in facts[:2]]
    if index == 2:
        return normalize_korean_spacing(f"{texts[0]}고, {texts[1]}.")
    return normalize_korean_spacing(f"{texts[0]}; {texts[1]}.")


def _fact_text_for_summary_line(fact: dict[str, Any]) -> str:
    text = str(fact.get("normalized_fact") or fact.get("evidence_text") or "").strip()
    text = normalize_korean_spacing(text)
    if fact.get("fact_type") == "uncertain_fact":
        text = _soften_uncertain_sentence(text)
    if text and not text.endswith((".", "다", "요", "죠")):
        text = f"{text}."
    return text
