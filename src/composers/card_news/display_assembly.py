"""card_news display_assembly — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from src.composers.card_news.classification import (  # noqa: F401
    _filter_false_positive_sectors,
    _importance_band,
    _infer_event_type,
    _infer_sectors,
    _normalize_exposure_band,
    _normalize_sector,
    _normalized_importance_score,
)
from src.composers.card_news.evidence_media import (  # noqa: F401
    _article_image_urls,
    _dedupe_card_fact_basis,
    _default_sources,
    _evidence_chain,
    _fact_basis,
    _image_asset_score,
    _image_urls_from_value,
    _is_image_reference,
    _load_source_articles,
    _media_assets,
    _media_assets_for_cluster,
    _normalize_fact_basis_evidence_type,
    _rich_sources,
    _sector_has_direct_evidence,
    _source_article_ids,
    _source_indexes,
)
from src.composers.card_news.schema import (  # noqa: F401
    _CARD_DETAIL_MAX,
    _CARD_PROMPT_VERSION,
    _DEFAULT_COVER_IMAGE_ALT,
    _DEFAULT_COVER_IMAGE_URL,
    _DISPLAY_ITEM_MAX,
    _DISPLAY_ITEM_MIN,
    _DISPLAY_ITEM_PREFERRED,
    _DISPLAY_TRUNCATED_MARKER_PATTERN,
    _DISPLAY_ZONE,
    _EVENT_TO_FACT_BASIS_TYPE,
    _FACT_BASIS_EVIDENCE_TYPES,
    _FRONTEND_EVENT_TYPES,
    _FRONTEND_SECTOR_IDS,
    _SUMMARY_ACTION_TOKENS,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _attach_card_news_schema_fields,
)
from src.composers.card_news.text_utils0 import (  # noqa: F401
    _accent_color,
    _action_semantic_role_for_index,
    _action_semantic_role_for_linkage_field,
    _article_content_sentences,
    _bounded_detail_items,
    _build_cluster_fetch_ids,
    _card_news_id,
    _card_signals,
    _card_text,
    _card_text_from_card,
    _clean_display_truncated_fragment,
    _clean_issue_term,
    _clean_source_headline,
    _compact_ascii,
    _compact_card_title,
    _compact_repeated_subject_in_phrase,
    _company_surface_replacements,
    _db_record,
    _dedupe_keep_order,
    _detail_line_candidates,
    _detail_line_key,
    _display_company_name,
    _display_token_pieces,
    _first_amount_like_term,
    _first_list_item,
    _first_non_empty,
    _first_sentence_is_too_thin,
    _first_text,
    _follow_up_action_from_statement,
    _format_numeric_text,
    _get_llm,
    _grounding_path_matches,
    _has_final_consonant,
    _has_non_money_quantity,
    _has_numeric_or_period_signal,
    _has_schedule_signal,
    _has_target_capacity_or_schedule,
    _insight_candidates_from_strategy,
    _is_financial_only_title,
    _is_internal_analysis_copy,
    _is_labeled_public_frontend_item,
    _is_market_reaction_summary_line,
    _is_near_duplicate_detail,
    _is_signature_noun_phrase,
    _is_untranslated_english_text,
    _issue_execution_detail_phrase,
    _issue_signature,
    _issue_terms,
    _join_context_terms,
    _json_like_text,
    _labeled_frontend_ready_items,
    _linkage_area_records,
    _linkage_record_terms,
    _linkage_text,
    _list_dicts,
    _list_string,
    _list_value,
    _llm,
    _looks_like_action,
    _looks_like_article_boilerplate,
    _looks_like_non_summary_line,
    _looks_like_sentence_title,
    _low_specificity_display_match_term,
    _metadata,
    _normalize_card_news_statement_style,
    _normalize_display_token,
    _normalize_event_type,
    _normalize_peer_id,
    _normalize_summary_similarity_token,
    _now_iso,
    _optional_float,
    _optional_int,
    _order_articles,
    _parse_datetime_for_display,
    _parse_json,
    _polish_frontend_ready_internal_terms,
    _profile_linkage_level,
    _profile_phrase_from_peer_copy,
    _public_copy_cleanup,
    _public_sentences,
    _published_datetime,
    _set_card_signals,
    _skax_business_phrase_from_implication,
    _split_public_conclusion_evidence,
    _statement_to_check_subject,
    _string_or_none,
    _strip_number_prefix,
    _strip_public_section_prefixes,
    _subject_company_names,
    _surface_alias_pattern,
    _unique_nonempty,
)
from src.composers.card_news.text_utils1 import (  # noqa: F401
    _action_candidates_from_strategy,
    _actionize_detail_lines,
    _bounded_detail_lines,
    _business_area_record_has_issue_overlap,
    _business_context_title,
    _business_focus_from_summary,
    _clean_card_editorial_text,
    _clean_frontend_ready_text,
    _clean_structured_blocks,
    _clean_subject_phrase,
    _cleanup_public_frontend_implication,
    _contains_key_number_text,
    _editorial_candidate_has_minimum_grounding,
    _ensure_card_sentence,
    _fatal_summary_validation_issues,
    _format_public_frontend_item,
    _frontend_implication,
    _frontend_implication_from_result,
    _frontend_ready_display_items,
    _frontend_ready_section_blocks,
    _grounding_tokens,
    _has_specific_skax_linkage,
    _has_token_overlap,
    _implication,
    _implication_from_result,
    _industry_frontend_ready_display_items,
    _is_summary_repeat_insight,
    _linkage_has_concrete_profile_detail,
    _linkage_priority_phrase,
    _literal_summary_lines,
    _missing_fact_basis_line_indexes,
    _normalize_company_surface_names,
    _plain_summary_lines,
    _profile_phrase_from_linkage,
    _profile_reason_from_linkage,
    _profile_specificity_note,
    _published_date,
    _record_specificity_level,
    _signals,
    _skax_business_phrase_from_linkage,
    _skax_reason_from_linkage,
    _skax_specificity_note,
    _split_main_detail_block,
    _structured_blocks_from_labeled_lines,
    _subtitle,
    _summary_candidate_lines,
    _summary_fact_text,
    _summary_focus_tokens,
    _summary_line_count_valid,
    _summary_lines_too_similar,
    _summary_similarity_tokens,
    _trust_score,
    _validation_missing,
    _validation_pass,
    _validation_sc_score,
    _with_particle,
    _with_structured_frontend_blocks,
    mark_near_duplicate_card_candidates,
    peer_company_label,
)

log = logging.getLogger(__name__)


def _card_from_summary(
    *,
    summary: dict[str, Any],
    articles: list[dict[str, Any]],
    company: str,
    cluster_id: int,
    representative_id: int,
    classification: dict[str, Any],
) -> dict[str, Any] | None:
    if not summary.get("is_valid_summary"):
        return None

    effective_company = _first_non_empty(summary.get("main_company"), company)
    summary_lines = _plain_summary_lines(summary, use_llm=True)
    if not summary_lines:
        return None
    if len(summary_lines) < _SUMMARY_LINE_MIN:
        summary_lines = _merge_summary_lines(summary_lines, _article_title_summary_lines(articles))

    title = _first_non_empty(
        summary.get("display_headline"),
        summary.get("headline"),
        summary.get("display_one_line_summary"),
        summary.get("one_line_summary"),
        classification.get("title"),
        articles[0].get("title"),
    )
    title = _compact_card_title(
        title,
        summary=summary,
        classification=classification,
        articles=articles,
    )
    title = _business_context_title(
        title,
        summary=summary,
        event_type=str(classification.get("event_type") or summary.get("cluster_event_type") or ""),
    )
    generated_at = _now_iso()
    published_date = _published_date(articles, generated_at)
    created_at = _published_datetime(articles, generated_at)
    media_assets = _media_assets(articles)

    card = {
        "id": _card_news_id(cluster_id, published_date),
        "company": effective_company,
        "cluster_id": cluster_id,
        "representative_id": representative_id,
        "published_date": published_date,
        "created_at": created_at,
        "title": title[:100],
        "summary_lines": summary_lines,
        "event_type": classification.get("event_type", "tech"),
        "sector": classification.get("sector", "other"),
        "sectors": classification.get("sectors", ["other"]),
        "exposure_score": classification.get("exposure_score", 0.0),
        "exposure_band": classification.get("exposure_band", "low"),
        "signals": classification.get("signals", {}),
        "importance": classification.get("importance", "low"),
        "importance_score": classification.get("importance_score", 0.0),
        "sources": _default_sources(articles),
        "news_summary": summary,
        "image_assets": media_assets,
    }
    _attach_card_news_schema_fields(card)
    return card


def _numbered_summary_lines(value: Any) -> list[str]:
    lines = [str(item).strip() for item in _list_value(value) if str(item).strip()]
    numbered: list[str] = []
    for index, line in enumerate(lines[:_SUMMARY_LINE_MAX], start=1):
        prefix = f"{index}."
        numbered.append(line if line.startswith(prefix) else f"{prefix} {line}")
    return numbered


def _display_summary_lines(
    lines: list[str],
    summary: dict[str, Any],
    *,
    use_llm: bool = False,
) -> list[str]:
    candidates = _summary_candidate_lines(lines, summary)
    if use_llm:
        refined = _llm_display_summary_lines(summary, candidates)
        if _SUMMARY_LINE_MIN <= len(refined) <= _SUMMARY_LINE_MAX:
            return refined

    key_numbers = _key_number_display_map(summary)
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(candidates):
        text = _summary_line_for_display(_strip_number_prefix(line), key_numbers)
        dedupe_key = re.sub(r"\s+", " ", text).casefold()
        if not text or dedupe_key in seen:
            continue
        ranked.append((_summary_line_priority(text, summary), -index, text))
        seen.add(dedupe_key)
    ranked.sort(reverse=True)
    selected: list[str] = []
    skipped: list[str] = []
    selected_roles: set[str] = set()
    selected_keys: set[str] = set()
    for _, _, text in ranked:
        if len(selected) >= _SUMMARY_LINE_MAX:
            break
        role = _summary_line_role(text, summary)
        if role == "market_reaction":
            continue
        if role in selected_roles and role != "context" and len(selected) < _SUMMARY_LINE_MIN:
            skipped.append(text)
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            skipped.append(text)
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        selected.append(text)
        selected_keys.add(key)
        selected_roles.add(role)
    for text in skipped:
        if len(selected) >= _SUMMARY_LINE_MIN:
            break
        key = re.sub(r"\s+", " ", text).casefold()
        if key in selected_keys:
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            continue
        selected.append(text)
        selected_keys.add(key)
    return selected[:_SUMMARY_LINE_MAX] if len(selected) >= _SUMMARY_LINE_MIN else selected


def _llm_display_summary_lines(summary: dict[str, Any], candidates: list[str]) -> list[str]:
    if not os.getenv("OPENAI_API_KEY"):
        return []
    clean_candidates = []
    seen: set[str] = set()
    key_numbers = _key_number_display_map(summary)
    for line in candidates:
        text = _summary_line_for_display(_strip_number_prefix(line), key_numbers)
        key = re.sub(r"\s+", " ", text).casefold()
        if text and key not in seen:
            clean_candidates.append(text)
            seen.add(key)
    if len(clean_candidates) < _SUMMARY_LINE_MIN:
        return []

    context = {
        "headline": summary.get("headline"),
        "main_event": summary.get("main_event"),
        "main_issue": summary.get("main_issue"),
        "one_line_summary": summary.get("one_line_summary"),
        "event_type": summary.get("cluster_event_type") or summary.get("event_type"),
        "candidate_facts": clean_candidates[:20],
    }
    prompt = (
        "You are selecting frontend summary lines for a Korean executive card news item.\n"
        "Choose 3 to 5 lines only from candidate_facts. Do not invent facts.\n"
        "Avoid repeating the same fact in different wording. Prefer lines that together cover "
        "different factual dimensions such as the event, amount/scale, period/schedule, purpose, "
        "execution scope, or source-backed consequence when present.\n"
        "Do not include stock/market reaction unless the event_type itself is stock_market.\n"
        'Return strict JSON only: {"summary_lines": ["..."]}.\n\n'
        f"INPUT:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    try:
        from src.observability import tracing_config

        response = _get_llm().invoke(
            prompt,
            config=tracing_config(
                agent="CardNewsComposer",
                prompt_version="card-news-summary-select-v1.0",
                company=str(summary.get("main_company") or ""),
                cluster_id=_optional_int(summary.get("cluster_id")),
            ),
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json(content)
    except Exception as exc:  # noqa: BLE001 - display fallback should not block card generation.
        log.warning("카드뉴스 표시 요약 LLM 선별 실패 | error=%s", exc)
        return []

    selected: list[str] = []
    candidate_set = {re.sub(r"\s+", " ", item).casefold() for item in clean_candidates}
    for item in _list_string(parsed.get("summary_lines")):
        text = _summary_line_for_display(_strip_number_prefix(item), key_numbers)
        key = re.sub(r"\s+", " ", text).casefold()
        if not text or key not in candidate_set:
            continue
        if any(_summary_lines_too_similar(text, existing) for existing in selected):
            continue
        selected.append(text)
        if len(selected) >= _SUMMARY_LINE_MAX:
            break
    return selected if len(selected) >= _SUMMARY_LINE_MIN else []


def _summary_line_priority(text: str, summary: dict[str, Any]) -> int:
    value = str(text or "")
    score = 0
    event_type = str(
        summary.get("cluster_event_type") or summary.get("event_type") or ""
    ).casefold()
    if _is_market_reaction_summary_line(value) and event_type != "stock_market":
        score -= 100
    shared_focus = _summary_similarity_tokens(value) & _summary_focus_tokens(summary)
    score += min(len(shared_focus), 8) * 9
    if _has_numeric_or_period_signal(value):
        score += 20
    if _has_target_capacity_or_schedule(value):
        score += 45
    if _has_schedule_signal(value) and _has_non_money_quantity(value):
        score += 35
    elif _has_schedule_signal(value):
        score += 15
    if _has_target_capacity_or_schedule(value) and not _has_schedule_signal(value):
        if re.search(r"최종|계약|확정|완료", value):
            score -= 25
    if _contains_key_number_text(value, summary):
        score += 25
    if re.search(r"선정|확정|수주|계약|체결|참여|사업자", value):
        score += 20
    if re.search(r"구축|설립|운영|착공|도입|전환|현대화|협약|계약", value):
        score += 15
    if re.search(r"주가|거래소|거래\s*(중|마쳤)|상승|하락|급등|급락", value):
        score -= 30
    return score


def _summary_line_role(text: str, summary: dict[str, Any]) -> str:
    value = str(text or "")
    event_type = str(
        summary.get("cluster_event_type") or summary.get("event_type") or ""
    ).casefold()
    if _is_market_reaction_summary_line(value) and event_type != "stock_market":
        return "market_reaction"
    if re.search(r"최종\s*선정|민간\s*참여|사업자로\s*선정|사업자에", value):
        return "core_event"
    if re.search(r"협약|주주\s*간|SPC|특수목적법인|출자|설립", value, re.IGNORECASE):
        return "governance"
    if _has_target_capacity_or_schedule(value):
        return "scale_schedule"
    if re.search(r"선정|확정|수주|계약|체결|참여|사업자", value):
        return "core_event"
    if _has_numeric_or_period_signal(value):
        return "scale_schedule"
    return "context"


def _summary_line_for_display(line: str, key_numbers: dict[str, str]) -> str:
    text = _clean_display_truncated_fragment(line)
    if not text:
        return ""
    if text in key_numbers:
        return key_numbers[text]
    metric_match = re.fullmatch(
        r"(?P<label>[가-힣A-Za-z&·/\s]+?)\s+(?P<value>-?\d+(?:\.\d+)?)",
        text,
    )
    if metric_match:
        label = re.sub(r"\s+", " ", metric_match.group("label")).strip()
        value = _format_numeric_text(metric_match.group("value"))
        lookup_key = f"{label} {metric_match.group('value')}"
        if lookup_key in key_numbers:
            return key_numbers[lookup_key]
        if re.search(r"\b(YoY|QoQ)\b|증감|성장률|이익률|마진", label, re.IGNORECASE):
            return f"{label} {value}%"
        return f"{label} {value}"
    return text


def _key_number_display_map(summary: dict[str, Any]) -> dict[str, str]:
    display: dict[str, str] = {}
    for item in _list_dicts(summary.get("key_numbers")):
        label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
        raw_value = str(item.get("value") or "").strip()
        if not label or not raw_value:
            continue
        value = _format_numeric_text(raw_value)
        unit = str(item.get("unit") or "").strip()
        text = f"{label} {value}{unit}" if unit and not value.endswith(unit) else f"{label} {value}"
        display[f"{label} {raw_value}"] = text
        display[f"{label} {value}"] = text
    return display


def _display_sections_from_strategy_result(
    *,
    summary: dict[str, Any],
    analysis: dict[str, Any],
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
    sentence_grounding: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    summary_lines: list[str] | None = None,
) -> list[dict[str, Any]]:
    del sources
    summary_items = _literal_summary_lines(summary, fallback=summary_lines)
    frontend_ready_items = _frontend_ready_display_items(implication)
    insight_items = frontend_ready_items.get("insight") or []
    action_items = frontend_ready_items.get("action") or []
    insight_blocks = frontend_ready_items.get("insight_blocks") or []
    action_blocks = frontend_ready_items.get("action_blocks") or []
    section_metadata: dict[str, Any] = {}
    if not (insight_items or action_items):
        industry_items = _industry_frontend_ready_display_items(implication)
        insight_items = industry_items.get("insight") or []
        action_items = industry_items.get("action") or []
        insight_blocks = industry_items.get("insight_blocks") or []
        action_blocks = industry_items.get("action_blocks") or []
        section_metadata = industry_items.get("metadata") or {}
    return [
        {"type": "summary", "title": "요약", "items": summary_items},
        {
            "type": "insight",
            "title": "시사점",
            "items": insight_items,
            "structured_items": insight_blocks,
            **section_metadata,
        },
        {
            "type": "action",
            "title": "대응방안",
            "items": action_items,
            "structured_items": action_blocks,
            **section_metadata,
        },
    ]


def _display_sections_missing_frontend_ready(display_sections: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for section_type, label in (("insight", "시사점"), ("action", "대응방안")):
        section = next(
            (
                item
                for item in display_sections
                if isinstance(item, dict) and item.get("type") == section_type
            ),
            {},
        )
        items = _list_string(section.get("items") if isinstance(section, dict) else [])
        structured_items = (
            _clean_structured_blocks(section.get("structured_items"))
            if isinstance(section, dict)
            else []
        )
        if not items and not structured_items:
            missing.append(label)
    return missing


def _sync_frontend_implication_from_display_sections(
    frontend: Any,
    *,
    display_sections: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(frontend) if isinstance(frontend, dict) else {}
    sections = {
        str(section.get("type") or ""): _list_string(section.get("items"))
        for section in display_sections
        if isinstance(section, dict)
    }
    structured_sections = {
        str(section.get("type") or ""): _clean_structured_blocks(
            section.get("structured_items")
            or _structured_blocks_from_labeled_lines(section.get("items"))
        )
        for section in display_sections
        if isinstance(section, dict)
    }
    insight_items = sections.get("insight") or []
    action_items = sections.get("action") or []
    insight_blocks = structured_sections.get("insight") or []
    action_blocks = structured_sections.get("action") or []
    payload["key_implications"] = insight_items
    payload["peer_implications"] = insight_items
    payload["response_directions"] = action_items
    payload["suggested_actions"] = action_items
    payload["key_implication_blocks"] = insight_blocks
    payload["key_implication_items"] = insight_blocks
    payload["response_direction_blocks"] = action_blocks
    payload["suggested_action_items"] = action_blocks
    payload["follow_up_questions"] = []
    for section in display_sections:
        if not isinstance(section, dict):
            continue
        for key in ("signal_scope", "display_policy"):
            if section.get(key):
                payload[key] = section.get(key)
    return _cleanup_public_frontend_implication(payload)


def _sync_implication_frontend_from_display_sections(
    implication: Any,
    *,
    frontend: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(implication) if isinstance(implication, dict) else {}
    payload["frontend"] = frontend
    return payload


def _article_title_summary_lines(articles: list[dict[str, Any]]) -> list[str]:
    """Fallback factual summary when IntegrationAgent marks a cluster invalid."""
    out: list[str] = []
    seen: set[str] = set()

    def add_line(value: Any) -> None:
        text = _clean_card_editorial_text(str(value or ""))
        text = re.sub(r"\s+", " ", text).strip(" .")
        if not text or _looks_like_article_boilerplate(text) or _looks_like_non_summary_line(text):
            return
        key = re.sub(r"\W+", "", text).casefold()
        if key in seen:
            return
        out.append(text)
        seen.add(key)

    for article in articles:
        title = str(article.get("title") or "").strip()
        add_line(title)
        content = str(article.get("content") or "")
        for sentence in _article_content_sentences(content):
            add_line(sentence)
            if len(out) >= _SUMMARY_LINE_MIN:
                break
        if len(out) >= _SUMMARY_LINE_MAX:
            break
    return out[:_SUMMARY_LINE_MAX]


def _merge_summary_lines(primary: list[str], fallback: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for line in [*primary, *fallback]:
        text = _clean_card_editorial_text(str(line or "")).strip(" .")
        if _looks_like_non_summary_line(text):
            continue
        key = re.sub(r"\W+", "", text).casefold()
        if not text or key in seen:
            continue
        merged.append(text)
        seen.add(key)
        if len(merged) >= _SUMMARY_LINE_MAX:
            break
    return merged


def _display_subject_from_summary(summary: dict[str, Any]) -> str:
    intelligence = summary.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        products = _list_string(intelligence.get("products_or_services"))
        if products:
            product = re.sub(r"\s+", " ", products[0]).strip(" .")
            amount = _first_amount_like_term(_list_string(intelligence.get("numbers_and_dates")))
            if amount and amount not in product:
                return _clean_subject_phrase(f"{amount} 규모의 {product}", summary)
            return _clean_subject_phrase(product, summary)
    subject = _first_text(
        summary.get("main_issue"),
        summary.get("main_event"),
        summary.get("headline"),
        summary.get("one_line_summary"),
        _first_list_item(summary.get("fact_summary")),
        "현재 이슈",
    )
    subject = re.sub(r"\s+", " ", subject).strip(" .")
    subject = re.sub(r"^(이번|해당)\s*", "", subject)
    return _clean_subject_phrase(subject, summary)


def _select_display_items(
    candidates: list[dict[str, Any]],
    *,
    sentence_grounding: dict[str, Any] | None,
    summary: dict[str, Any] | None = None,
    strategic_root: dict[str, Any] | None = None,
    preferred: int = _DISPLAY_ITEM_PREFERRED,
    max_items: int = _DISPLAY_ITEM_MAX,
    min_items: int = _DISPLAY_ITEM_MIN,
    section_type: str = "",
) -> list[str]:
    del preferred
    ranked = sorted(
        candidates,
        key=lambda item: _optional_int(item.get("priority")) or 0,
        reverse=True,
    )
    selected: list[str] = []
    seen: set[str] = set()
    seen_roles: set[str] = set()
    for candidate in ranked:
        text = _clean_display_section_text(
            candidate.get("text"),
            section_type=section_type,
        )
        if not text:
            continue
        if _display_internal_term_violation(text):
            continue
        semantic_role = str(candidate.get("semantic_role") or "").strip()
        path = str(candidate.get("path") or "")
        if section_type == "insight" and _is_summary_repeat_insight(text, summary or {}):
            is_editorial_peer_meaning = (
                semantic_role == "peer_business_meaning" and path.startswith("card_editorial.")
            )
            if not is_editorial_peer_meaning:
                continue
        if not candidate.get("allow_without_grounding") and _is_internal_analysis_copy(text):
            continue
        if semantic_role and semantic_role in seen_roles:
            continue
        if _section_role_violation(section_type, text):
            continue
        if candidate.get(
            "allow_without_grounding"
        ) and not _editorial_candidate_has_minimum_grounding(
            text,
            summary=summary or {},
            strategic_root=strategic_root or {},
            section_type=section_type,
        ):
            continue
        if not candidate.get("allow_without_grounding") and not _is_grounded_display_candidate(
            text,
            path,
            sentence_grounding or {},
            allow_watch_point=bool(candidate.get("allow_watch_point")),
        ):
            continue
        key = _detail_line_key(text)
        if not key or key in seen or _is_near_duplicate_detail(text, selected):
            continue
        selected.append(text)
        seen.add(key)
        if semantic_role:
            seen_roles.add(semantic_role)
        if len(selected) >= max_items:
            break
    return selected if len(selected) >= min_items else []


def _clean_display_section_text(value: Any, *, section_type: str) -> str:
    text = _clean_card_editorial_text(value)
    text = _fix_display_particle_spacing(text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    if section_type == "insight":
        text = re.sub(
            r"\bSK\s*AX\b[^.。!?！？]*(?:점검|보완|대응|모니터링)[^.。!?！？]*[.。!?！？]?",
            "",
            text,
        )
    return _ensure_card_sentence(text.strip())


def _fix_display_particle_spacing(text: str) -> str:
    def replace_confirmed(match: re.Match[str]) -> str:
        term = match.group(1)
        particle = "이" if _has_final_consonant(term[-1]) else "가"
        return f"{term}{particle} 확인"

    return re.sub(
        r"([가-힣A-Za-z0-9&·+_-]{2,})[이가]\s+확인",
        replace_confirmed,
        str(text or ""),
    )


def _section_role_violation(section_type: str, text: str) -> bool:
    value = str(text or "")
    if section_type == "insight":
        return bool(
            re.search(
                r"SK\s*AX|우리\s*회사|자사|내부적으로|점검해야|구분해야|보완해야|"
                r"대응해야|모니터링해야|확인해야|맞춰야",
                value,
            )
        )
    if section_type == "action":
        return not bool(
            re.search(
                r"SK\s*AX|자사|내부적으로|점검|보완|구분|대응|모니터링|비교",
                value,
            )
        )
    return False


def _display_internal_term_violation(text: str) -> bool:
    return bool(
        re.search(
            r"프로필\s*근거|프로필\s*접점|profile|linkage|현재\s*입력에서|matched_|"
            r"business_novelty_status",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _is_grounded_display_candidate(
    text: str,
    path: str,
    sentence_grounding: dict[str, Any],
    *,
    allow_watch_point: bool = False,
) -> bool:
    del text
    entries = [
        entry for entry in _list_value(sentence_grounding.get("entries")) if isinstance(entry, dict)
    ]
    if not entries:
        return True
    matched = [
        entry for entry in entries if _grounding_path_matches(str(entry.get("path") or ""), path)
    ]
    if not matched:
        return allow_watch_point
    if allow_watch_point:
        return True
    return any(
        str(entry.get("grounding_type") or "") in {"fact", "profile", "fact+profile"}
        and not entry.get("needs_review")
        for entry in matched
    )


def _slides(
    title: str,
    summary_lines: list[str],
    insights: list[str],
    sources: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
    cover_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT
    slides = [
        {
            "order": 1,
            "title": title,
            "body": "\n".join(summary_lines[:_SUMMARY_LINE_MAX]) or None,
            "image_url": cover_image,
            "image_alt": cover_alt,
            "evidence_source_indexes": _source_indexes(sources),
            "layout_type": "summary",
        }
    ]
    if insights:
        slides.append(
            {
                "order": 2,
                "title": "의미 분석",
                "body": "\n".join(insights[:3]),
                "image_url": media_assets[1]["url"] if len(media_assets) > 1 else None,
                "image_alt": media_assets[1]["alt"] if len(media_assets) > 1 else None,
                "evidence_source_indexes": _source_indexes(sources),
                "layout_type": "implication",
            }
        )
    return slides


def _slides_from_display_sections(
    *,
    title: str,
    display_sections: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    media_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not display_sections:
        return []
    source_indexes = _source_indexes(sources)
    cover_image = media_assets[0]["url"] if media_assets else _DEFAULT_COVER_IMAGE_URL
    cover_alt = media_assets[0]["alt"] if media_assets else _DEFAULT_COVER_IMAGE_ALT
    slides: list[dict[str, Any]] = []
    for index, section in enumerate(display_sections, start=1):
        items = _list_string(section.get("items"))
        if not items:
            continue
        section_type = str(section.get("type") or "").strip() or "section"
        section_title = str(section.get("title") or "").strip() or "카드뉴스"
        image_url = (
            cover_image
            if index == 1
            else (media_assets[index - 1]["url"] if len(media_assets) >= index else None)
        )
        image_alt = (
            cover_alt
            if index == 1
            else (media_assets[index - 1]["alt"] if len(media_assets) >= index else None)
        )
        slides.append(
            {
                "order": len(slides) + 1,
                "title": title if section_type == "summary" else section_title,
                "section_title": section_title,
                "body": "\n".join(items[:_DISPLAY_ITEM_MAX]) or None,
                "image_url": image_url,
                "image_alt": image_alt,
                "evidence_source_indexes": source_indexes,
                "layout_type": section_type,
            }
        )
    return slides


def _display_meta(sector: str, background_asset_url: str) -> dict[str, Any]:
    return {
        "home_carousel": True,
        "carousel_order": None,
        "slide_count": None,
        "visual_style": "editorial",
        "background_asset_url": background_asset_url,
        "accent_color": _accent_color(sector),
    }
