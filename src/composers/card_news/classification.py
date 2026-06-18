"""card_news classification — extracted from facade (move-only)."""

from __future__ import annotations

from typing import Any

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
    _ISSUE_CARD_PROMPT,
    _PROMPT_VERSION,
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
    _display_token_pieces,
    _first_amount_like_term,
    _first_list_item,
    _first_non_empty,
    _first_sentence_is_too_thin,
    _first_text,
    _follow_up_action_from_statement,
    _format_articles,
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
from src.config.sectors import match_sectors


def _normalize_sector(value: Any) -> str:
    sector = str(value or "").strip()
    if sector == "deal":
        return "biz_area"
    return sector if sector in _FRONTEND_SECTOR_IDS else "other"


def _infer_event_type(
    summary: dict[str, Any],
    classification: dict[str, Any],
    card_text: str,
) -> str:
    del card_text
    summary_event = _normalize_event_type(summary.get("cluster_event_type"))
    if summary_event not in {"general_update", "unknown"}:
        return summary_event

    classification_event = _normalize_event_type(classification.get("event_type"))
    if classification_event not in {"general_update", "unknown"}:
        return classification_event
    return "general_update"


def _infer_sectors(classification: dict[str, Any], card_text: str) -> list[str]:
    raw_sectors = _list_string(classification.get("sectors"))
    if not raw_sectors and classification.get("sector"):
        raw_sectors = [str(classification.get("sector"))]
    normalized = [_normalize_sector(sector) for sector in raw_sectors]
    normalized = [sector for sector in normalized if sector != "other"]
    matched = [_normalize_sector(sector) for sector in match_sectors(card_text)]
    matched = [sector for sector in matched if sector != "other"]
    candidates = _dedupe_keep_order([*normalized, *matched])
    return _dedupe_keep_order(_filter_false_positive_sectors(candidates, card_text)) or ["other"]


def _filter_false_positive_sectors(sectors: list[str], card_text: str) -> list[str]:
    return [
        sector
        for sector in sectors
        if sector != "other" and _sector_has_direct_evidence(sector, card_text)
    ]


def _normalized_importance_score(
    *,
    classification: dict[str, Any],
    card_text: str,
) -> float:
    score = _optional_float(
        classification.get("importance_score", classification.get("exposure_score"))
    )
    if score is None:
        band = _normalize_exposure_band(
            classification.get("importance", classification.get("exposure_band"))
        )
        score = {"high": 0.75, "medium": 0.6, "low": 0.45}.get(band, 0.5)
    score = max(0.0, min(score, 1.0))
    del card_text
    return round(score, 2)


def _importance_band(score: float | None) -> str:
    value = score if score is not None else 0.0
    if value >= 0.75:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _normalize_exposure_band(value: Any) -> str:
    band = str(value or "").strip()
    return band if band in {"high", "medium", "low"} else "medium"
