"""briefing_render _constants — extracted from facade (move-only)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal

from src.agents.briefing.basis_builder import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _BRIEFING_SYNTHESIS_PROMPT_VERSION,
    _LLM_MODEL,
    _PROMPT_VERSION,
    _action_details_from_analysis_packages,
    _analysis_basis_block,
    _analysis_package_entries,
    _and_particle,
    _basis_evidence,
    _brief_sentence,
    _brief_sentences,
    _briefing_action_reason,
    _briefing_basis_from_analysis_packages,
    _briefing_clause,
    _briefing_customer_scope,
    _briefing_decision_focus,
    _briefing_synthesis_quality_issues,
    _briefing_synthesis_revision_prompt,
    _combine_blocks,
    _comparison_finding,
    _dedupe_action_pairs,
    _display_sk_ax_title,
    _executive_action_details_from_entries,
    _front_evidence_card_ids,
    _get_llm,
    _invalid_synthesis_card_ids,
    _join_korean,
    _limit_sentences,
    _llm,
    _merge_briefing_basis_synthesis,
    _normalize_synthesis_action_details,
    _normalize_synthesis_block,
    _parse_json_object,
    _recommended_action_pairs,
    _refine_briefing_basis_with_llm,
    _sanitize_synthesis_list,
    _valid_card_ids,
)
from src.agents.briefing.data_layer import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _DEFAULT_LIMIT,
    _DEFAULT_MOCK_PATH,
    _MAX_LIMIT,
    _SECTOR_FILTER_FETCH_MULTIPLIER,
    _analysis_from_integrated_issue,
    _analysis_package_from_integrated_issue_row,
    _append_in_filter,
    _append_integrated_issue_card_filter,
    _append_integrated_issue_peer_filter,
    _append_integrated_issue_sector_filter,
    _card_from_integrated_issue_row,
    _classification_from_integrated_issue_row,
    _clip_text,
    _consolidated_facts_from_period_evidence,
    _dedupe_keep_order,
    _fact_basis_from_period_evidence,
    _fact_summary_from_issue_brief,
    _fetch_mock_period_cards,
    _fetch_period_analysis_units,
    _fetch_period_cards,
    _fetch_period_integrated_issue_rows,
    _filter_by_sectors,
    _first_source_published_at,
    _implication_from_integrated_issue,
    _integrated_issue_from_period_row,
    _is_missing_integrated_issue_storage,
    _load_mock_items,
    _normalize_card_row,
    _normalize_mock_item,
)
from src.agents.briefing.display_copy import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _DISPLAY_COPY_PROMPT_VERSION,
    _MAX_MARKET_ITEMS,
    _MAX_SKAX_ITEMS,
    _analysis_packages_from_result,
    _basis_signal_phrases_from_result,
    _basis_signal_sentence_from_result,
    _brief_noun_phrase,
    _business_signal_phrase,
    _compact_visible_item,
    _company_detail_phrases_from_result,
    _description_needs_detail,
    _detailed_basis_sentence_from_result,
    _display_card_signal_index,
    _display_copy_quality_issues,
    _display_copy_visible_items,
    _display_copy_visible_texts,
    _display_payload_structure,
    _filter_focus_phrases,
    _find_display_item_source,
    _find_key_change_by_type,
    _format_key_number_context,
    _grounded_sk_ax_description,
    _is_too_similar,
    _is_vague_display_text,
    _key_change_card_coverage_issues,
    _key_number_context_phrase,
    _looks_like_key_number,
    _mentions_any_source_company,
    _merge_flow,
    _merge_key_change_cards,
    _merge_market_reading,
    _merge_sk_ax_view,
    _merged_display_list,
    _normalize_similarity_text,
    _period_from_report,
    _recommended_action_sentences_from_result,
    _repair_sk_ax_view_descriptions,
    _sanitize_flow_step_items,
    _shares_keyword,
    _sk_ax_description_from_title,
    _source_company_names,
    _strip_terminal_punctuation,
    _subject_particle,
    _update_text_field,
)
from src.agents.briefing.prompts import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    _briefing_basis_synthesis_view,
    _briefing_contract_schema_hint,
    _briefing_synthesis_context,
    _briefing_synthesis_system_prompt,
    _briefing_synthesis_user_prompt,
    _display_copy_revision_prompt,
    _display_copy_schema_hint,
    _display_copy_system_prompt,
    _display_copy_user_prompt,
)
from src.agents.briefing.support import (  # noqa: F401  — 분리 모듈 re-export (호환 유지)
    KST,
    _analysis_package,
    _analysis_package_from_sources,
    _compact_analysis_package,
    _compact_analysis_unit_for_display,
    _company_label,
    _first_from_list,
    _first_int,
    _first_text,
    _int_list,
    _iso_or_none,
    _json_dict,
    _json_list,
    _nested_get,
    _optional_int,
    _parse_datetime,
    _safe_float,
    _str_values,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


BriefingType = Literal["daily", "weekly", "monthly"]


_MAX_DISPLAY_CARDS = 3


_DISPLAY_TITLE_MAX = 260


_DISPLAY_BODY_MAX = 640


_DISPLAY_REASON_MAX = 520


_DISPLAY_FLOW_MAX = 720


_REUSE_TTL_CURRENT_PERIOD = timedelta(minutes=30)


_ROBOT_OPS_TOKENS = ("로봇", "스마트팩토리", "피지컬")


_PRIVATE_AI_TOKENS = ("프라이빗", "데이터 통제", "거버넌스", "자체 AI")
