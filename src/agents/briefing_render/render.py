"""briefing_render render — extracted from facade (move-only)."""

from __future__ import annotations

import copy
from typing import Any

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
from src.agents.briefing_render._constants import (  # noqa: F401
    _DISPLAY_BODY_MAX,
    _DISPLAY_FLOW_MAX,
    _DISPLAY_REASON_MAX,
    _DISPLAY_TITLE_MAX,
    _MAX_DISPLAY_CARDS,
    _PRIVATE_AI_TOKENS,
    _REPO_ROOT,
    _REUSE_TTL_CURRENT_PERIOD,
    _ROBOT_OPS_TOKENS,
    BriefingType,
)
from src.agents.briefing_render.card_selector import (  # noqa: F401
    _aggregate_text,
    _average_confidence,
)
from src.agents.briefing_render.text_normalizer import (  # noqa: F401
    _ensure_sentence_field,
    _ensure_sentence_fields,
    _ensure_sentence_list,
    _ensure_terminal_period,
    _normalize_visible_sentence_endings,
)
from src.agents.briefing_render.utils0 import (  # noqa: F401
    _action_text,
    _action_use_case,
    _action_values,
    _block_text,
    _briefing_id,
    _briefing_lead,
    _briefing_trends_summary,
    _card_display_title,
    _card_summary,
    _clean_ids,
    _combine_company_signals,
    _compact_flow_step,
    _compact_reasoning_summary,
    _compact_reasoning_trace,
    _company_issue_phrases_from_result,
    _company_signal_map_from_result,
    _competitor_importance_sentence,
    _competitor_move_group_title,
    _competitor_move_summary,
    _competitor_move_title_candidates,
    _competitor_so_what_candidates,
    _cross_card_importance_sentence,
    _display_core_summary,
    _display_core_title,
    _display_date_ko,
    _display_evidence_ids,
    _display_flow_steps,
    _display_sk_ax_view,
    _entry_basis_sort_key,
    _entry_company_labels,
    _entry_evidence_card_ids,
    _entry_has_tokens,
    _evidence_source_names,
    _evidence_texts,
    _first_company_signal,
    _first_distinct_text,
    _flow_item,
    _front_action_title,
    _front_briefing_benchmark,
    _front_briefing_count,
    _front_briefing_label,
    _front_briefing_signal_cards,
    _front_business_signals,
    _front_company_signal_phrase,
    _front_competition_impact_title,
    _front_flow_detail,
    _front_key_number_for_tokens,
    _front_key_numbers,
    _front_limited_step_items,
    _front_polite_sentence,
    _front_signal_description,
    _front_signal_summary,
    _front_signal_title,
    _front_step_item,
    _front_strategy_implication_title,
    _frontend_daily_title,
    _frontend_weekly_title,
    _generic_distinct_sentence,
    _grounded_why_important,
    _interpretation_reasoning_summary,
    _is_profile_linkage_display_text,
    _is_program_artifact_action,
    _join_display_sentences,
    _key_change_source_entries,
    _looks_like_mostly_english,
    _market_description_is_card_listing,
    _market_signal_reason_summary,
    _normalize_duplicate_company_prefix,
    _normalize_korean_plain_ending,
    _parse_date,
    _period_label,
    _period_perspective_tail,
    _primary_source,
    _provenance_base,
    _public_sk_ax_view_from_result,
    _reasoning_decision_rule,
    _reasoning_output_sentence,
    _reasoning_question,
    _reasoning_source_bundle_sentence,
    _reasoning_stage_name,
    _reasoning_stage_source_field,
    _repeated_signal_candidate,
    _sanitize_internal_display_terms,
    _saved_report_is_fresh,
    _source_name,
    _source_published_at,
    _split_trend_cards,
    _strip_leading_company_prefix,
    _strip_period_scope_prefix,
    _strip_visual_ellipsis_from_payload,
    _trend_reason,
    _trend_source_label,
    _trend_title,
    _unique_texts,
)
from src.agents.briefing_render.utils1 import (  # noqa: F401
    _company_issue_clauses,
    _company_issue_sentence_from_result,
    _company_issue_sentence_text,
    _competitor_move_summary_candidates,
    _evidence_summary_from_basis,
    _front_action_description,
    _front_briefing_flow_steps,
    _front_briefing_meaning,
    _front_briefing_report_payload,
    _front_card_news_item,
    _front_card_source_payload,
    _front_card_summary_lines,
    _front_entry_by_tokens,
    _front_evaluation_shift_title,
    _front_evidence_entries,
    _front_generic_observation_items,
    _front_issue_name,
    _front_join_company_signal_clauses,
    _front_peer_move_sentence,
    _front_primary_signal_names,
    _front_private_ai_market_description,
    _front_private_ai_skax_description,
    _front_robot_market_description,
    _front_robot_skax_description,
    _front_signal_clause,
    _front_signal_name,
    _front_strategy_implication_description,
    _frontend_briefings_payload,
    _frontend_section_item,
    _frontend_snapshot,
    _grounded_interpretation_description,
    _has_token_entry,
    _hidden_details,
    _interpretation_flow_payload,
    _market_signal_summary_candidates,
    _market_signal_title_candidates,
    _needs_more_grounding,
    _normalize_briefing_sections,
    _normalize_briefing_trends,
    _normalize_flow_step,
    _period_scope_phrase,
    _public_selected_cards,
    _readable_briefing_phrase,
    _sanitize_internal_display_terms_from_payload,
    _sanitize_visible_copy_text,
    _section_bullets_from_basis,
    _select_distinct_sentence,
    _sk_ax_view_payload,
    _so_what_candidates,
    _trend_item_from_card,
    _visible_company_count,
)
from src.agents.briefing_render.utils2 import (  # noqa: F401
    _apply_period_perspective_to_report,
    _attach_interpretation_reasoning_trace,
    _briefing_contract_payload,
    _build_report,
    _competitor_move_flow_summary,
    _core_change_insight_items,
    _empty_report,
    _front_competition_impact_description,
    _front_evaluation_shift_description,
    _front_generic_sk_ax_items,
    _front_interpretation_step_items,
    _front_observation_description,
    _front_observation_title,
    _front_reference_signal_summary,
    _front_reference_skax_description,
    _grounded_front_interpretation_flow,
    _grounded_front_sk_ax_view,
    _grounded_key_change_description,
    _interpretation_step_reasoning_trace,
    _key_change_cards_payload,
    _period_competitor_move_description,
    _period_key_change_description,
    _period_market_signal_description,
    _period_signal_names,
    _period_signal_phrase,
    _public_interpretation_flow_from_result,
    _reasoning_entry_evidence_text,
    _reasoning_intermediate_artifacts,
    _reasoning_source_inputs,
    _reasoning_stage_output,
    _refresh_contract_payload,
    _repair_interpretation_flow_steps,
    _repair_public_display_items,
    _representative_entries_for_period,
)


def _refresh_front_briefing_report(
    *,
    report: dict[str, Any],
    briefing_type: BriefingType,
    period: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(report)
    grounded_key_changes = _grounded_front_key_change_cards(updated)
    if grounded_key_changes:
        updated["key_change_cards"] = grounded_key_changes
        updated["core_change"] = {"items": copy.deepcopy(grounded_key_changes)}
    grounded_lead = _grounded_front_briefing_lead(updated)
    if grounded_lead:
        updated["briefing_lead"] = grounded_lead
        updated["executive_summary"] = grounded_lead
    briefing_report = _front_briefing_report_payload(
        result=updated,
        briefing_type=briefing_type,
        period=period,
        selected_cards=selected_cards,
    )
    updated["briefingReport"] = briefing_report
    updated["flowSteps"] = briefing_report["flowSteps"]
    return updated


def _display_market_reading(
    briefing_basis: dict[str, Any],
    default_evidence: list[Any],
) -> list[dict[str, Any]]:
    evidence_ids = [str(item) for item in default_evidence if str(item).strip()]
    blocks = [
        ("common_pattern", "시장 변화"),
        ("comparison_point", "평가 기준"),
        ("hidden_conclusion", "경쟁 구도"),
    ]
    items = []
    for key, label in blocks:
        block = briefing_basis.get(key)
        if not isinstance(block, dict):
            continue
        finding = _clip_text(str(block.get("finding") or "").strip(), max_chars=_DISPLAY_TITLE_MAX)
        if not finding:
            continue
        items.append(
            (
                label,
                finding,
                _brief_sentences(
                    block.get("rationale"), max_sentences=2, max_chars=_DISPLAY_REASON_MAX
                ),
            )
        )
    return [
        {
            "seq": index,
            "label": label,
            "title": title,
            "description": description,
            "evidence_card_ids": evidence_ids,
        }
        for index, (label, title, description) in enumerate(items, 1)
    ][:_MAX_MARKET_ITEMS]


def _market_reading_payload(
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    default_evidence = _json_list(
        _nested_get(briefing_basis, "provenance", "source_card_ids")
    ) or _json_list(briefing_basis.get("sources_used"))
    display_items = _display_market_reading(briefing_basis, default_evidence)
    if display_items:
        return display_items

    blocks = [
        ("common_pattern", "시장 변화"),
        ("comparison_point", "평가 기준"),
        ("hidden_conclusion", "경쟁 구도"),
    ]
    items = []
    for index, (key, label) in enumerate(blocks, 1):
        block = briefing_basis.get(key)
        if not isinstance(block, dict):
            continue
        finding = str(block.get("finding") or "").strip()
        if not finding:
            continue
        items.append(
            {
                "seq": index,
                "label": label,
                "title": _brief_sentence(finding),
                "description": _brief_sentence(block.get("rationale")),
                "evidence_card_ids": block.get("evidence_card_ids") or [],
            }
        )
    return items[:_MAX_MARKET_ITEMS]


def _grounded_front_key_change_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _front_evidence_entries(result)
    if not entries:
        return []
    evidence_ids = _front_evidence_card_ids(entries)
    market_title = _front_market_change_title(entries)
    competitor_title = _front_competitor_move_title(entries)
    return [
        {
            "seq": 1,
            "display_label": "시장 신호",
            "insight_type": "market_signal",
            "title": market_title,
            "description": _front_market_change_description(result, entries),
            "why_important": _front_market_change_importance(entries),
            "evidence_card_ids": evidence_ids,
        },
        {
            "seq": 2,
            "display_label": "경쟁사 움직임",
            "insight_type": "competitor_move",
            "title": competitor_title,
            "description": _front_competitor_move_description(entries),
            "why_important": _front_competitor_move_importance(entries),
            "evidence_card_ids": evidence_ids,
        },
    ]


def _grounded_front_market_reading(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _front_evidence_entries(result)
    if not entries:
        return []
    market_items = [
        {
            "seq": 1,
            "title": _front_market_overview_title(entries),
            "description": _front_market_overview_description(result, entries),
            "evidence_card_ids": _front_evidence_card_ids(entries),
        }
    ]
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    if private_entry:
        market_items.append(
            {
                "seq": len(market_items) + 1,
                "title": _front_private_ai_market_title(private_entry),
                "description": _front_private_ai_market_description(private_entry),
                "evidence_card_ids": _entry_evidence_card_ids(private_entry),
            }
        )
    if robot_entry:
        market_items.append(
            {
                "seq": len(market_items) + 1,
                "title": _front_robot_market_title(robot_entry),
                "description": _front_robot_market_description(robot_entry),
                "evidence_card_ids": _entry_evidence_card_ids(robot_entry),
            }
        )
    market_items.extend(_front_generic_market_items(entries, start_seq=len(market_items) + 1))
    return market_items[:_MAX_MARKET_ITEMS]


def _grounded_front_briefing_lead(result: dict[str, Any]) -> str:
    entries = _front_evidence_entries(result)
    if not entries:
        return ""
    clauses = _front_join_company_signal_clauses(entries)
    summary = _front_primary_signal_summary(entries)
    return f"오늘 수집된 경쟁사 신호에서는 {summary} {clauses}".strip()


def _front_market_change_title(entries: list[dict[str, Any]]) -> str:
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    if robot_entry and private_entry and robot_entry is not private_entry:
        return "AX 경쟁의 초점이 운영 인프라와 데이터 통제 중심으로 이동하고 있습니다."
    issue_names = _front_issue_names(entries)
    if issue_names:
        joined = _join_korean(issue_names[:2])
        return f"{joined}{_subject_particle(joined)} 이번 기간 판단 근거로 확인됩니다."
    market_signal = _first_text(*[entry.get("market_signal") for entry in entries])
    return _brief_sentence(
        market_signal or "기간 내 수요 변화가 운영 기반 중심으로 구체화되고 있습니다.",
        max_chars=_DISPLAY_BODY_MAX,
    )


def _front_market_overview_title(entries: list[dict[str, Any]]) -> str:
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    if robot_entry and private_entry and robot_entry is not private_entry:
        return "자동화와 AI 수요가 모두 도입 이후 운영 문제로 모이고 있습니다."
    if private_entry:
        return "데이터 통제가 AX 수요의 핵심 조건으로 부상하고 있습니다."
    if robot_entry:
        return "운영 데이터가 제조 AX 판단 기준으로 부각되고 있습니다."
    return _front_market_change_title(entries)


def _front_market_overview_description(
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> str:
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    clauses = _dedupe_keep_order(
        [
            _front_signal_clause(robot_entry, tokens=_ROBOT_OPS_TOKENS) if robot_entry else "",
            _front_signal_clause(private_entry, tokens=_PRIVATE_AI_TOKENS) if private_entry else "",
        ]
    )
    if len(clauses) >= 2:
        joined = _join_korean(clauses[:2])
        return (
            f"{_period_scope_phrase(result)}에서 {joined}{_subject_particle(joined)} "
            "함께 확인됩니다. 전자는 현장 데이터와 운영 소프트웨어를 어떻게 굴릴지의 "
            "문제이고, 후자는 보안과 데이터 통제를 어떤 구조로 책임질지의 문제입니다. "
            "두 신호가 함께 나온다는 것은 시장이 도입할 기술보다 도입 이후 운영·통제 "
            "방식을 더 구체적으로 묻기 시작했다는 뜻입니다."
        )
    if clauses:
        return (
            f"{_period_scope_phrase(result)}에서 {clauses[0]}{_subject_particle(clauses[0])} "
            "확인됩니다. 이 신호는 시장 해석에서 기술 자체보다 도입 이후의 운영 책임, "
            "통제 방식, 성과 검증 가능성을 함께 봐야 함을 보여줍니다."
        )
    return _front_market_change_description(result, entries)


def _front_primary_signal_summary(entries: list[dict[str, Any]]) -> str:
    signals = _front_issue_names(entries) or _front_primary_signal_names(entries)
    if len(signals) >= 2:
        joined = _join_korean(signals[:2])
        return f"{joined}{_subject_particle(joined)} 함께 확인됩니다."
    if signals:
        return f"{signals[0]} 신호가 확인됩니다."
    return "기간 내 주요 변화 신호가 확인됩니다."


def _front_market_change_description(
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> str:
    issue_names = _front_issue_names(entries)
    if issue_names:
        joined = _join_korean(issue_names[:3])
        return (
            f"{_period_scope_phrase(result)}에서 {joined}{_subject_particle(joined)} 확인됩니다. "
            "이 근거를 함께 보면 해당 기간 고객 수요, 경쟁 방식, 평가 기준 중 "
            "무엇이 실제 기사 단위에서 움직였는지 구분할 수 있습니다."
        )
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    clauses = _dedupe_keep_order(
        [
            _front_signal_clause(robot_entry, tokens=_ROBOT_OPS_TOKENS) if robot_entry else "",
            _front_signal_clause(private_entry, tokens=_PRIVATE_AI_TOKENS) if private_entry else "",
        ]
    )
    if clauses:
        joined = _join_korean(clauses[:2])
        return (
            f"{_period_scope_phrase(result)}에서 {joined}{_subject_particle(joined)} "
            "함께 확인됩니다. 이는 시장의 관심이 단일 기술 도입보다 운영 기반, "
            "데이터 통제, 성과 검증 가능성으로 이동하고 있음을 보여줍니다."
        )
    return _brief_sentence(
        _block_text(result.get("common_pattern"), "rationale"),
        max_chars=_DISPLAY_BODY_MAX,
    )


def _front_generic_market_items(
    entries: list[dict[str, Any]],
    *,
    start_seq: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used_titles: set[str] = set()
    for entry in entries:
        title = _front_generic_market_title(entry)
        if not title or title in used_titles:
            continue
        used_titles.add(title)
        items.append(
            {
                "seq": start_seq + len(items),
                "title": title,
                "description": _front_generic_market_description(entry),
                "evidence_card_ids": _entry_evidence_card_ids(entry),
            }
        )
        if len(items) >= _MAX_MARKET_ITEMS:
            break
    return items


def _front_generic_market_title(entry: dict[str, Any]) -> str:
    signal = _front_signal_name(entry)
    if signal:
        return f"{signal}{_subject_particle(signal)} 시장 판단 근거로 부각됩니다."
    return _brief_sentence(
        _first_text(entry.get("market_signal"), entry.get("main_issue")),
        max_chars=_DISPLAY_TITLE_MAX,
    )


def _front_generic_market_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry)
    impact = _brief_sentence(
        _first_text(entry.get("impact_reason"), entry.get("analysis_summary")),
        max_chars=_DISPLAY_BODY_MAX,
    )
    first = (
        f"{company}에서 {signal}{_subject_particle(signal)} 확인됩니다."
        if company and signal
        else impact
    )
    second = f"{impact} " if impact and impact not in first else ""
    return (
        f"{first} {second}"
        "이 근거는 해당 기간 시장 해석에서 고객 수요, 경쟁 방식, "
        "평가 기준 중 무엇이 바뀌고 있는지 확인하는 기준이 됩니다."
    ).strip()


def _front_market_change_importance(entries: list[dict[str, Any]]) -> str:
    return (
        "고객이 기술 보유 여부보다 운영 성과와 리스크 감소 근거를 보게 되므로, "
        "제안에서는 실행 범위와 성과 검증 기준을 먼저 보여줘야 합니다."
    )


def _front_competitor_move_title(entries: list[dict[str, Any]]) -> str:
    companies = _entry_company_labels(entries)
    if len(entries) >= 2:
        if companies:
            joined = _join_korean(companies[:3])
            return f"{joined}의 경쟁 메시지가 서로 다른 실행 축으로 갈라지고 있습니다."
        return "경쟁사별 사업 메시지가 고객 적용 범위와 구축 방식으로 나뉘고 있습니다."
    return "경쟁사는 감지된 수요 변화에 맞춰 사업 메시지를 조정하고 있습니다."


def _front_competitor_move_description(entries: list[dict[str, Any]]) -> str:
    clauses = []
    for entry in entries[:3]:
        peer_sentence = _front_peer_move_sentence(entry)
        if peer_sentence:
            clauses.append(peer_sentence)
    if clauses:
        joined = " ".join(clauses[:2])
        return (
            f"{joined} 따라서 단일 기술 발표보다 경쟁사별 고객 적용 범위, "
            "구축 방식, 성과 근거의 차이를 비교해야 합니다."
        )
    return "경쟁사의 사업·기술 메시지 변화가 확인됩니다."


def _front_competitor_move_importance(entries: list[dict[str, Any]]) -> str:
    return (
        "따라서 SK AX는 경쟁사 메시지가 실제 수주, 고객 사례, 운영 성과 지표로 "
        "이어지는지 확인하면서 고객군, 오퍼링, 자원 배분 우선순위를 조정해야 합니다."
    )


def _front_private_ai_market_title(entry: dict[str, Any]) -> str:
    return "프라이빗 AI 수요는 데이터 통제 기준을 끌어올리고 있습니다."


def _front_robot_market_title(entry: dict[str, Any]) -> str:
    return "로봇 운영 데이터는 제조 SW 인프라의 핵심 판단 근거가 됩니다."


def _front_issue_names(entries: list[dict[str, Any]]) -> list[str]:
    names = []
    for entry in entries[:3]:
        name = _front_issue_name(entry)
        if name:
            names.append(name)
    return _dedupe_keep_order(names)


def _repair_market_reading_items(
    items: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for item in items:
        current = copy.deepcopy(item)
        seq = _optional_int(current.get("seq")) or len(repaired) + 1
        title = str(current.get("title") or "")
        description = str(current.get("description") or "")
        grounded = _grounded_market_description(title, result, seq=seq)
        if grounded:
            current["description"] = grounded
        elif (
            not description
            or _is_too_similar(description, [title], threshold=0.7)
            or _is_too_similar(
                description,
                [str(item.get("description") or "") for item in repaired],
            )
            or _market_description_is_card_listing(description)
            or _description_needs_detail(description)
        ):
            current["description"] = grounded or description
        repaired.append(current)
    return repaired


def _grounded_market_description(title: str, result: dict[str, Any], *, seq: int) -> str:
    all_basis = _detailed_basis_sentence_from_result(result)
    basis = (
        all_basis if seq == 1 else _detailed_basis_sentence_from_result(result, focus_text=title)
    )
    normalized = str(title or "")
    company_signals = _company_signal_map_from_result(result)
    robot_signal = _first_company_signal(
        company_signals,
        ("로봇", "스마트팩토리", "SW", "자동화"),
    )
    private_ai_signal = _first_company_signal(
        company_signals,
        ("프라이빗", "AI", "클라우드", "통제"),
    )
    scope = _period_scope_phrase(result)
    if seq == 1 and (robot_signal or private_ai_signal):
        signals = _join_korean([text for text in (robot_signal, private_ai_signal) if text])
        return _join_display_sentences(
            f"{scope}에서 {signals}{_subject_particle(signals)} 함께 확인됩니다.",
            (
                "이는 시장 해석의 초점이 개별 기업 뉴스가 아니라 자동화와 AX 수요가 "
                "운영 데이터, 현장 소프트웨어, 통합 운영 역량으로 넓어지는 흐름에 "
                "있다는 뜻입니다."
            ),
        )
    if seq == 2 and private_ai_signal:
        return _join_display_sentences(
            f"{private_ai_signal}{_subject_particle(private_ai_signal)} 확인됩니다.",
            (
                "이 신호는 고객이 AI 기능 자체보다 데이터가 어디에 머무르고, "
                "누가 운영 책임을 지며, 어떤 거버넌스로 통제되는지를 함께 "
                "평가한다는 뜻입니다."
            ),
        )
    if seq == 3 and robot_signal:
        return _join_display_sentences(
            f"{robot_signal}{_subject_particle(robot_signal)} 확인됩니다.",
            (
                "이 신호는 제조 AX 경쟁에서 로봇 도입 자체보다 로봇 데이터 관리, "
                "현장 소프트웨어 연동, 운영 안정화 역량이 더 중요한 판단 근거가 "
                "되고 있음을 보여줍니다."
            ),
        )
    if any(token in normalized for token in ("평가", "운영 성과", "데이터 통제", "리스크")):
        return _join_display_sentences(
            private_ai_signal or basis or all_basis,
            (
                "이 때문에 고객의 판단 기준은 기술 보유 여부보다 운영 가능성, "
                "통제 책임, 성과 검증 근거로 이동하고 있습니다."
            ),
        )
    if any(token in normalized for token in ("프라이빗", "보안", "AI")):
        return _join_display_sentences(
            private_ai_signal or basis or all_basis,
            (
                "보안과 데이터 통제 요구가 커질수록 고객은 범용 기능보다 "
                "자체 구축 방식과 운영 거버넌스를 함께 확인하려 합니다."
            ),
        )
    if any(token in normalized for token in ("로봇", "운영", "SW", "인프라", "데이터")):
        return _join_display_sentences(
            robot_signal or basis or all_basis,
            (
                "이는 자동화 경쟁의 초점이 장비 도입 자체에서 운영 데이터, "
                "현장 소프트웨어, 통합 운영 역량으로 넓어지고 있음을 의미합니다."
            ),
        )
    if any(token in normalized for token in ("패키지", "성장 논리", "시장 메시지")):
        return _join_display_sentences(
            _company_issue_sentence_from_result(result) or basis or all_basis,
            (
                "따라서 시장 메시지는 단일 기능 설명보다 고객이 바로 이해할 수 있는 "
                "운영 패키지와 성과 논리 중심으로 재구성되고 있습니다."
            ),
        )
    return _join_display_sentences(
        all_basis,
        "이 신호는 기간 내 카드들을 묶어 볼 때 시장의 우선순위가 바뀌고 있음을 보여줍니다.",
    )
