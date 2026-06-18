"""briefing_render utils2 — extracted from facade (move-only)."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, cast

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
    _dedupe_cards_for_prompt,
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
    _front_briefing_lead_prefix,
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


def _apply_period_perspective_to_report(report: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(report)
    if _nested_get(updated, "provenance", "display_copy_prompt_version"):
        return updated
    entries = _front_evidence_entries(updated)
    for items in (
        _json_list(updated.get("key_change_cards")),
        _json_list(_json_dict(updated.get("core_change")).get("items")),
    ):
        for item in items:
            if not isinstance(item, dict):
                continue
            insight_type = str(item.get("insight_type") or "")
            period_description = _period_key_change_description(updated, insight_type, entries)
            if period_description:
                item["description"] = period_description
                continue
            tail = _period_perspective_tail(updated, insight_type)
            description = str(item.get("description") or "").strip()
            if tail and description and tail not in description:
                item["description"] = _brief_sentences(
                    _join_display_sentences(description, tail),
                    max_sentences=4,
                    max_chars=_DISPLAY_BODY_MAX,
                )
    return updated


def _refresh_contract_payload(
    report: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    period = _period_from_report(report)
    if not period:
        return report
    briefing_type = cast(BriefingType, report.get("briefing_type") or "daily")
    created_at = _first_text(report.get("created_at"), datetime.now(KST).isoformat())
    contract_payload = _briefing_contract_payload(
        report_id=str(report.get("id") or ""),
        briefing_type=briefing_type,
        period=period,
        report_title=str(report.get("title") or ""),
        executive_summary=_first_text(report.get("briefing_lead"), report.get("executive_summary")),
        selected_cards=selected_cards,
        briefing_basis=_json_dict(report.get("briefing_basis")),
        created_at=created_at,
    )
    updated = copy.deepcopy(report)
    updated.update(contract_payload)
    return updated


def _briefing_contract_payload(
    *,
    report_id: str,
    briefing_type: BriefingType,
    period: dict[str, Any],
    report_title: str,
    executive_summary: str,
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    source_card_ids = [str(card.get("id")) for card in selected_cards if card.get("id")]
    immediate_cards, watch_cards = _split_trend_cards(selected_cards)
    immediate_trends = _normalize_briefing_trends(
        briefing_basis.get("immediate_trends"),
        selected_cards=selected_cards,
        fallback_cards=immediate_cards,
        role="immediate",
    )
    watch_trends = _normalize_briefing_trends(
        briefing_basis.get("watch_trends"),
        selected_cards=selected_cards,
        fallback_cards=watch_cards,
        role="watch",
    )
    sections = _normalize_briefing_sections(
        briefing_basis.get("sections"),
        executive_summary=executive_summary,
        source_card_ids=source_card_ids,
        immediate_trends=immediate_trends,
        watch_trends=watch_trends,
        briefing_basis=briefing_basis,
    )
    evidence_summary = _evidence_summary_from_basis(
        briefing_basis,
        selected_cards=selected_cards,
        source_card_ids=source_card_ids,
    )
    frontend_payload = _frontend_briefings_payload(
        report_id=report_id,
        briefing_type=briefing_type,
        period=period,
        report_title=report_title,
        executive_summary=executive_summary,
        selected_cards=selected_cards,
        immediate_trends=immediate_trends,
        watch_trends=watch_trends,
        evidence_summary=evidence_summary,
    )
    return {
        "source_card_ids": source_card_ids,
        "executive_summary": executive_summary,
        "immediate_trends": immediate_trends,
        "watch_trends": watch_trends,
        "sections": sections,
        "evidence_summary": evidence_summary,
        "created_at": created_at,
        "frontend_briefings": frontend_payload,
        "dailySnapshot": frontend_payload["dailySnapshot"],
        "weeklySnapshot": frontend_payload["weeklySnapshot"],
        "evidenceSources": frontend_payload["evidenceSources"],
        "history": frontend_payload["history"],
    }


def _build_report(
    *,
    report_id: str,
    briefing_type: BriefingType,
    period: dict[str, Any],
    title: str | None,
    requested_by_user_id: int | None,
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
    provenance_base: dict[str, Any],
) -> dict[str, Any]:
    source_card_ids = [card["id"] for card in selected_cards]
    source_integrated_issue_ids = _json_list(
        briefing_basis.get("source_integrated_issue_ids")
        or provenance_base.get("source_integrated_issue_ids")
    )
    quality_flags = _json_list(
        briefing_basis.get("quality_flags") or provenance_base.get("quality_flags")
    )
    confidence = float(briefing_basis.get("confidence") or 0.0)
    report_title = title or f"{period['label']} 브리핑"
    briefing_lead = _brief_sentences(
        _block_text(briefing_basis.get("lead"), "finding") or _display_core_summary(briefing_basis),
        max_sentences=2,
        max_chars=_DISPLAY_BODY_MAX,
    )
    key_summary = _brief_sentence(
        _display_core_title(selected_cards, briefing_basis),
        max_chars=_DISPLAY_TITLE_MAX,
    )
    sk_ax_implication = _first_text(
        _block_text(briefing_basis.get("strategy_implication"), "finding"),
        _display_sk_ax_title(selected_cards),
    )
    sk_ax_implication = _brief_sentence(sk_ax_implication)
    key_change_cards = _key_change_cards_payload(selected_cards, briefing_basis)
    hidden_details = _hidden_details(selected_cards, briefing_basis)
    created_at = datetime.now(KST).isoformat()
    contract_payload = _briefing_contract_payload(
        report_id=report_id,
        briefing_type=briefing_type,
        period=period,
        report_title=report_title,
        executive_summary=briefing_lead or _briefing_lead(period, key_summary, selected_cards),
        selected_cards=selected_cards,
        briefing_basis=briefing_basis,
        created_at=created_at,
    )
    report = {
        "id": report_id,
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "title": report_title,
        "briefing_type": briefing_type,
        "date_from": period["date_from"].isoformat(),
        "date_to": period["date_to"].isoformat(),
        "period_label": period["label"],
        "created_at": created_at,
        "requested_by_user_id": requested_by_user_id,
        "status": "completed",
        "progress": 1.0,
        "key_summary": key_summary,
        "sk_implication": sk_ax_implication,
        "briefing_lead": briefing_lead or _briefing_lead(period, key_summary, selected_cards),
        "selected_cards": _public_selected_cards(selected_cards),
        "related_card_ids": source_card_ids,
        "source_integrated_issue_ids": source_integrated_issue_ids,
        "primary_card_news_id": source_card_ids[0] if source_card_ids else None,
        "primary_peer_company_id": selected_cards[0].get("peer_id") if selected_cards else None,
        "key_change_cards": key_change_cards,
        "core_change": {"items": copy.deepcopy(key_change_cards)},
        "interpretation_flow": _interpretation_flow_payload(briefing_basis, selected_cards),
        "hidden_details": hidden_details,
        "briefing_basis": briefing_basis,
        "confidence": confidence,
        **contract_payload,
        "provenance": {
            **provenance_base,
            "agent": "BriefingGenerationAgent",
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": source_card_ids,
            "source_integrated_issue_ids": source_integrated_issue_ids,
            "quality_flags": quality_flags,
            "briefing_analysis_basis": (
                "integrated_issues primary facts; card_news supplies card id anchor and "
                "legacy analysis_package fallback"
            ),
        },
    }
    grounded_flow = _grounded_front_interpretation_flow(report)
    if grounded_flow:
        report["interpretation_flow"] = grounded_flow
    briefing_report = _front_briefing_report_payload(
        result=report,
        briefing_type=briefing_type,
        period=period,
        selected_cards=selected_cards,
    )
    report["briefingReport"] = briefing_report
    report["flowSteps"] = briefing_report["flowSteps"]
    return report


def _empty_report(
    *,
    report_id: str,
    briefing_type: BriefingType,
    period: dict[str, Any],
    title: str | None,
    requested_by_user_id: int | None,
    provenance_base: dict[str, Any],
) -> dict[str, Any]:
    created_at = datetime.now(KST).isoformat()
    contract_payload = _briefing_contract_payload(
        report_id=report_id,
        briefing_type=briefing_type,
        period=period,
        report_title=title or f"{period['label']} 브리핑",
        executive_summary="",
        selected_cards=[],
        briefing_basis={},
        created_at=created_at,
    )
    return {
        "id": report_id,
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "title": title or f"{period['label']} 브리핑",
        "briefing_type": briefing_type,
        "date_from": period["date_from"].isoformat(),
        "date_to": period["date_to"].isoformat(),
        "period_label": period["label"],
        "created_at": created_at,
        "requested_by_user_id": requested_by_user_id,
        "status": "failed",
        "progress": 1.0,
        "error_message": "기간 조건에 맞는 카드뉴스가 없습니다.",
        "key_summary": "",
        "briefing_lead": "",
        "selected_cards": [],
        "related_card_ids": [],
        "source_integrated_issue_ids": [],
        "primary_card_news_id": None,
        "key_change_cards": [],
        "core_change": {"items": []},
        "interpretation_flow": {
            "label": "INTERPRETATION FLOW",
            "title": "해석 흐름 — 관찰부터 시사까지",
            "reasoning_summary": {
                "display_title": "해석 흐름 — 관찰부터 시사까지",
                "disclosure_level": "summarized_intermediate_artifacts",
                "note": "기간 조건에 맞는 입력이 없어 추론 trace를 생성하지 않았습니다.",
            },
            "steps": [],
        },
        "hidden_details": [],
        "requested_card_ids": provenance_base.get("requested_card_ids", []),
        "requested_integrated_issue_ids": provenance_base.get("requested_integrated_issue_ids", []),
        "confidence": 0.0,
        **contract_payload,
        "provenance": {
            **provenance_base,
            "agent": "BriefingGenerationAgent",
            "prompt_version": _PROMPT_VERSION,
        },
    }


def _key_change_cards_payload(
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[dict[str, Any]]:
    return _core_change_insight_items(selected_cards, briefing_basis)


def _core_change_insight_items(
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence_ids = [card["id"] for card in selected_cards if card.get("id")]
    evidence_ids = _dedupe_keep_order([str(card_id) for card_id in evidence_ids])
    entries = _key_change_source_entries(selected_cards)
    lead_text = _first_text(
        briefing_basis.get("briefing_insight"),
        _block_text(briefing_basis.get("lead"), "finding"),
    )
    market_title = _select_distinct_sentence(
        _market_signal_title_candidates(entries, briefing_basis),
        avoid=[],
        fallback="기간 내 카드뉴스에서 시장 변화 신호가 확인되었습니다.",
        max_chars=_DISPLAY_TITLE_MAX,
    )
    market_summary = _select_distinct_sentence(
        [
            _market_signal_reason_summary(entries),
            *_market_signal_summary_candidates(entries, briefing_basis),
        ],
        avoid=[lead_text, market_title],
        fallback="공통 수요와 평가 기준 변화가 같은 방향으로 확인되고 있습니다.",
        max_chars=_DISPLAY_BODY_MAX,
    )
    market_so_what = _select_distinct_sentence(
        _so_what_candidates(entries, briefing_basis),
        avoid=[market_title, market_summary],
        fallback=(
            "이 변화는 고객 제안과 경쟁사 대응에서 확인해야 할 평가 기준을 바꿀 수 있습니다."
        ),
        max_chars=_DISPLAY_REASON_MAX,
    )
    competitor_title = _select_distinct_sentence(
        [
            _competitor_move_group_title(entries),
            *_competitor_move_title_candidates(entries, briefing_basis),
        ],
        avoid=[market_title, market_summary, market_so_what],
        fallback=(
            "경쟁사들은 기간 내 감지된 시장 변화에 맞춰 사업과 기술 메시지를 조정하고 있습니다."
        ),
        max_chars=_DISPLAY_TITLE_MAX,
    )
    competitor_summary = _select_distinct_sentence(
        [
            _competitor_move_flow_summary(entries),
            *_competitor_move_summary_candidates(entries, briefing_basis, selected_cards),
        ],
        avoid=[market_title, market_summary, market_so_what, competitor_title],
        fallback=(
            "경쟁사 움직임은 개별 기술 발표보다 구축 범위, 실행 구조, "
            "고객 설득 근거를 함께 제시하는 방향으로 이동하고 있습니다."
        ),
        max_chars=_DISPLAY_BODY_MAX,
    )
    if _is_too_similar(competitor_summary, [market_summary], threshold=0.62):
        competitor_summary = _select_distinct_sentence(
            _competitor_move_summary_candidates(entries, briefing_basis, selected_cards),
            avoid=[market_title, market_summary, market_so_what, competitor_title],
            fallback=(
                "경쟁사들은 기술 역량을 단일 기능 설명이 아니라 고객이 비교할 "
                "구축 방식과 실행 근거로 전환하고 있습니다."
            ),
            max_chars=_DISPLAY_BODY_MAX,
        )
    if (
        len(_entry_company_labels(entries)) >= 2
        and _visible_company_count(
            competitor_summary,
            entries,
        )
        < 2
    ):
        competitor_summary = _brief_sentences(
            _competitor_move_flow_summary(entries),
            max_sentences=3,
            max_chars=_DISPLAY_BODY_MAX,
        )
    competitor_so_what = _select_distinct_sentence(
        _competitor_so_what_candidates(entries, briefing_basis),
        avoid=[market_title, market_summary, market_so_what, competitor_title, competitor_summary],
        fallback=(
            "경쟁사 움직임을 함께 보면 개별 이슈보다 "
            "경쟁 방식과 고객 설득 기준의 변화가 더 분명해집니다."
        ),
        max_chars=_DISPLAY_REASON_MAX,
    )
    return [
        {
            "seq": 1,
            "display_label": "시장 신호",
            "insight_type": "market_signal",
            "title": market_title,
            "description": market_summary,
            "why_important": market_so_what,
            "evidence_card_ids": evidence_ids,
        },
        {
            "seq": 2,
            "display_label": "경쟁사 움직임",
            "insight_type": "competitor_move",
            "title": competitor_title,
            "description": competitor_summary,
            "why_important": competitor_so_what,
            "evidence_card_ids": evidence_ids,
        },
    ]


def _competitor_move_flow_summary(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        clauses = _company_issue_clauses(entries)
        if clauses:
            joined = " ".join(clauses[:2])
            return (
                f"{joined} 이 흐름은 같은 시장 신호라도 경쟁사마다 고객 적용 범위와 "
                "구축 방식을 다르게 제시하고 있음을 보여줍니다."
            )
        return (
            "경쟁사별 발표가 고객 적용 범위, 구축 방식, 성과 근거를 비교할 신호로 나뉘고 있습니다."
        )
    signal = _first_text(
        _first_from_list([entry.get("peer_meaning") for entry in entries]),
        _first_from_list([entry.get("analysis_summary") for entry in entries]),
    )
    return _brief_sentence(signal, max_chars=_DISPLAY_BODY_MAX)


def _grounded_front_interpretation_flow(result: dict[str, Any]) -> dict[str, Any]:
    entries = _front_evidence_entries(result)
    if not entries:
        return {}
    evidence_ids = _front_evidence_card_ids(entries)
    flow = {
        "label": "INTERPRETATION FLOW",
        "title": "해석 흐름 — 관찰부터 시사까지",
        "steps": [
            {
                "seq": 1,
                "label": "관찰된 변화",
                "items": _front_interpretation_step_items(1, result, entries),
                "evidence_card_ids": evidence_ids,
            },
            {
                "seq": 2,
                "label": "평가축의 이동",
                "items": _front_interpretation_step_items(2, result, entries),
                "evidence_card_ids": evidence_ids,
            },
            {
                "seq": 3,
                "label": "경쟁 구도 영향",
                "items": _front_interpretation_step_items(3, result, entries),
                "evidence_card_ids": evidence_ids,
            },
            {
                "seq": 4,
                "label": "전략 시사",
                "items": _front_interpretation_step_items(4, result, entries),
                "evidence_card_ids": evidence_ids,
            },
        ],
    }
    return _attach_interpretation_reasoning_trace(flow, result, entries)


def _attach_interpretation_reasoning_trace(
    flow: dict[str, Any],
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(flow)
    updated["reasoning_summary"] = _interpretation_reasoning_summary(result, entries)
    steps = [step for step in _json_list(updated.get("steps")) if isinstance(step, dict)]
    for step in steps:
        step_seq = _optional_int(step.get("seq")) or 0
        step["reasoning_trace"] = _interpretation_step_reasoning_trace(
            step_seq=step_seq,
            label=str(step.get("label") or ""),
            result=result,
            entries=entries,
            items=[item for item in _json_list(step.get("items")) if isinstance(item, dict)],
        )
    updated["steps"] = steps
    return updated


def _interpretation_step_reasoning_trace(
    *,
    step_seq: int,
    label: str,
    result: dict[str, Any],
    entries: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    source_inputs = _reasoning_source_inputs(entries, step_seq=step_seq)
    intermediate = _reasoning_intermediate_artifacts(
        step_seq=step_seq,
        result=result,
        entries=entries,
        items=items,
        source_inputs=source_inputs,
    )
    return {
        "stage": _reasoning_stage_name(step_seq),
        "label": label,
        "question": _reasoning_question(step_seq),
        "source_inputs": source_inputs,
        "intermediate_artifacts": intermediate,
        "output_items": [
            {
                "seq": item.get("seq"),
                "title": _brief_sentence(item.get("title"), max_chars=_DISPLAY_TITLE_MAX),
                "description": _brief_sentences(
                    item.get("description"),
                    max_sentences=2,
                    max_chars=_DISPLAY_FLOW_MAX,
                ),
                "evidence_card_ids": _json_list(item.get("evidence_card_ids")),
            }
            for item in items[:3]
        ],
    }


def _reasoning_source_inputs(
    entries: list[dict[str, Any]],
    *,
    step_seq: int,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for entry in entries[:4]:
        evidence_text = _reasoning_entry_evidence_text(entry, step_seq=step_seq)
        if not evidence_text:
            continue
        inputs.append(
            {
                "card_id": entry.get("card_id"),
                "integrated_issue_id": entry.get("integrated_issue_id"),
                "company": entry.get("company"),
                "evidence": _brief_sentences(
                    evidence_text,
                    max_sentences=2,
                    max_chars=_DISPLAY_FLOW_MAX,
                ),
                "source_names": _json_list(entry.get("source_names"))[:3],
            }
        )
    return inputs[:4]


def _reasoning_entry_evidence_text(entry: dict[str, Any], *, step_seq: int) -> str:
    if step_seq == 1:
        return _first_text(
            _front_signal_name(entry),
            entry.get("market_signal"),
            entry.get("analysis_summary"),
            entry.get("main_issue"),
        )
    if step_seq == 2:
        return _first_text(
            entry.get("market_signal"),
            entry.get("impact_reason"),
            entry.get("sk_why"),
            entry.get("analysis_summary"),
        )
    if step_seq == 3:
        return _first_text(
            entry.get("peer_meaning"),
            entry.get("capability_change"),
            entry.get("analysis_summary"),
            entry.get("market_signal"),
        )
    if step_seq == 4:
        return _first_text(
            entry.get("sk_why"),
            entry.get("sk_impact"),
            _first_from_list(entry.get("recommended_actions")),
            entry.get("analysis_summary"),
        )
    return _first_text(entry.get("analysis_summary"), entry.get("main_issue"))


def _reasoning_intermediate_artifacts(
    *,
    step_seq: int,
    result: dict[str, Any],
    entries: list[dict[str, Any]],
    items: list[dict[str, Any]],
    source_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": "입력 근거 묶음",
            "output": _reasoning_source_bundle_sentence(source_inputs),
            "source_field": "integrated_issues + analysis_package",
        },
        {
            "name": "중간 산출물",
            "output": _reasoning_stage_output(step_seq, result, entries),
            "source_field": _reasoning_stage_source_field(step_seq),
        },
        {
            "name": "판단 기준",
            "output": _reasoning_decision_rule(step_seq),
            "source_field": "agent_rule",
        },
        {
            "name": "화면 반영",
            "output": _reasoning_output_sentence(items),
            "source_field": "interpretation_flow.steps.items",
        },
    ]


def _reasoning_stage_output(
    step_seq: int,
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> str:
    if step_seq == 1:
        return _front_observation_description(result, entries)
    if step_seq == 2:
        return _front_evaluation_shift_description(entries)
    if step_seq == 3:
        return _front_competition_impact_description(entries)
    if step_seq == 4:
        return _front_strategy_implication_description(entries)
    return _front_observation_description(result, entries)


def _front_interpretation_step_items(
    step_seq: int,
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    candidates: list[dict[str, Any]] = []
    if step_seq == 1:
        if robot_entry:
            robot_signal = _front_signal_name(robot_entry, tokens=_ROBOT_OPS_TOKENS)
            candidates.append(
                _front_step_item(
                    title=f"{robot_signal}가 운영 기반 신호로 확인됩니다.",
                    description=_front_robot_market_description(robot_entry),
                    evidence_card_ids=_entry_evidence_card_ids(robot_entry),
                )
            )
        if private_entry:
            private_signal = _front_signal_name(private_entry, tokens=_PRIVATE_AI_TOKENS)
            candidates.append(
                _front_step_item(
                    title=f"{private_signal}가 수요 변화 신호로 확인됩니다.",
                    description=_front_private_ai_market_description(private_entry),
                    evidence_card_ids=_entry_evidence_card_ids(private_entry),
                )
            )
        if not candidates:
            candidates.extend(_front_generic_observation_items(entries))
    elif step_seq == 2:
        if private_entry:
            candidates.append(
                _front_step_item(
                    title="데이터 통제 방식이 고객 평가 기준으로 올라오고 있습니다.",
                    description=_front_private_ai_market_description(private_entry),
                    evidence_card_ids=_entry_evidence_card_ids(private_entry),
                )
            )
        if robot_entry:
            candidates.append(
                _front_step_item(
                    title="도입 후 현장 운영 가능성이 평가 기준으로 올라오고 있습니다.",
                    description=_front_robot_market_description(robot_entry),
                    evidence_card_ids=_entry_evidence_card_ids(robot_entry),
                )
            )
        if not candidates:
            candidates.append(
                _front_step_item(
                    title=_front_evaluation_shift_title(entries),
                    description=_front_evaluation_shift_description(entries),
                    evidence_card_ids=_front_evidence_card_ids(entries),
                )
            )
    elif step_seq == 3:
        for entry in entries[:2]:
            sentence = _front_peer_move_sentence(entry)
            company = _first_text(entry.get("company"))
            signal = _front_signal_name(entry)
            if sentence and company:
                candidates.append(
                    _front_step_item(
                        title=f"{company}는 {signal} 신호를 사업 메시지로 연결하고 있습니다.",
                        description=sentence,
                        evidence_card_ids=_entry_evidence_card_ids(entry),
                    )
                )
        if not candidates:
            candidates.append(
                _front_step_item(
                    title=_front_competition_impact_title(entries),
                    description=_front_competition_impact_description(entries),
                    evidence_card_ids=_front_evidence_card_ids(entries),
                )
            )
    elif step_seq == 4:
        candidates.extend(
            _front_step_item(
                title=str(item.get("title") or ""),
                description=str(item.get("description") or ""),
                evidence_card_ids=_json_list(item.get("evidence_card_ids")),
            )
            for item in _grounded_front_sk_ax_view(result)
            if isinstance(item, dict)
        )
    return _front_limited_step_items(
        candidates,
        fallback=_front_step_item(
            title=_front_strategy_implication_title(entries)
            if step_seq == 4
            else _front_observation_title(entries),
            description=_front_strategy_implication_description(entries)
            if step_seq == 4
            else _front_observation_description(result, entries),
            evidence_card_ids=_front_evidence_card_ids(entries),
        ),
    )


def _grounded_front_sk_ax_view(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _front_evidence_entries(result)
    if not entries:
        return []
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    items: list[dict[str, Any]] = []
    if robot_entry:
        items.append(
            {
                "seq": len(items) + 1,
                "title": "제조 AX는 운영 책임과 성과 기준을 먼저 정해야 합니다.",
                "description": _front_robot_skax_description(robot_entry),
                "evidence_card_ids": _entry_evidence_card_ids(robot_entry),
            }
        )
    if private_entry:
        items.append(
            {
                "seq": len(items) + 1,
                "title": "프라이빗 AI는 데이터 통제 책임과 리스크 게이트를 먼저 정해야 합니다.",
                "description": _front_private_ai_skax_description(private_entry),
                "evidence_card_ids": _entry_evidence_card_ids(private_entry),
            }
        )
    items.append(
        {
            "seq": len(items) + 1,
            "title": "레퍼런스는 구축 사실보다 운영 성과 기준으로 재정렬해야 합니다.",
            "description": _front_reference_skax_description(entries),
            "evidence_card_ids": _front_evidence_card_ids(entries),
        }
    )
    if len(items) < _MAX_SKAX_ITEMS:
        items.extend(_front_generic_sk_ax_items(entries, start_seq=len(items) + 1))
    return items[:_MAX_SKAX_ITEMS]


def _front_generic_sk_ax_items(
    entries: list[dict[str, Any]],
    *,
    start_seq: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for entry in entries:
        for action in _json_list(entry.get("recommended_actions")):
            if _is_program_artifact_action(action):
                continue
            title = _front_action_title(action)
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            items.append(
                {
                    "seq": start_seq + len(items),
                    "title": title,
                    "description": _front_action_description(entry),
                    "evidence_card_ids": _entry_evidence_card_ids(entry),
                }
            )
            if len(items) >= _MAX_SKAX_ITEMS:
                return items
    return items


def _front_observation_title(entries: list[dict[str, Any]]) -> str:
    signals = _front_primary_signal_names(entries)
    if signals:
        joined = _join_korean(signals[:2])
        return f"{joined}{_subject_particle(joined)} 함께 확인됩니다."
    return "기간 내 핵심 변화 신호가 함께 확인됩니다."


def _front_observation_description(result: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    return (
        f"{_period_scope_phrase(result)}에서 {_front_join_company_signal_clauses(entries)} "
        "이 조합은 한 회사의 단일 이벤트가 아니라 운영 기반 확대와 수요 구조 변화가 "
        "같은 기간에 함께 나타난 것으로 읽힙니다."
    )


def _front_evaluation_shift_description(entries: list[dict[str, Any]]) -> str:
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    parts = []
    if private_entry:
        parts.append(
            f"{_front_signal_clause(private_entry, tokens=_PRIVATE_AI_TOKENS)}는 "
            "보안과 데이터 통제가 고객 평가 기준으로 올라오고 있음을 보여줍니다."
        )
    if robot_entry:
        parts.append(
            f"{_front_signal_clause(robot_entry, tokens=_ROBOT_OPS_TOKENS)}는 "
            "도입 후 현장 운영 가능성과 성과 검증이 중요해지고 있음을 보여줍니다."
        )
    return " ".join(parts[:2]) or "고객 평가 기준이 기능 보유 여부에서 운영 성과로 이동합니다."


def _front_competition_impact_description(entries: list[dict[str, Any]]) -> str:
    clauses = _front_join_company_signal_clauses(entries)
    return (
        f"{clauses} 이 조합은 경쟁사가 개별 기능을 따로 설명하는 단계에서 "
        "벗어나, 운영 인프라·데이터 통제·성장 전망을 하나의 패키지로 묶어 "
        "경쟁력을 설명하는 방향으로 가고 있음을 보여줍니다."
    )


def _front_reference_skax_description(entries: list[dict[str, Any]]) -> str:
    clauses = _front_reference_signal_summary(entries)
    prefix = f"대표 근거는 {clauses}입니다. " if clauses else ""
    return (
        f"{prefix}이 근거들을 레퍼런스로 보여줄 때는 무엇을 구축했는지보다 "
        "어떤 운영 문제가 줄었고, 어떤 지표로 안정화됐으며, 어디까지 확산됐는지를 "
        "먼저 정리해야 신뢰 가능한 운영 실적으로 읽힙니다."
    )


def _front_reference_signal_summary(entries: list[dict[str, Any]]) -> str:
    phrases: list[str] = []
    for entry in _representative_entries_for_period(entries, {"briefing_type": "monthly"}, limit=3):
        company = _first_text(entry.get("company"))
        signal = _period_signal_phrase(entry)
        phrase = _front_company_signal_phrase(company, signal)
        phrase = _readable_briefing_phrase(phrase, max_chars=90)
        if phrase:
            phrases.append(phrase)
    return _join_korean(_dedupe_keep_order(phrases[:3]))


def _public_interpretation_flow_from_result(
    result: dict[str, Any],
    *,
    repair: bool = True,
) -> dict[str, Any]:
    flow = _json_dict(result.get("interpretation_flow"))
    steps = [
        _compact_flow_step(step) for step in _json_list(flow.get("steps")) if isinstance(step, dict)
    ]
    return {
        "label": flow.get("label"),
        "title": flow.get("title"),
        "reasoning_summary": _compact_reasoning_summary(flow.get("reasoning_summary")),
        "steps": _repair_interpretation_flow_steps(steps, result) if repair else steps,
    }


def _repair_interpretation_flow_steps(
    steps: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for step in steps:
        current = copy.deepcopy(step)
        label = str(current.get("label") or "")
        title = str(step.get("title") or "")
        description = str(step.get("description") or "")
        grounded = _grounded_interpretation_description(label, title, result)
        current.pop("title", None)
        current.pop("description", None)
        items = [
            item
            for item in _json_list(current.get("items"))
            if isinstance(item, dict) and item.get("title") and item.get("description")
        ]
        if not items and (grounded or description or title):
            items = [
                {
                    "seq": 1,
                    "title": title or label,
                    "description": grounded or description or title,
                }
            ]
        if items:
            current["items"] = [
                _compact_visible_item(item, ("seq", "title", "description")) for item in items[:3]
            ]
        repaired.append(current)
    return repaired


def _period_key_change_description(
    result: dict[str, Any],
    insight_type: str,
    entries: list[dict[str, Any]],
) -> str:
    briefing_type = str(result.get("briefing_type") or "").strip()
    if briefing_type not in {"weekly", "monthly"} or not entries:
        return ""
    if insight_type == "market_signal":
        return _period_market_signal_description(result, entries)
    if insight_type == "competitor_move":
        return _period_competitor_move_description(result, entries)
    return ""


def _period_market_signal_description(
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> str:
    briefing_type = str(result.get("briefing_type") or "").strip()
    signal_text = _join_korean(_period_signal_names(entries, result)[:3])
    if not signal_text:
        return ""
    if briefing_type == "monthly":
        return (
            f"{signal_text}{_subject_particle(signal_text)} 함께 확인됩니다. "
            "월간 관점에서는 단발 이벤트보다 한 달 동안 반복된 수요 변화와 "
            "고객 평가 기준의 이동을 봐야 합니다."
        )
    return (
        f"{signal_text}{_subject_particle(signal_text)} 반복 확인됩니다. "
        "주간 관점에서는 이번 주 안에서 새로 강해진 신호와 "
        "후속 확인이 필요한 신호를 분리해 봐야 합니다."
    )


def _period_competitor_move_description(
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> str:
    briefing_type = str(result.get("briefing_type") or "").strip()
    representative_entries = _representative_entries_for_period(entries, result, limit=3)
    clauses = _company_issue_clauses(representative_entries)
    if not clauses:
        return ""
    joined = " ".join(clauses[:2])
    if briefing_type == "monthly":
        return (
            f"{joined} 따라서 개별 발표보다 한 달 동안 누적된 경쟁사별 실행 축과 "
            "고객 설득 방식의 차이를 비교해야 합니다."
        )
    return (
        f"{joined} 따라서 같은 방향의 메시지가 며칠 간격으로 반복되는지와 "
        "후속 수주·고객 사례로 이어지는지를 함께 봐야 합니다."
    )


def _period_signal_names(
    entries: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[str]:
    names: list[str] = []
    for entry in _representative_entries_for_period(entries, result, limit=4):
        signal = _period_signal_phrase(entry)
        if signal:
            names.append(signal)
    return _dedupe_keep_order(names)


def _representative_entries_for_period(
    entries: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    briefing_type = str(result.get("briefing_type") or "").strip()
    sorted_entries = sorted(entries, key=_entry_basis_sort_key)
    if briefing_type == "monthly" and len(sorted_entries) >= 3:
        indexes = [0, len(sorted_entries) // 2, len(sorted_entries) - 1]
        seeds = [sorted_entries[index] for index in indexes]
        candidates = [*seeds, *sorted_entries]
    elif briefing_type == "weekly":
        candidates = [*reversed(sorted_entries)]
    else:
        candidates = entries

    selected: list[dict[str, Any]] = []
    seen_companies: set[str] = set()
    seen_signals: set[str] = set()
    for entry in candidates:
        company = str(entry.get("company") or "").strip()
        signal = _front_signal_name(entry)
        signal_key = _normalize_similarity_text(signal)
        if company and company in seen_companies and signal_key in seen_signals:
            continue
        selected.append(entry)
        if company:
            seen_companies.add(company)
        if signal_key:
            seen_signals.add(signal_key)
        if len(selected) >= limit:
            break
    return selected or entries[:limit]


def _period_signal_phrase(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    candidates = [
        entry.get("main_issue"),
        entry.get("market_signal"),
        _front_signal_name(entry),
        entry.get("analysis_summary"),
    ]
    for candidate in candidates:
        phrase = _readable_briefing_phrase(candidate, company=company, max_chars=76)
        if phrase:
            return phrase
    return ""


def _repair_public_display_items(
    items: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    competitor_basis = _company_issue_sentence_from_result(result)
    signal_basis = _basis_signal_sentence_from_result(result)
    detailed_basis = _detailed_basis_sentence_from_result(result)
    for item in items:
        current = copy.deepcopy(item)
        title = str(current.get("title") or "")
        description = str(current.get("description") or "")
        insight_type = str(current.get("insight_type") or "")
        grounded_description = _grounded_key_change_description(insight_type, title, result)
        if grounded_description and (
            insight_type in {"market_signal", "competitor_move"}
            or not description
            or _is_vague_display_text(description)
            or _description_needs_detail(description)
            or _is_too_similar(
                description,
                [str(prev.get("description") or "") for prev in repaired],
            )
        ):
            current["description"] = grounded_description
            description = grounded_description
        joined = f"{title} {description}"
        if "경쟁사" in joined and competitor_basis and _needs_more_grounding(description, result):
            current["description"] = f"{description} {competitor_basis}".strip()
        elif _is_vague_display_text(description) and signal_basis:
            current["description"] = f"{description} {detailed_basis or signal_basis}".strip()
        if current.get("why_important") and _is_vague_display_text(str(current["why_important"])):
            current["why_important"] = _grounded_why_important(
                str(current["why_important"]),
                result,
            )
        repaired.append(current)
    return repaired


def _grounded_key_change_description(
    insight_type: str,
    title: str,
    result: dict[str, Any],
) -> str:
    scope = _period_scope_phrase(result)
    period_tail = _period_perspective_tail(result, insight_type)
    signals = _basis_signal_phrases_from_result(result)
    signal_text = _join_korean(signals[:2]) if signals else ""
    company_basis = _company_issue_sentence_from_result(result)
    if insight_type == "market_signal":
        if signal_text:
            return _join_display_sentences(
                f"{scope}에서 {signal_text}{_subject_particle(signal_text)} 함께 확인됩니다. "
                "이 조합은 개별 기업 이벤트보다 시장의 수요와 평가 기준이 "
                "운영 기반 중심으로 움직이고 있음을 보여줍니다.",
                period_tail,
            )
        return company_basis
    if insight_type == "competitor_move":
        if company_basis:
            return _join_display_sentences(
                f"{company_basis} 이 움직임은 경쟁사들이 단일 기술 설명보다 "
                "운영 역량과 구축 방식을 사업 메시지로 묶고 있음을 보여줍니다.",
                period_tail,
            )
        if signal_text:
            return _join_display_sentences(
                f"{scope}에서 {signal_text}{_subject_particle(signal_text)} 확인됩니다. "
                "이는 경쟁사들이 시장 신호를 고객 적용 범위와 구축 방식으로 "
                "해석하고 있음을 뜻합니다.",
                period_tail,
            )
    return _brief_sentence(title, max_chars=_DISPLAY_TITLE_MAX)
