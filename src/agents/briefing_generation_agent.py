# 작성일: 2026-05-27
# 작성자: 심유정
# 변경이력:
#   2026-05-27 심유정 — 브리핑 생성 에이전트 신설, 포맷
#   2026-05-29 최종민 — 분석/브리핑 조회에서 DELETED 카드 제외
#   2026-06-04 박진 — 통합 이슈 기반 mixer·executive 브리핑 플로우 추가
"""BriefingGenerationAgent.

기간별 integrated_issues를 선택한 뒤, 통합 이슈 저장소의 사실 근거와
분석/시사점 결과로 브리핑 화면 payload를 생성한다.

card_news는 화면과 저장 매핑에 필요한 card id anchor로 사용하고, 분석/시사점이
아직 정규화 테이블로 분리되지 않은 경우에만 legacy analysis_package 보조 경로로
참조한다.
"""

# ruff: noqa: E402  — sys.path 설정 후 import 필요(원본 패턴)

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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

if TYPE_CHECKING:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from src.agents.briefing_render._constants import (  # noqa: F401
    _DISPLAY_BODY_MAX,
    _DISPLAY_FLOW_MAX,
    _DISPLAY_REASON_MAX,
    _DISPLAY_TITLE_MAX,
    _MAX_DISPLAY_CARDS,
    _PRIVATE_AI_TOKENS,
    _REUSE_TTL_CURRENT_PERIOD,
    _ROBOT_OPS_TOKENS,
    BriefingType,
)
from src.agents.briefing_render.card_selector import (  # noqa: F401
    _aggregate_text,
    _average_confidence,
)
from src.agents.briefing_render.key_change import (  # noqa: F401
    _key_change_cards_from_result,
    _public_key_change_cards_from_result,
    _public_market_reading_from_result,
)
from src.agents.briefing_render.render import (  # noqa: F401
    _display_market_reading,
    _front_competitor_move_description,
    _front_competitor_move_importance,
    _front_competitor_move_title,
    _front_generic_market_description,
    _front_generic_market_items,
    _front_generic_market_title,
    _front_issue_names,
    _front_market_change_description,
    _front_market_change_importance,
    _front_market_change_title,
    _front_market_overview_description,
    _front_market_overview_title,
    _front_primary_signal_summary,
    _front_private_ai_market_title,
    _front_robot_market_title,
    _grounded_front_briefing_lead,
    _grounded_front_key_change_cards,
    _grounded_front_market_reading,
    _grounded_market_description,
    _market_reading_payload,
    _refresh_front_briefing_report,
    _repair_market_reading_items,
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
from src.config.env_loader import load_profile  # noqa: E402
from src.db.briefing_reports import load_briefing_report, save_briefing_report  # noqa: E402
from src.services.analysis_units import (  # noqa: E402
    analysis_units_from_cards,
    card_like_from_units,
    quality_flags_for_units,
    source_integrated_issue_ids,
)

log = logging.getLogger(__name__)


class BriefingGenerationAgent:
    """기간 단위 카드뉴스 브리핑을 생성하는 오케스트레이터.

    책임:
    - daily / weekly / monthly 기간에 해당하는 card_news만 선택한다.
    - 선택된 card_news의 evidence_payload.analysis_package를 읽는다.
    - integrated_issue / analysis / implication / classification / validation 결과를
      브리핑 화면 구조에 맞는 payload로 변환한다.

    책임 아님:
    - 카드뉴스 자체 생성
    - 원문 뉴스 재요약
    - 이메일 발송

    TODO: DB migration에서 briefing_type CHECK 제약은 daily / weekly / monthly만
    허용하고 custom은 제외해야 한다.
    """

    prompt_version = _PROMPT_VERSION

    def __init__(self, *, llm: Any | None = None) -> None:
        self._llm = llm

    async def generate(
        self,
        *,
        briefing_type: BriefingType,
        anchor_date: str | date | None = None,
        card_ids: list[str] | None = None,
        integrated_issue_ids: list[str] | None = None,
        peer_ids: list[str] | None = None,
        sectors: list[str] | None = None,
        title: str | None = None,
        requested_by_user_id: int | None = None,
        ratios: dict[str, Any] | None = None,
        user_context: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        save: bool = False,
        use_mock: bool = False,
        mock_path: str | Path | None = None,
        mock_items: list[dict[str, Any]] | None = None,
        refine_display_copy: bool = True,
        reuse_saved: bool = True,
    ) -> dict[str, Any]:
        """기간에 맞는 카드뉴스를 모아 브리핑 payload를 반환한다.

        ``card_ids``가 들어와도 기간 필터는 유지한다. 즉, 일간 브리핑이면 해당 일자
        범위에 속한 카드만 브리핑 근거로 사용된다.

        ``reuse_saved=True`` 이고 필터 없는 기본형 요청이면 ``briefing_reports`` 에
        저장된 동일 기간 브리핑을 재사용한다 (과거 기간은 무기한, 진행 중 기간은
        TTL 30분). LLM 정제(최대 4회 GPT 호출)를 매 조회마다 반복하지 않기 위한
        read-through 캐시 — 기본형 요청은 생성 후 항상 저장해 캐시를 채운다.
        """

        load_profile()
        requested_card_ids = _clean_ids(card_ids)
        requested_integrated_issue_ids = _clean_ids(integrated_issue_ids)
        period = _resolve_period(briefing_type, anchor_date)
        source_mode = (
            "mock_fixture"
            if use_mock or mock_path or mock_items
            else "integrated_issue_period_lookup"
        )
        report_id = _briefing_id(briefing_type, period["date_from"])
        # 캐시 가능한 요청 = 저장본(report_id 단위)과 내용이 동일해지는 요청.
        # 필터·사용자 맥락이 붙으면 저장본과 다른 결과라 재사용·적재 모두 제외.
        cacheable_request = (
            source_mode != "mock_fixture"
            and refine_display_copy
            and not requested_card_ids
            and not requested_integrated_issue_ids
            and not peer_ids
            and not sectors
            and not user_context
            and title is None
        )
        if reuse_saved and cacheable_request:
            saved = await asyncio.to_thread(load_briefing_report, report_id)
            if saved is not None and _saved_report_is_fresh(saved["completed_at"], period):
                log.info("Briefing 저장본 재사용 | id=%s", report_id)
                return saved["payload"]
        if source_mode == "mock_fixture":
            mock_source_items = _load_mock_items(mock_path=mock_path, mock_items=mock_items)
            selected_cards = _fetch_mock_period_cards(
                period=period,
                items=mock_source_items,
                card_ids=requested_card_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
                peer_ids=peer_ids,
                sectors=sectors,
                limit=limit,
            )
            selected_units = analysis_units_from_cards(selected_cards)
        else:
            mock_source_items = []
            selected_units = _fetch_period_analysis_units(
                period=period,
                card_ids=requested_card_ids,
                integrated_issue_ids=requested_integrated_issue_ids,
                peer_ids=peer_ids,
                sectors=sectors,
                limit=limit,
            )
        selected_cards = card_like_from_units(selected_units)
        selected_card_ids = [card["id"] for card in selected_cards]
        selected_integrated_issue_ids = source_integrated_issue_ids(selected_units)
        quality_flags = quality_flags_for_units(selected_units)
        excluded_card_ids = [
            card_id for card_id in requested_card_ids if card_id not in set(selected_card_ids)
        ]
        excluded_integrated_issue_ids = [
            issue_id
            for issue_id in requested_integrated_issue_ids
            if issue_id not in set(selected_integrated_issue_ids)
        ]
        provenance_base = _provenance_base(
            requested_card_ids=requested_card_ids,
            requested_integrated_issue_ids=requested_integrated_issue_ids,
            selected_card_ids=selected_card_ids,
            selected_integrated_issue_ids=selected_integrated_issue_ids,
            excluded_card_ids=excluded_card_ids,
            excluded_integrated_issue_ids=excluded_integrated_issue_ids,
            quality_flags=quality_flags,
            source_mode=source_mode,
        )

        if not selected_cards:
            return _empty_report(
                report_id=report_id,
                briefing_type=briefing_type,
                period=period,
                title=title,
                requested_by_user_id=requested_by_user_id,
                provenance_base=provenance_base,
            )

        _ = ratios, mock_source_items
        # 아래 동기 호출들(CPU 합성·sync llm.invoke·DB 쓰기)을 to_thread 로 격리 —
        # async generate 가 이벤트 루프에서 직접 실행하면 GPT 왕복 동안 루프가 멈춰
        # readiness/liveness probe 무응답 → 단일 replica 순단 (2026-06-11 실측, issue #146).
        # router.py 의 delivery_graph 처리(L291)와 동일 패턴.
        briefing_basis = await asyncio.to_thread(
            _briefing_basis_from_analysis_packages,
            selected_cards=selected_cards,
            period=period,
            user_context=user_context,
        )
        if refine_display_copy:
            briefing_basis = await asyncio.to_thread(
                _refine_briefing_basis_with_llm,
                briefing_basis=briefing_basis,
                selected_cards=selected_cards,
                period=period,
                user_context=user_context,
                llm=self._llm,
            )

        report = _build_report(
            report_id=report_id,
            briefing_type=briefing_type,
            period=period,
            title=title,
            requested_by_user_id=requested_by_user_id,
            selected_cards=selected_cards,
            briefing_basis=briefing_basis,
            provenance_base=provenance_base,
        )
        if refine_display_copy:
            report = await asyncio.to_thread(
                _refine_display_copy_with_llm,
                report=report,
                selected_cards=selected_cards,
                llm=self._llm,
            )
        report = _apply_period_perspective_to_report(report)
        report = _refresh_front_briefing_report(
            report=report,
            briefing_type=briefing_type,
            period=period,
            selected_cards=selected_cards,
        )
        report = _strip_visual_ellipsis_from_payload(report)
        report = _sanitize_internal_display_terms_from_payload(report)
        report = _normalize_visible_sentence_endings(report)
        if save or cacheable_request:
            # 기본형 요청은 save 플래그와 무관하게 저장 — 다음 조회가 재사용하도록
            # 캐시를 채운다 (id 단위 UPSERT 라 중복 적재 없음).
            await asyncio.to_thread(save_briefing_report, report, selected_cards=selected_cards)
        return report


# 진행 중 기간 브리핑의 재사용 허용 시간. 수집 파이프라인이 1시간 주기라
# 30분이면 최신 카드 반영 지연이 수집 주기의 절반을 넘지 않는다.


def _resolve_period(
    briefing_type: BriefingType,
    anchor_date: str | date | None,
) -> dict[str, Any]:
    today = datetime.now(KST).date()

    if briefing_type == "daily":
        target = _parse_date(anchor_date) or today
        start = target
        end = target
    elif briefing_type == "weekly":
        anchor = _parse_date(anchor_date) or today
        start = anchor - timedelta(days=anchor.weekday())
        end = anchor
    elif briefing_type == "monthly":
        anchor = _parse_date(anchor_date) or today
        start = anchor.replace(day=1)
        end = anchor
    else:
        raise ValueError(f"unsupported briefing_type: {briefing_type}")

    if start > end:
        raise ValueError(f"invalid briefing date range: {start} > {end}")

    start_at = datetime.combine(start, time.min, tzinfo=KST).astimezone(UTC)
    end_exclusive_at = datetime.combine(end + timedelta(days=1), time.min, tzinfo=KST)
    end_exclusive_at = end_exclusive_at.astimezone(UTC)
    return {
        "date_from": start,
        "date_to": end,
        "start_at": start_at,
        "end_exclusive_at": end_exclusive_at,
        "label": _period_label(briefing_type, start, end),
    }


def _refine_display_copy_with_llm(
    *,
    report: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    llm: Any | None,
) -> dict[str, Any]:
    """AnalysisUnit 기반 payload의 화면 표시문만 LLM으로 정제한다.

    다른 에이전트와 분리하기 위해 이 단계는 card_id별 analysis_unit만 입력으로 사용하고,
    evidence/provenance/hidden_details 같은 추적 필드는 코드가 그대로 보존한다.
    """

    if llm is None and not os.getenv("OPENAI_API_KEY"):
        log.info("Briefing display copy refinement skipped: OPENAI_API_KEY is not set")
        return report

    context = _display_copy_context(report, selected_cards)
    from src.observability import tracing_config

    messages = [
        ("system", _display_copy_system_prompt()),
        ("human", _display_copy_user_prompt(context)),
    ]
    try:
        response = (llm or _get_llm()).invoke(
            messages,
            config=tracing_config(agent="BriefingGenerationAgent", phase="refine_display_copy"),
        )
    except Exception as exc:  # pragma: no cover - external API safety net
        log.warning("Briefing display copy refinement failed | error=%s", exc)
        return report

    parsed = _parse_json_object(getattr(response, "content", response))
    if not parsed:
        return report
    issues = _display_copy_quality_issues(parsed, selected_cards)
    if issues:
        revision_messages = [
            ("system", _display_copy_system_prompt()),
            ("human", _display_copy_revision_prompt(context, parsed, issues)),
        ]
        try:
            revision_response = (llm or _get_llm()).invoke(
                revision_messages,
                config=tracing_config(
                    agent="BriefingGenerationAgent", phase="refine_display_copy_revision"
                ),
            )
        except Exception as exc:  # pragma: no cover - external API safety net
            log.warning("Briefing display copy revision failed | error=%s", exc)
        else:
            revised = _parse_json_object(getattr(revision_response, "content", revision_response))
            if revised:
                parsed = revised
        remaining_issues = _display_copy_quality_issues(parsed, selected_cards)
        if remaining_issues:
            log.info(
                "Briefing display copy refinement rejected | issues=%s",
                remaining_issues,
            )
            return report
    return _merge_display_copy(report, parsed, selected_cards=selected_cards)


def _display_copy_context(
    report: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "period": {
            "title": report.get("title"),
            "briefing_type": report.get("briefing_type"),
            "date_from": report.get("date_from"),
            "date_to": report.get("date_to"),
            "period_label": report.get("period_label"),
        },
        "source_card_ids": report.get("related_card_ids") or [],
        "source_integrated_issue_ids": report.get("source_integrated_issue_ids") or [],
        "current_display_structure": _display_payload_structure(_frontend_display_payload(report)),
        "card_signal_index": _display_card_signal_index(selected_cards),
        "analysis_units": [_compact_analysis_unit_for_display(card) for card in selected_cards],
    }


def _merge_display_copy(
    report: dict[str, Any],
    display_copy: dict[str, Any],
    *,
    selected_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(report)
    _update_text_field(updated, display_copy, "key_summary")
    _update_text_field(updated, display_copy, "briefing_lead")
    if updated.get("key_summary"):
        updated["sk_implication"] = _first_text(
            _nested_get(display_copy, "sk_implication"),
            updated.get("sk_implication"),
        )

    _merge_key_change_cards(updated, display_copy)

    _merge_flow(updated, display_copy)

    provenance = _json_dict(updated.get("provenance"))
    provenance["display_copy_prompt_version"] = _DISPLAY_COPY_PROMPT_VERSION
    provenance["display_copy_model"] = _LLM_MODEL
    updated["provenance"] = provenance
    if selected_cards is not None:
        updated = _refresh_contract_payload(updated, selected_cards)
    return updated


def _frontend_display_payload(result: dict[str, Any]) -> dict[str, Any]:
    has_display_copy = bool(_nested_get(result, "provenance", "display_copy_prompt_version"))
    if has_display_copy:
        briefing_lead = result.get("briefing_lead")
        key_change_cards = _public_key_change_cards_from_result(result, repair=False)
        interpretation_flow = _public_interpretation_flow_from_result(result, repair=False)
    else:
        briefing_lead = _grounded_front_briefing_lead(result) or result.get("briefing_lead")
        key_change_cards = _grounded_front_key_change_cards(
            result
        ) or _public_key_change_cards_from_result(result)
        interpretation_flow = _grounded_front_interpretation_flow(
            result
        ) or _public_interpretation_flow_from_result(result)
    visible = {
        "title": result.get("title"),
        "briefing_lead": briefing_lead,
        "key_change_cards": key_change_cards,
        "interpretation_flow": interpretation_flow,
        "related_card_ids": result.get("related_card_ids"),
        "primary_card_news_id": result.get("primary_card_news_id"),
        "hidden_details_count": len(result.get("hidden_details") or []),
    }
    return cast(dict[str, Any], _strip_default_hidden_fields(visible))


def _strip_default_hidden_fields(value: object) -> object:
    hidden_keys = {
        "basis_at",
        "evidence_card_ids",
        "evidence_refs",
        "importance_score",
        "langfuse_observation_id",
        "provenance",
    }
    if isinstance(value, dict):
        return {
            key: _strip_default_hidden_fields(item)
            for key, item in value.items()
            if key not in hidden_keys
        }
    if isinstance(value, list):
        return [_strip_default_hidden_fields(item) for item in value]
    return value


__all__ = ["BriefingGenerationAgent"]


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run BriefingGenerationAgent locally.")
    parser.add_argument("--type", default="daily", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--anchor-date", default="2026-05-26")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--no-mock", action="store_false", dest="mock")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Deprecated: default output is summary.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full payload including hidden details.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="llm",
        default=True,
        help="Skip LLM display-copy refinement.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show internal logs.")
    args = parser.parse_args()

    async def _main() -> None:
        if not args.verbose:
            logging.getLogger("src.middleware.analysis_ledger").setLevel(logging.ERROR)
        result = await BriefingGenerationAgent().generate(
            briefing_type=args.type,
            anchor_date=args.anchor_date,
            use_mock=args.mock,
            refine_display_copy=args.llm,
        )
        if not args.full:
            result = _frontend_display_payload(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(_main())
