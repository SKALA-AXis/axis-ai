"""briefing_render utils1 — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any, Literal

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


def _sanitize_internal_display_terms_from_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_internal_display_terms(value)
    if isinstance(value, list):
        return [_sanitize_internal_display_terms_from_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_internal_display_terms_from_payload(item) for key, item in value.items()
        }
    return value


def _sanitize_visible_copy_text(value: str) -> str:
    text = _sanitize_internal_display_terms(value)
    text = _strip_period_scope_prefix(text)
    text = _normalize_duplicate_company_prefix(text)
    text = _normalize_korean_plain_ending(text)
    return re.sub(r"\s+", " ", text).strip()


def _company_issue_sentence_from_result(result: dict[str, Any]) -> str:
    phrases = _company_issue_phrases_from_result(result)
    if not phrases:
        return ""
    joined = _join_korean(phrases[:3])
    return f"근거로는 {joined}{_subject_particle(joined)} 함께 확인됩니다."


def _needs_more_grounding(value: str, result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    text = str(value or "")
    company_phrases = _company_issue_phrases_from_result(result)
    signals = _basis_signal_phrases_from_result(result)
    grounding_terms = [
        *[phrase.split("의 ", 1)[0] for phrase in company_phrases if "의 " in phrase],
        *signals,
    ]
    return not any(term and term in text for term in grounding_terms)


def _normalize_briefing_trends(
    value: object,
    *,
    selected_cards: list[dict[str, Any]],
    fallback_cards: list[dict[str, Any]],
    role: Literal["immediate", "watch"],
) -> list[dict[str, Any]]:
    cards_by_id = {str(card.get("id")): card for card in selected_cards if card.get("id")}
    trends: list[dict[str, Any]] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("related_card_id") or "").strip()
        card = cards_by_id.get(card_id)
        if card is None and len(selected_cards) == 1:
            card = selected_cards[0]
        if card is None:
            continue
        trend = _trend_item_from_card(card, role=role)
        trend.update(
            {
                "title": _brief_sentence(
                    item.get("title") or trend.get("title"),
                    max_chars=_DISPLAY_TITLE_MAX,
                ),
                "reason": _brief_sentences(
                    item.get("reason") or trend.get("reason"),
                    max_sentences=2,
                    max_chars=_DISPLAY_REASON_MAX,
                ),
                "source_name": _first_text(item.get("source_name"), trend.get("source_name")),
                "published_at": _first_text(item.get("published_at"), trend.get("published_at")),
                "related_card_id": card.get("id"),
            }
        )
        trends.append(trend)
    if not trends:
        trends = [_trend_item_from_card(card, role=role) for card in fallback_cards]
    return trends[:3]


def _trend_item_from_card(
    card: dict[str, Any],
    *,
    role: Literal["immediate", "watch"],
) -> dict[str, Any]:
    source = _primary_source(card)
    return {
        "title": _trend_title(card),
        "peer_id": _first_text(card.get("peer_id"), card.get("company")),
        "reason": _trend_reason(card, role=role),
        "source_name": _source_name(source),
        "published_at": _source_published_at(card, source),
        "related_card_id": card.get("id"),
    }


def _normalize_briefing_sections(
    value: object,
    *,
    executive_summary: str,
    source_card_ids: list[str],
    immediate_trends: list[dict[str, Any]],
    watch_trends: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    allowed_ids = set(source_card_ids)
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        title = _brief_sentence(item.get("title"), max_chars=_DISPLAY_TITLE_MAX)
        if not title:
            continue
        related_ids = _valid_card_ids(
            item.get("related_card_ids"),
            allowed_ids=allowed_ids,
            fallback=source_card_ids,
        )
        bullets = [
            _brief_sentence(bullet, max_chars=_DISPLAY_BODY_MAX)
            for bullet in _json_list(item.get("bullets"))
            if str(bullet or "").strip()
        ][:4]
        sections.append(
            {
                "title": title,
                "summary": _brief_sentences(
                    item.get("summary") or executive_summary,
                    max_sentences=2,
                    max_chars=_DISPLAY_BODY_MAX,
                ),
                "bullets": bullets,
                "related_card_ids": related_ids,
            }
        )
    if sections:
        return sections[:4]

    key_bullets = _section_bullets_from_basis(briefing_basis)
    sections.append(
        {
            "title": "핵심 인사이트 요약",
            "summary": _brief_sentences(
                executive_summary,
                max_sentences=2,
                max_chars=_DISPLAY_BODY_MAX,
            ),
            "bullets": key_bullets,
            "related_card_ids": source_card_ids,
        }
    )
    if immediate_trends:
        sections.append(
            {
                "title": "오늘 바로 검토할 동향",
                "summary": _briefing_trends_summary(immediate_trends),
                "bullets": [str(item.get("title")) for item in immediate_trends],
                "related_card_ids": [
                    str(item.get("related_card_id"))
                    for item in immediate_trends
                    if item.get("related_card_id")
                ],
            }
        )
    if watch_trends:
        sections.append(
            {
                "title": "지속 관찰할 동향",
                "summary": _briefing_trends_summary(watch_trends),
                "bullets": [str(item.get("title")) for item in watch_trends],
                "related_card_ids": [
                    str(item.get("related_card_id"))
                    for item in watch_trends
                    if item.get("related_card_id")
                ],
            }
        )
    return sections


def _section_bullets_from_basis(briefing_basis: dict[str, Any]) -> list[str]:
    candidates = [
        _block_text(briefing_basis.get("common_pattern"), "finding"),
        _block_text(briefing_basis.get("comparison_point"), "finding"),
        _block_text(briefing_basis.get("hidden_conclusion"), "finding"),
        _block_text(briefing_basis.get("strategy_implication"), "finding"),
    ]
    return [
        _brief_sentence(item, max_chars=_DISPLAY_BODY_MAX)
        for item in _dedupe_keep_order([str(value or "").strip() for value in candidates])
        if item
    ][:4]


def _evidence_summary_from_basis(
    briefing_basis: dict[str, Any],
    *,
    selected_cards: list[dict[str, Any]],
    source_card_ids: list[str],
) -> list[str]:
    provided = [
        _brief_sentence(item, max_chars=_DISPLAY_BODY_MAX)
        for item in _json_list(briefing_basis.get("evidence_summary"))
        if str(item or "").strip()
    ]
    if provided:
        return provided[:4]
    source_names = _evidence_source_names(selected_cards)
    if source_names:
        joined_sources = ", ".join(source_names[:3])
        return [
            (
                f"{joined_sources} 등 {len(source_names)}개 출처가 "
                "integrated_issues 근거 체인에 연결됨"
            ),
            f"card_news anchor {len(source_card_ids)}건을 화면 이동 id로 사용함",
        ]
    return [
        "integrated_issues 저장 근거를 1차 입력으로 사용함",
        f"card_news anchor {len(source_card_ids)}건을 화면 이동 id로 사용함",
    ]


def _frontend_briefings_payload(
    *,
    report_id: str,
    briefing_type: BriefingType,
    period: dict[str, Any],
    report_title: str,
    executive_summary: str,
    selected_cards: list[dict[str, Any]],
    immediate_trends: list[dict[str, Any]],
    watch_trends: list[dict[str, Any]],
    evidence_summary: list[str],
) -> dict[str, Any]:
    evidence_sources = _evidence_source_names(selected_cards) or [
        "integrated_issues 통합 이슈 저장소",
        "card_news ID anchor",
        "legacy analysis_package 보조 근거",
    ]
    history = [
        {
            "id": report_id,
            "date": period["date_to"].isoformat(),
            "title": report_title,
            "status": "delivered",
            "summary": _brief_sentence(executive_summary, max_chars=_DISPLAY_BODY_MAX),
            "primaryCount": len(immediate_trends),
            "watchCount": len(watch_trends),
            "evidence": evidence_sources[:3],
        }
    ]
    return {
        "dailySnapshot": _frontend_snapshot(
            snapshot_type="daily",
            period=period,
            selected_cards=selected_cards,
            immediate_trends=immediate_trends,
            watch_trends=watch_trends,
            executive_summary=executive_summary,
        ),
        "weeklySnapshot": _frontend_snapshot(
            snapshot_type="weekly",
            period=period,
            selected_cards=selected_cards,
            immediate_trends=immediate_trends,
            watch_trends=watch_trends,
            executive_summary=executive_summary,
        ),
        "evidenceSources": _dedupe_keep_order([*evidence_sources, *evidence_summary])[:6],
        "history": history,
        "activeBriefingType": briefing_type,
    }


def _frontend_snapshot(
    *,
    snapshot_type: Literal["daily", "weekly"],
    period: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    immediate_trends: list[dict[str, Any]],
    watch_trends: list[dict[str, Any]],
    executive_summary: str,
) -> dict[str, Any]:
    if snapshot_type == "daily":
        title = _frontend_daily_title(period)
        primary_title = "오늘 바로 검토할 동향"
        watch_title = "지속 관찰할 동향"
        summary = (
            f"총 {len(selected_cards)}개 Peer사의 동향을 우선 검토 "
            f"{len(immediate_trends)}건과 지속 관찰 {len(watch_trends)}건으로 정리했습니다."
        )
    else:
        title = _frontend_weekly_title(period)
        primary_title = "이번 주 핵심 변화"
        watch_title = "연속 관찰 포인트"
        summary = (
            f"한 주 동안 {len(selected_cards)}개 Peer사의 사업 메시지와 "
            "근거 변화를 묶어서 정리했습니다."
        )
    return {
        "title": title,
        "summary": _brief_sentence(summary or executive_summary, max_chars=_DISPLAY_BODY_MAX),
        "sections": [
            {
                "title": primary_title,
                "items": [_frontend_section_item(item) for item in immediate_trends],
            },
            {
                "title": watch_title,
                "items": [_frontend_section_item(item) for item in watch_trends],
            },
        ],
    }


def _frontend_section_item(trend: dict[str, Any]) -> dict[str, str]:
    return {
        "headline": _brief_sentence(trend.get("title"), max_chars=_DISPLAY_TITLE_MAX),
        "source": _trend_source_label(trend),
    }


def _front_briefing_report_payload(
    *,
    result: dict[str, Any],
    briefing_type: BriefingType,
    period: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    signal_cards = _front_briefing_signal_cards(result)
    flow_steps = _front_briefing_flow_steps(_json_dict(result.get("interpretation_flow")))
    summary_line = _first_text(result.get("key_summary"), _front_signal_summary(signal_cards))
    digest = _dedupe_keep_order(
        [
            _first_text(result.get("briefing_lead"), result.get("executive_summary")),
            summary_line,
            *[_first_text(card.get("title"), card.get("summary")) for card in signal_cards],
            *_json_list(result.get("evidence_summary")),
        ]
    )
    front_cards = [_front_card_news_item(card) for card in selected_cards]
    return {
        "label": _front_briefing_label(briefing_type),
        "title": result.get("title") or f"{period['label']} 브리핑",
        "window": period.get("label") or "",
        "count": _front_briefing_count(briefing_type),
        "selectedCards": front_cards,
        "peers": _dedupe_keep_order(
            [_company_label(card) for card in selected_cards if _company_label(card)]
        ),
        "headline": _first_text(result.get("key_summary"), _front_signal_title(signal_cards)),
        "briefingLead": _first_text(result.get("briefing_lead"), result.get("executive_summary")),
        "briefingSummaryLine": summary_line,
        "whatHappenedDigest": digest,
        "signalCards": signal_cards,
        "meaning": _front_briefing_meaning(result, flow_steps),
        "benchmark": _front_briefing_benchmark(result),
        "flowSteps": flow_steps,
    }


def _front_card_news_item(card: dict[str, Any]) -> dict[str, Any]:
    source = _primary_source(card)
    source_url = _first_text(source.get("url"), source.get("link"), source.get("source_url"), "#")
    source_name = _source_name(source)
    title = _card_display_title(card)
    summary = [_sanitize_visible_copy_text(line) for line in _front_card_summary_lines(card)]
    package = _analysis_package(card)
    analysis = _json_dict(package.get("analysis"))
    implication = _json_dict(package.get("implication"))
    skax = _json_dict(implication.get("skax_implication"))
    published_at = _source_published_at(card, source)
    return {
        "id": str(card.get("id") or ""),
        "category": _first_text(card.get("sector"), card.get("primary_keyword_category"), "AX"),
        "date": published_at,
        "title": title,
        "coverImageUrl": _first_text(
            card.get("cover_image_url"),
            _nested_get(card, "display", "background_asset_url"),
            "/png.png",
        ),
        "coverImageAlt": f"{title} 대표 이미지",
        "summary": summary,
        "articlePages": [{"title": title, "paragraphs": summary}],
        "insights": _json_list(analysis.get("strategic_meaning")),
        "source": source_name,
        "sourceUrl": source_url,
        "detailTitle": title,
        "detailDescription": _sanitize_visible_copy_text(_card_summary(card)),
        "detailPoints": _dedupe_keep_order(
            [
                _first_text(analysis.get("analysis_summary")),
                _first_text(analysis.get("market_signal")),
                *_json_list(analysis.get("strategic_meaning")),
            ]
        ),
        "actionItems": _json_list(skax.get("recommended_actions")),
        "peer_id": card.get("peer_id"),
        "cluster_id": card.get("cluster_id"),
        "subtitle": _first_text(card.get("subtitle")),
        "category_label": _first_text(card.get("category_label"), card.get("sector")),
        "published_date": published_at,
        "summary_lines": summary,
        "event_type": card.get("event_type"),
        "sector": card.get("sector"),
        "keywords": _json_list(card.get("keywords")),
        "exposure_score": card.get("exposure_score"),
        "importance_score": card.get("importance_score"),
        "trust_score": card.get("trust_score"),
        "sources": [_front_card_source_payload(source)] if source else [],
        "source_count": len(_json_list(card.get("sources"))),
        "created_at": _first_text(card.get("created_at"), published_at),
    }


def _front_card_source_payload(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _first_text(source.get("title"), _source_name(source)),
        "url": _first_text(source.get("url"), source.get("link"), source.get("source_url"), "#"),
        "source_name": _source_name(source),
        "published_at": _first_text(source.get("published_at")),
    }


def _front_card_summary_lines(card: dict[str, Any]) -> list[str]:
    summary_lines = [
        str(item).strip() for item in _json_list(card.get("summary_lines")) if str(item).strip()
    ]
    if summary_lines:
        return summary_lines[:4]
    summary = _card_summary(card)
    return [summary] if summary else []


def _front_briefing_flow_steps(flow: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(_json_list(flow.get("steps")), 1):
        if not isinstance(step, dict):
            continue
        items = [item for item in _json_list(step.get("items")) if isinstance(item, dict)]
        headline = _first_text(*(item.get("title") for item in items))
        description = _first_text(*(item.get("description") for item in items))
        details = [_front_flow_detail(item) for item in items[1:]]
        details = [detail for detail in details if detail]
        if not headline and not description and not details:
            continue
        steps.append(
            {
                "id": f"generated-{step.get('seq') or index}",
                "label": _first_text(step.get("label"), f"Step {index}"),
                "headline": headline or _first_text(step.get("label"), f"Step {index}"),
                "description": description,
                "details": details,
            }
        )
    return steps


def _front_briefing_meaning(
    result: dict[str, Any],
    flow_steps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    basis = _json_dict(result.get("briefing_basis"))
    candidates = [
        (
            _block_text(basis.get("comparison_point"), "finding"),
            _block_text(basis.get("comparison_point"), "rationale"),
        ),
        (
            _block_text(basis.get("hidden_conclusion"), "finding"),
            _block_text(basis.get("hidden_conclusion"), "rationale"),
        ),
        *[
            (
                _first_text(step.get("headline")),
                _first_text(step.get("description")),
            )
            for step in flow_steps
        ],
    ]
    return [
        {
            "title": _brief_sentence(title, max_chars=_DISPLAY_TITLE_MAX),
            "reason": _brief_sentences(reason, max_sentences=2, max_chars=_DISPLAY_REASON_MAX),
        }
        for title, reason in candidates
        if _first_text(title)
    ][:4]


def _market_signal_title_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[object]:
    return [
        _block_text(briefing_basis.get("common_pattern"), "finding"),
        briefing_basis.get("briefing_insight"),
        *[entry.get("market_signal") for entry in entries],
        *[text for entry in entries for text in _json_list(entry.get("strategic_meaning"))],
        _repeated_signal_candidate(entries),
    ]


def _market_signal_summary_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[object]:
    return [
        _block_text(briefing_basis.get("common_pattern"), "rationale"),
        _display_core_summary(briefing_basis),
        _combine_blocks([entry.get("analysis_summary") for entry in entries], "", max_items=3),
        _combine_blocks([entry.get("integrated_text") for entry in entries], "", max_items=2),
    ]


def _competitor_move_summary_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> list[object]:
    return [
        _block_text(briefing_basis.get("comparison_point"), "rationale"),
        _combine_blocks([entry.get("peer_meaning") for entry in entries], "", max_items=2),
        _combine_blocks([entry.get("analysis_summary") for entry in entries], "", max_items=2),
        _competitor_move_summary(selected_cards),
    ]


def _so_what_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[object]:
    return [
        _cross_card_importance_sentence(entries),
        _block_text(briefing_basis.get("hidden_conclusion"), "finding"),
        _block_text(briefing_basis.get("hidden_conclusion"), "rationale"),
        *[entry.get("impact_reason") for entry in entries],
        *[entry.get("sk_why") for entry in entries],
        *[entry.get("sk_impact") for entry in entries],
        *[text for entry in entries for text in _json_list(entry.get("opportunities"))],
        *[text for entry in entries for text in _json_list(entry.get("threats"))],
    ]


def _visible_company_count(value: str, entries: list[dict[str, Any]]) -> int:
    text = str(value or "")
    return sum(1 for company in _entry_company_labels(entries) if company in text)


def _company_issue_clauses(entries: list[dict[str, Any]]) -> list[str]:
    clauses: list[str] = []
    seen_companies: set[str] = set()
    for entry in entries:
        company = _first_text(entry.get("company_label"), entry.get("company"))
        if company in seen_companies:
            continue
        issue = _company_issue_sentence_text(entry)
        if company and issue.startswith(company):
            issue = _strip_leading_company_prefix(issue, company)
        if company and issue:
            clauses.append(f"{company}는 {issue}")
            seen_companies.add(company)
        if len(clauses) >= 3:
            break
    return _dedupe_keep_order(clauses)


def _company_issue_sentence_text(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company_label"), entry.get("company"))
    for candidate in (
        entry.get("peer_meaning"),
        entry.get("main_issue"),
        entry.get("analysis_summary"),
    ):
        raw = str(candidate or "")
        if "프로필" in raw or "profile" in raw.lower():
            continue
        phrase = _readable_briefing_phrase(raw, company=company, max_chars=140)
        if phrase:
            return _front_polite_sentence(phrase)
    return ""


def _select_distinct_sentence(
    candidates: list[object],
    *,
    avoid: list[str],
    fallback: str,
    max_chars: int,
) -> str:
    for candidate in _unique_texts(candidates):
        sentence = _brief_sentences(candidate, max_sentences=2, max_chars=max_chars)
        if sentence and not _is_too_similar(sentence, avoid):
            return sentence
    fallback_sentence = _brief_sentence(fallback, max_chars=max_chars)
    if not _is_too_similar(fallback_sentence, avoid):
        return fallback_sentence
    return _generic_distinct_sentence(avoid, max_chars=max_chars)


def _interpretation_flow_payload(
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_evidence = _json_list(
        _nested_get(briefing_basis, "provenance", "source_card_ids")
    ) or _json_list(briefing_basis.get("sources_used"))
    display_steps = _display_flow_steps(briefing_basis, default_evidence)
    if display_steps:
        return {
            "label": "INTERPRETATION FLOW",
            "title": "해석 흐름 — 관찰부터 시사까지",
            "steps": display_steps,
        }
    trail = [
        _flow_item(
            1,
            "관찰된 변화",
            _block_text(briefing_basis.get("common_pattern"), "finding"),
        ),
        _flow_item(
            2,
            "평가축의 이동",
            _block_text(briefing_basis.get("comparison_point"), "finding"),
        ),
        _flow_item(
            3,
            "경쟁 구도 영향",
            _block_text(briefing_basis.get("hidden_conclusion"), "finding"),
        ),
        _flow_item(
            4,
            "전략 시사",
            _block_text(briefing_basis.get("strategy_implication"), "finding")
            or _action_text(briefing_basis.get("action_details"), "action"),
        ),
    ]
    return {
        "label": "INTERPRETATION FLOW",
        "title": "해석 흐름 — 관찰부터 시사까지",
        "steps": [_normalize_flow_step(step, default_evidence) for step in trail],
    }


def _sk_ax_view_payload(
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    default_evidence = _json_list(
        _nested_get(briefing_basis, "provenance", "source_card_ids")
    ) or _json_list(briefing_basis.get("sources_used"))
    display_items = _display_sk_ax_view(briefing_basis, default_evidence)
    if display_items:
        return display_items

    actions = briefing_basis.get("action_details")
    if not isinstance(actions, list):
        return []
    items = []
    for index, action in enumerate(actions, 1):
        if not isinstance(action, dict):
            continue
        items.append(
            {
                "seq": index,
                "use_case": action.get("use_case") or "SK AX 관점",
                "title": _brief_sentence(action.get("action")),
                "description": _brief_sentence(action.get("why")),
                "evidence_card_ids": action.get("evidence_card_ids") or [],
            }
        )
    return items[:_MAX_SKAX_ITEMS]


def _normalize_flow_step(step: object, default_evidence: list[Any] | None = None) -> dict[str, Any]:
    if not isinstance(step, dict):
        return _flow_item(0, "", "")
    evidence = step.get("evidence_card_ids") or step.get("evidence_refs") or []
    evidence_ids = [str(item) for item in _json_list(evidence) if str(item).strip()]
    if not evidence_ids:
        evidence_ids = [str(item) for item in default_evidence or [] if str(item).strip()]
    one_liner = _brief_sentence(step.get("one_liner") or step.get("answer"))
    title = _brief_sentence(step.get("title") or one_liner)
    description = _brief_sentences(
        step.get("description") or step.get("rationale") or one_liner,
        max_sentences=2,
        max_chars=_DISPLAY_FLOW_MAX,
    )
    return {
        "seq": step.get("seq") or step.get("step_idx") or 0,
        "label": str(step.get("label") or step.get("phase") or ""),
        "title": title,
        "description": description,
        "one_liner": one_liner or title,
        "evidence_card_ids": evidence_ids,
        "evidence_refs": evidence_ids,
        "langfuse_observation_id": step.get("langfuse_observation_id"),
    }


def _public_selected_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": card.get("id"),
            "card_id": card.get("id"),
            "integrated_issue_id": card.get("integrated_issue_id"),
            "company": card.get("company"),
            "peer_id": card.get("peer_id"),
            "company_label": _company_label(card),
            "sector": card.get("sector"),
            "sectors": card.get("sectors") or [],
            "title": _card_display_title(card),
            "importance_score": card.get("importance_score"),
            "basis_at": card.get("basis_at"),
            "evidence_card_ids": card.get("evidence_card_ids") or [card.get("id")],
            "has_analysis_package": bool(_analysis_package(card)),
        }
        for card in cards
    ]


def _hidden_details(
    cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    _ = briefing_basis
    details = []
    for card in cards:
        sources = _json_list(card.get("sources"))
        package = _analysis_package(card)
        details.append(
            {
                "card_id": card.get("id"),
                "integrated_issue_id": card.get("integrated_issue_id")
                or package.get("integrated_issue_id"),
                "evidence_card_ids": card.get("evidence_card_ids") or [card.get("id")],
                "source_raw_article_ids": card.get("source_raw_article_ids") or [],
                "sources": sources,
                "quality_flags": _json_list(card.get("quality_flags")),
                "confidence": _first_text(
                    _nested_get(package, "validation", "sc_score"),
                    _nested_get(package, "analysis", "confidence"),
                    _nested_get(package, "implication", "confidence"),
                ),
                "basis_at": card.get("basis_at") or _first_source_published_at(sources),
                "evidence_text": _evidence_texts(package),
                "analysis_package": package,
            }
        )
    return details


def _front_generic_observation_items(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in entries[:3]:
        signal = _front_signal_name(entry)
        if not signal:
            continue
        company = _first_text(entry.get("company"))
        title = (
            f"{company}에서 {signal}{_subject_particle(signal)} 확인됩니다."
            if company
            else f"{signal}{_subject_particle(signal)} 확인됩니다."
        )
        items.append(
            _front_step_item(
                title=title,
                description=_front_signal_description(entry),
                evidence_card_ids=_entry_evidence_card_ids(entry),
            )
        )
    return items


def _front_evidence_entries(result: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for detail in _json_list(result.get("hidden_details")):
        if not isinstance(detail, dict):
            continue
        package = _json_dict(detail.get("analysis_package"))
        if not package:
            continue
        integrated = _json_dict(package.get("integrated_issue"))
        analysis = _json_dict(package.get("analysis"))
        implication = _json_dict(package.get("implication"))
        peer = _json_dict(implication.get("peer_implication"))
        skax = _json_dict(implication.get("skax_implication"))
        sources = _json_list(detail.get("sources"))
        entries.append(
            {
                "card_id": _first_text(detail.get("card_id")),
                "integrated_issue_id": _first_text(
                    detail.get("integrated_issue_id"),
                    package.get("integrated_issue_id"),
                    integrated.get("integrated_issue_id"),
                ),
                "company": _first_text(
                    peer.get("company_name_ko"),
                    detail.get("company_label"),
                    integrated.get("main_company"),
                ),
                "main_issue": _first_text(integrated.get("main_issue")),
                "integrated_text": _first_text(integrated.get("integrated_text")),
                "business_signals": _front_business_signals(integrated),
                "key_numbers": _front_key_numbers(integrated),
                "analysis_summary": _first_text(analysis.get("analysis_summary")),
                "market_signal": _first_text(analysis.get("market_signal")),
                "impact_reason": _first_text(analysis.get("impact_reason")),
                "peer_meaning": _first_text(peer.get("peer_meaning")),
                "capability_change": _first_text(peer.get("capability_change")),
                "sk_why": _first_text(skax.get("why_important")),
                "sk_impact": _first_text(skax.get("potential_impact")),
                "recommended_actions": _json_list(skax.get("recommended_actions")),
                "basis_at": _first_text(detail.get("basis_at")),
                "source_names": _dedupe_keep_order(
                    [
                        _source_name(source)
                        for source in sources
                        if isinstance(source, dict) and _source_name(source) != "통합 이슈 근거"
                    ]
                ),
            }
        )
    return entries


def _front_entry_by_tokens(
    entries: list[dict[str, Any]],
    tokens: tuple[str, ...],
) -> dict[str, Any] | None:
    for entry in entries:
        if _entry_has_tokens(entry, tokens):
            return entry
    return None


def _has_token_entry(entries: list[dict[str, Any]], tokens: tuple[str, ...]) -> bool:
    return any(_entry_has_tokens(entry, tokens) for entry in entries)


def _front_signal_name(
    entry: dict[str, Any],
    *,
    tokens: tuple[str, ...] = (),
) -> str:
    signals = [item for item in _json_list(entry.get("business_signals")) if isinstance(item, dict)]
    if tokens:
        for item in signals:
            text = f"{item.get('signal', '')} {item.get('description', '')}"
            if any(token in text for token in tokens):
                return _brief_noun_phrase(item.get("signal"), max_chars=48)
    if signals:
        return _brief_noun_phrase(signals[0].get("signal"), max_chars=48)
    issue = _front_issue_name(entry)
    if issue:
        return issue
    return _brief_noun_phrase(
        _first_text(
            entry.get("market_signal"),
            entry.get("analysis_summary"),
        ),
        max_chars=48,
    )


def _front_issue_name(entry: dict[str, Any], *, max_chars: int = 76) -> str:
    company = _first_text(entry.get("company"))
    issue = _first_text(entry.get("main_issue"), entry.get("integrated_text"))
    if company and issue.startswith(company):
        issue = _strip_leading_company_prefix(issue, company)
    issue = _brief_noun_phrase(issue, max_chars=max_chars)
    return re.sub(r"\s+", " ", issue).strip(" ,，;:/·ㆍ-")


def _front_signal_clause(entry: dict[str, Any], *, tokens: tuple[str, ...] = ()) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=tokens)
    phrase = _front_company_signal_phrase(company, signal)
    if phrase:
        return phrase
    return signal


def _front_peer_move_sentence(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    peer_meaning = _brief_sentence(
        _first_text(entry.get("peer_meaning"), entry.get("analysis_summary")),
        max_chars=_DISPLAY_BODY_MAX,
    )
    if _is_profile_linkage_display_text(peer_meaning):
        issue = _front_issue_name(entry)
        peer_meaning = f"{issue}{_subject_particle(issue)} 확인됩니다." if issue else ""
    if company and peer_meaning.startswith(company):
        peer_meaning = _strip_leading_company_prefix(peer_meaning, company)
    peer_meaning = _front_polite_sentence(peer_meaning)
    if company and peer_meaning:
        return f"{company}는 {peer_meaning}"
    return peer_meaning


def _front_action_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry)
    reason = _brief_sentences(
        _first_text(entry.get("sk_why"), entry.get("sk_impact"), entry.get("analysis_summary")),
        max_sentences=1,
        max_chars=130,
    )
    basis = f"{company}의 {signal} 신호가 근거입니다." if company and signal else ""
    return _join_display_sentences(
        basis,
        reason,
        (
            "따라서 SK AX는 이 근거를 기능 설명이 아니라 고객군, 실행 범위, "
            "책임 조직, 성과 확인 기준을 정하는 의사결정으로 반영해야 합니다."
        ),
    )


def _front_evaluation_shift_title(entries: list[dict[str, Any]]) -> str:
    if _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS):
        return "고객 평가는 기능 보유보다 운영 가능성과 데이터 통제로 이동하고 있습니다."
    return "고객 평가는 기술 보유보다 운영 성과 검증으로 이동하고 있습니다."


def _front_strategy_implication_description(entries: list[dict[str, Any]]) -> str:
    return _join_display_sentences(
        (
            "운영 인프라와 데이터 통제 신호가 함께 확인되므로, 고객은 기능 보유보다 "
            "도입 후 누가 책임지고 어떤 성과 기준으로 안착시킬지를 먼저 보게 됩니다."
        ),
        (
            "따라서 기능 목록보다 운영 리스크를 어떻게 줄이고, 어떤 책임 범위와 "
            "성과 기준으로 안착을 검증할지 임원 의사결정 게이트로 먼저 정해야 합니다."
        ),
    )


def _front_private_ai_market_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=_PRIVATE_AI_TOKENS)
    phrase = _front_company_signal_phrase(company, signal)
    return (
        f"{phrase}{_subject_particle(phrase)} 확인됩니다. "
        "따라서 고객은 AI 기능 자체보다 데이터 위치, 접근 권한, 운영 책임, "
        "거버넌스 체계를 함께 평가하게 됩니다."
    ).strip()


def _front_robot_market_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=_ROBOT_OPS_TOKENS)
    number = _front_key_number_for_tokens(entry, _ROBOT_OPS_TOKENS)
    number_text = f"{number}와 함께 " if number else ""
    return (
        f"{company}에서 {number_text}{signal}{_subject_particle(signal)} 확인됩니다. "
        "이 근거는 제조 AX 경쟁에서 로봇 도입 자체보다 데이터 관리, 현장 SW 연동, "
        "운영 안정화 역량이 더 중요해지고 있음을 보여줍니다."
    ).strip()


def _front_robot_skax_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=_ROBOT_OPS_TOKENS)
    return (
        f"{company}의 {signal} 신호는 고객이 단일 기능보다 도입 후 운영 흐름을 "
        "함께 본다는 점을 보여줍니다. 따라서 SK AX는 로봇·설비 데이터 수집, "
        "이상 감지, 현장 SW 연동, 성과 확인 책임을 오퍼링과 책임 조직에 함께 "
        "배정해야 합니다."
    )


def _front_private_ai_skax_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=_PRIVATE_AI_TOKENS)
    phrase = _front_company_signal_phrase(company, signal)
    return (
        f"{phrase} 신호는 고객이 AI 기능보다 데이터 통제 방식과 "
        "운영 책임을 먼저 확인한다는 뜻입니다. 따라서 데이터 보관 위치, 접근 권한, "
        "책임 범위, 운영 거버넌스를 오퍼링 필수 조건과 리스크 승인 기준으로 "
        "함께 정해야 합니다."
    )


def _front_primary_signal_names(entries: list[dict[str, Any]]) -> list[str]:
    names = []
    for entry in entries[:3]:
        name = _front_signal_name(entry)
        if name:
            names.append(name)
    return _dedupe_keep_order(names)


def _front_join_company_signal_clauses(entries: list[dict[str, Any]]) -> str:
    clauses = []
    for entry in entries[:3]:
        company = _first_text(entry.get("company"))
        signal = _front_signal_name(entry)
        phrase = _front_company_signal_phrase(company, signal)
        if phrase:
            clauses.append(phrase)
    joined = _join_korean(_dedupe_keep_order(clauses))
    return f"{joined}{_subject_particle(joined)} 확인됩니다." if joined else ""


def _grounded_interpretation_description(
    label: str,
    title: str,
    result: dict[str, Any],
) -> str:
    focus = f"{label} {title}"
    basis = _detailed_basis_sentence_from_result(result, focus_text=focus)
    company_basis = _company_issue_sentence_from_result(result)
    scope = _period_scope_phrase(result)
    signals = _basis_signal_phrases_from_result(result)
    signal_text = _join_korean(signals[:2]) if signals else ""
    company_signals = _company_signal_map_from_result(result)
    robot_signal = _first_company_signal(
        company_signals,
        ("로봇", "스마트팩토리", "SW", "자동화"),
    )
    private_ai_signal = _first_company_signal(
        company_signals,
        ("프라이빗", "AI", "클라우드", "통제"),
    )
    if "관찰" in label:
        observed = _join_korean([robot_signal, private_ai_signal])
        observed_sentence = (
            f"{scope}에서 {observed}{_subject_particle(observed)} 동시에 확인됩니다."
            if observed
            else f"{scope}에서 {signal_text}{_subject_particle(signal_text)} 함께 확인됩니다."
            if signal_text
            else _detailed_basis_sentence_from_result(result)
        )
        return _join_display_sentences(
            observed_sentence or basis or company_basis,
            (
                "이 신호들이 함께 나타나면서 기간 내 변화가 단일 사건이 아니라 "
                "운영 기반과 수요 구조의 변화로 읽힙니다."
            ),
        )
    if "평가축" in label:
        evaluation_basis = _join_korean(
            [
                text
                for text in (
                    private_ai_signal,
                    robot_signal,
                )
                if text
            ]
        )
        return _join_display_sentences(
            (
                f"{evaluation_basis}{_subject_particle(evaluation_basis)} 근거가 됩니다."
                if evaluation_basis
                else basis
            ),
            (
                "따라서 고객 평가는 기술 보유 여부보다 보안·데이터 통제, "
                "운영 가능성, 성과 검증 근거를 더 강하게 보게 됩니다."
            ),
        )
    if "경쟁" in label:
        return _join_display_sentences(
            company_basis or basis,
            (
                "이 흐름은 경쟁사들이 각자의 기술 신호를 단일 기능 설명이 아니라 "
                "운영 패키지, 구축 방식, 성장 논리로 묶어 경쟁 메시지를 "
                "재구성하고 있음을 보여줍니다."
            ),
        )
    if "전략" in label:
        strategy_basis = _join_korean([text for text in (robot_signal, private_ai_signal) if text])
        return _join_display_sentences(
            (
                f"{strategy_basis}{_subject_particle(strategy_basis)} 전략 시사의 근거가 됩니다."
                if strategy_basis
                else "이 단계는 앞선 관찰과 평가축 이동을 SK AX의 회사 대응으로 옮기는 결론입니다."
            ),
            (
                "따라서 의사결정 기준은 기술 기능 중심에서 운영 리스크, "
                "책임 범위, 성과 검증 기준 중심으로 옮겨야 합니다."
            ),
        )
    return basis or company_basis


def _period_scope_phrase(result: dict[str, Any]) -> str:
    date_from = str(result.get("date_from") or "").strip()
    date_to = str(result.get("date_to") or "").strip()
    briefing_type = str(result.get("briefing_type") or "").strip()
    if date_from and date_to and date_from == date_to:
        return f"{_display_date_ko(date_from)} 일간에 수집된 근거"
    if date_from and date_to:
        type_label = {"weekly": "주간", "monthly": "월간"}.get(briefing_type, "기간")
        start_label = _display_date_ko(date_from)
        end_label = _display_date_ko(date_to)
        return f"{start_label}부터 {end_label}까지 {type_label}에 수집된 근거"
    period_label = str(result.get("period_label") or "").strip().replace(".", "")
    if period_label:
        return f"{period_label}에 수집된 근거"
    if briefing_type == "daily":
        return "선택한 일간 범위에 수집된 근거"
    if briefing_type == "weekly":
        return "선택한 주간 범위에 수집된 근거"
    if briefing_type == "monthly":
        return "선택한 월간 범위에 수집된 근거"
    return "선택한 기간에 수집된 근거"


def _readable_briefing_phrase(
    value: object,
    *,
    company: str = "",
    max_chars: int,
) -> str:
    text = _sanitize_internal_display_terms(
        str(_strip_visual_ellipsis_from_payload(str(value or "")) or "")
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _looks_like_mostly_english(text):
        return ""
    company_text = str(company or "").strip()
    if company_text:
        text = re.sub(rf"^{re.escape(company_text)}\s*[,，]\s*", f"{company_text}의 ", text)
    text = _brief_noun_phrase(text, max_chars=max_chars)
    text = re.sub(r"\s+", " ", text).strip(" ,，;:/·ㆍ-")
    if text.endswith(("보인", "나타낸", "보여주는", "가리키는")):
        return ""
    return text
