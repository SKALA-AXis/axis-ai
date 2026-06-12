"""display_copy — briefing_generation_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md
"""

from __future__ import annotations

import copy
import re
from datetime import date
from difflib import SequenceMatcher
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
from src.services.analysis_units import (  # noqa: E402
    QUALITY_SUMMARY_ONLY_FALLBACK,
)

_DISPLAY_COPY_PROMPT_VERSION = "briefing-display-copy-v0.24-patterned-llm-guarded"


_MAX_MARKET_ITEMS = 3


_MAX_SKAX_ITEMS = 3


def _display_copy_quality_issues(
    draft: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    unit_quality_flags = _dedupe_keep_order(
        [
            str(flag)
            for card in selected_cards
            for flag in _json_list(card.get("quality_flags"))
            if str(flag).strip()
        ]
    )
    if QUALITY_SUMMARY_ONLY_FALLBACK in unit_quality_flags:
        issues.append(
            "summary_only_fallback 품질 플래그가 있는 분석 단위가 포함되어 있습니다. "
            "카드 표시 요약이 아니라 integrated_issue, analysis, implication 근거로 "
            "다시 작성하세요."
        )
    steps = _json_list(_nested_get(draft, "interpretation_flow", "steps"))
    typed_steps = [step for step in steps if isinstance(step, dict)]
    if len(typed_steps) < 4:
        issues.append("interpretation_flow는 반드시 4단계가 모두 있어야 합니다.")
    steps_without_items = [step for step in typed_steps if not _json_list(step.get("items"))]
    if steps_without_items:
        issues.append(
            "interpretation_flow의 각 step은 title/description 대신 items를 가져야 합니다. "
            "items는 단계별로 최소 1개, 최대 3개까지 작성하세요."
        )

    texts = _display_copy_visible_texts(draft)
    strong_terms = ("두각", "입지", "경쟁 우위", "주도하고", "선도하고")
    if any(term in text for text in texts for term in strong_terms):
        issues.append(
            "근거보다 강한 평가 표현이 포함되어 있습니다. '두각', '시장 입지', "
            "'경쟁 우위', '주도', '선도' 같은 표현은 analysis_units에 같은 "
            "의미의 근거가 없으면 '확인됩니다', '부각되고 있습니다', "
            "'중요성이 커지고 있습니다'처럼 낮춰 쓰세요."
        )
    weak_patterns = (
        "중요성이 커지고",
        "중요성을 부각",
        "중요한 역할",
        "핵심 요소로 자리",
        "강조하고 있습니다",
        "경쟁력을 강화",
        "성장을 도모",
        "전략적 포지셔닝",
    )
    weak_hits = [text for text in texts if any(pattern in text for pattern in weak_patterns)]
    if weak_hits:
        issues.append(
            "화면 문장에 이유 없는 중요도/성과 표현이 포함되어 있습니다. "
            "'중요성이 커지고 있습니다', '강조하고 있습니다', '경쟁력을 강화하고 있습니다' "
            "같은 표현은 실제 근거와 고객 평가 변화, 오퍼링 변화, 경쟁 방식 변화로 "
            "구체화하세요."
        )
    for item in _display_copy_visible_items(draft):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        description = str(item.get("description") or "")
        why = str(item.get("why_important") or "")
        if description and title and _is_too_similar(description, [title], threshold=0.64):
            issues.append(
                "description이 title을 반복합니다. description은 실제 근거와 "
                "그 근거가 title을 지지하는 이유를 설명해야 합니다."
            )
            break
        if why and (
            _is_too_similar(why, [title, description], threshold=0.62)
            or any(pattern in why for pattern in weak_patterns)
        ):
            issues.append(
                "why_important가 title/description을 반복하거나 추상적으로 끝납니다. "
                "고객 평가, 오퍼링 우선순위, 책임 조직, 자원 배분 중 무엇이 바뀌는지 "
                "구체적으로 쓰세요."
            )
            break
    for name in _dedupe_keep_order([_company_label(card) for card in selected_cards]):
        if not name:
            continue
        count = sum(text.count(name) for text in texts)
        if count > 8:
            issues.append(
                f"{name} 회사명이 {count}회 반복됩니다. 정확성이 필요한 곳만 남기고 "
                "나머지는 문맥형 표현으로 바꾸세요."
            )

    competitor = _find_key_change_by_type(draft, "competitor_move")
    if isinstance(competitor, dict):
        title = str(competitor.get("title") or "")
        if any(_company_label(card) and _company_label(card) in title for card in selected_cards):
            issues.append(
                "competitor_move.title이 특정 회사 하나의 움직임처럼 보입니다. "
                "경쟁사 움직임을 묶은 경쟁 방식 변화로 다시 쓰세요."
            )
    coverage_issues = _key_change_card_coverage_issues(draft, selected_cards)
    issues.extend(coverage_issues)
    return issues


def _key_change_card_coverage_issues(
    draft: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> list[str]:
    company_names = _dedupe_keep_order(
        [name for name in (_company_label(card) for card in selected_cards) if name]
    )
    if len(company_names) < 2:
        return []
    required_count = min(2, len(company_names))
    issues: list[str] = []
    for insight_type, label in (
        ("market_signal", "시장 신호"),
        ("competitor_move", "경쟁사 움직임"),
    ):
        item = _find_key_change_by_type(draft, insight_type)
        if not isinstance(item, dict):
            continue
        text = " ".join(
            str(item.get(key) or "") for key in ("title", "description", "why_important")
        )
        mentioned = [name for name in company_names if name in text]
        if len(mentioned) < required_count:
            issues.append(
                f"{label} 카드가 입력 카드 전체를 충분히 반영하지 못했습니다. "
                f"description에 최소 {required_count}개 대표 회사/카드 신호를 "
                "근거로 포함하고, 그 신호들이 왜 하나의 브리핑 판단으로 묶이는지 "
                "설명하세요."
            )
    return issues


def _display_copy_visible_texts(value: object) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"evidence_card_ids", "evidence_refs", "provenance"}:
                continue
            texts.extend(_display_copy_visible_texts(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_display_copy_visible_texts(item))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
    return texts


def _display_copy_visible_items(draft: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in _json_list(draft.get("key_change_cards")):
        if isinstance(item, dict):
            items.append(item)
    for step in _json_list(_nested_get(draft, "interpretation_flow", "steps")):
        if not isinstance(step, dict):
            continue
        items.append(step)
        for item in _json_list(step.get("items")):
            if isinstance(item, dict):
                items.append(item)
    return items


def _find_key_change_by_type(draft: dict[str, Any], insight_type: str) -> dict[str, Any] | None:
    for item in _json_list(draft.get("key_change_cards")):
        if isinstance(item, dict) and item.get("insight_type") == insight_type:
            return item
    return None


def _display_card_signal_index(selected_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for card in selected_cards:
        package = _analysis_package(card)
        integrated = _json_dict(package.get("integrated_issue"))
        analysis = _json_dict(package.get("analysis"))
        implication = _json_dict(package.get("implication"))
        peer = _json_dict(implication.get("peer_implication"))
        skax = _json_dict(implication.get("skax_implication"))
        business_signals = [
            _compact_visible_item(item, ("signal", "description"))
            for item in _json_list(integrated.get("business_signals"))
            if isinstance(item, dict)
        ]
        key_numbers = [
            _compact_visible_item(item, ("value", "context"))
            for item in _json_list(integrated.get("key_numbers"))
            if isinstance(item, dict)
        ]
        index.append(
            {
                "card_id": card.get("id"),
                "company_label": _company_label(card),
                "main_issue": _first_text(integrated.get("main_issue"), card.get("title")),
                "business_signals": business_signals[:3],
                "key_numbers": key_numbers[:3],
                "market_signal": _first_text(analysis.get("market_signal")),
                "analysis_summary": _first_text(analysis.get("analysis_summary")),
                "peer_meaning": _first_text(peer.get("peer_meaning")),
                "skax_why": _first_text(skax.get("why_important")),
                "recommended_actions": [
                    str(action).strip()
                    for action in _json_list(skax.get("recommended_actions"))[:3]
                    if str(action).strip()
                ],
            }
        )
    return index


def _display_payload_structure(value: object) -> object:
    if isinstance(value, dict):
        return {key: _display_payload_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_display_payload_structure(value[0])] if value else []
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return None
    return type(value).__name__


def _period_from_report(report: dict[str, Any]) -> dict[str, Any]:
    try:
        start = date.fromisoformat(str(report.get("date_from")))
        end = date.fromisoformat(str(report.get("date_to")))
    except ValueError:
        return {}
    return {
        "date_from": start,
        "date_to": end,
        "label": str(report.get("period_label") or ""),
    }


def _update_text_field(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    value = str(source.get(key) or "").strip()
    if value:
        target[key] = value


def _merge_key_change_cards(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_items = [
        item for item in _json_list(display_copy.get("key_change_cards")) if isinstance(item, dict)
    ]
    if not source_items:
        source_items = [
            item
            for item in _json_list(_nested_get(display_copy, "core_change", "items"))
            if isinstance(item, dict)
        ]
    current_items = [
        item for item in _json_list(updated.get("key_change_cards")) if isinstance(item, dict)
    ]
    if not current_items:
        current_items = [
            item
            for item in _json_list(_nested_get(updated, "core_change", "items"))
            if isinstance(item, dict)
        ]
    if not source_items:
        return
    by_type = {
        str(item.get("insight_type")): item
        for item in source_items
        if str(item.get("insight_type") or "").strip()
    }
    for index, item in enumerate(current_items):
        source = by_type.get(str(item.get("insight_type")))
        if source is None and index < len(source_items):
            source = source_items[index]
        if not isinstance(source, dict):
            continue
        for key in ("title", "description", "why_important"):
            _update_text_field(item, source, key)
        if str(item.get("insight_type")) == "competitor_move":
            title = str(item.get("title") or "")
            if _mentions_any_source_company(title, updated):
                item["title"] = "경쟁사들은 기술 신호를 운영 패키지와 성장 논리로 묶고 있습니다."
        if not item.get("description"):
            _update_text_field(item, source, "summary")
            if item.get("summary"):
                item["description"] = item.pop("summary")
        item.pop("summary", None)
        item.pop("so_what", None)
    updated["key_change_cards"] = current_items
    updated["core_change"] = {"items": copy.deepcopy(current_items)}


def _merge_flow(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_steps = _json_list(_nested_get(display_copy, "interpretation_flow", "steps"))
    current = _json_dict(updated.get("interpretation_flow"))
    current_steps = [step for step in _json_list(current.get("steps")) if isinstance(step, dict)]
    for index, step in enumerate(current_steps):
        source = _find_display_item_source(source_steps, step, index)
        if isinstance(source, dict):
            source_items = _json_list(source.get("items"))
            if not source_items and (source.get("title") or source.get("description")):
                source_items = [
                    {
                        "seq": 1,
                        "title": source.get("title"),
                        "description": source.get("description"),
                        "evidence_card_ids": source.get("evidence_card_ids"),
                    }
                ]
            if source_items:
                merged_items = _merged_display_list(
                    source_items=source_items,
                    current_items=[
                        item for item in _json_list(step.get("items")) if isinstance(item, dict)
                    ],
                    fields=("title", "description"),
                    max_items=3,
                    fill_remaining=False,
                )
                if merged_items:
                    step["items"] = _sanitize_flow_step_items({**step, "items": merged_items[:3]})
                else:
                    step.pop("items", None)
            else:
                step.pop("items", None)
        else:
            if step.get("items"):
                step["items"] = _sanitize_flow_step_items(step)
        step.pop("title", None)
        step.pop("description", None)
        step.pop("one_liner", None)
    if current_steps:
        current["steps"] = current_steps
        updated["interpretation_flow"] = current


def _sanitize_flow_step_items(step: dict[str, Any]) -> list[dict[str, Any]]:
    parent_title = str(step.get("title") or "")
    parent_description = str(step.get("description") or "")
    items: list[dict[str, Any]] = []
    seen_titles: list[str] = []
    seen_descriptions: list[str] = []
    for item in _json_list(step.get("items")):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title or not description:
            continue
        if _is_too_similar(title, [parent_title], threshold=0.9):
            continue
        if _is_too_similar(description, [parent_description], threshold=0.9):
            continue
        if _is_too_similar(title, seen_titles, threshold=0.82):
            continue
        if _is_too_similar(description, seen_descriptions, threshold=0.78):
            continue
        clean_item = _compact_visible_item(
            item,
            ("title", "description", "evidence_card_ids"),
        )
        clean_item["seq"] = len(items) + 1
        items.append(clean_item)
        seen_titles.append(title)
        seen_descriptions.append(description)
        if len(items) >= 3:
            break
    return items


def _merge_market_reading(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_items = _json_list(display_copy.get("market_reading"))
    current_items = [
        item for item in _json_list(updated.get("market_reading")) if isinstance(item, dict)
    ]
    merged_items = _merged_display_list(
        source_items=source_items,
        current_items=current_items,
        fields=("title", "description"),
        max_items=_MAX_MARKET_ITEMS,
    )
    if merged_items:
        updated["market_reading"] = merged_items


def _merge_sk_ax_view(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_items = _json_list(display_copy.get("sk_ax_view"))
    current_items = [
        item for item in _json_list(updated.get("sk_ax_view")) if isinstance(item, dict)
    ]
    merged_items = _merged_display_list(
        source_items=source_items,
        current_items=current_items,
        fields=("title", "description"),
        max_items=_MAX_SKAX_ITEMS,
    )
    if merged_items:
        updated["sk_ax_view"] = _repair_sk_ax_view_descriptions(merged_items, updated)


def _merged_display_list(
    *,
    source_items: list[Any],
    current_items: list[dict[str, Any]],
    fields: tuple[str, ...],
    max_items: int,
    fill_remaining: bool = True,
) -> list[dict[str, Any]]:
    typed_sources = [item for item in source_items if isinstance(item, dict)]
    if not typed_sources:
        return current_items[:max_items]
    merged_items: list[dict[str, Any]] = []
    for index, source in enumerate(typed_sources[:max_items]):
        current = current_items[index] if index < len(current_items) else {}
        merged: dict[str, Any] = {
            "seq": source.get("seq") or current.get("seq") or index + 1,
        }
        for key in fields:
            value = str(source.get(key) or "").strip()
            if value:
                merged[key] = value
            elif current.get(key):
                merged[key] = current[key]
        evidence = source.get("evidence_card_ids") or current.get("evidence_card_ids")
        if evidence:
            merged["evidence_card_ids"] = evidence
        merged_items.append(merged)
    if fill_remaining:
        for index in range(len(merged_items), min(len(current_items), max_items)):
            current = copy.deepcopy(current_items[index])
            current["seq"] = current.get("seq") or index + 1
            merged_items.append(current)
    return merged_items


def _mentions_any_source_company(text: str, result: dict[str, Any]) -> bool:
    value = str(text or "")
    if not value:
        return False
    for name in _source_company_names(result):
        if name and name in value:
            return True
    return False


def _source_company_names(result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for detail in _json_list(result.get("hidden_details")):
        if not isinstance(detail, dict):
            continue
        package = _json_dict(detail.get("analysis_package"))
        peer = _json_dict(_nested_get(package, "implication", "peer_implication"))
        company = _first_text(
            peer.get("company_name_ko"),
            detail.get("company_label"),
            _nested_get(package, "integrated_issue", "main_company"),
        )
        if company:
            names.append(company)
    return _dedupe_keep_order(names)


def _repair_sk_ax_view_descriptions(
    items: list[dict[str, Any]],
    result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    seen_descriptions: list[str] = []
    for item in items:
        current = copy.deepcopy(item)
        title = str(current.get("title") or "").strip()
        description = str(current.get("description") or "").strip()
        grounded = _grounded_sk_ax_description(title, result)
        if grounded:
            current["description"] = grounded
        elif (
            not description
            or _is_too_similar(description, [title], threshold=0.7)
            or _is_too_similar(description, seen_descriptions, threshold=0.72)
            or _is_vague_display_text(description)
            or _description_needs_detail(description)
        ):
            current["description"] = grounded or _sk_ax_description_from_title(title)
        seen_descriptions.append(str(current.get("description") or ""))
        repaired.append(current)
    return repaired


def _sk_ax_description_from_title(title: str) -> str:
    normalized = str(title or "")
    if any(token in normalized for token in ("성과", "KPI", "수치", "지표")):
        return (
            "고객은 기능 도입 자체보다 도입 후 장애, 통제, 생산성 문제가 "
            "얼마나 줄어드는지를 먼저 확인하려 합니다. 따라서 임원 의사결정에서는 "
            "적용 현장, 측정 지표, 안정화 기준을 사업 우선순위와 책임 조직 기준으로 "
            "함께 묶어야 합니다."
        )
    if any(token in normalized for token in ("레퍼런스", "사례")):
        return (
            "같은 구축 이력도 적용 현장, 운영 전환 과정, 확산 결과를 함께 보여줄 때 "
            "기술 공급 사례가 아니라 신뢰 가능한 운영 실적으로 읽힙니다."
        )
    if any(token in normalized for token in ("보안", "데이터 통제", "프라이빗")):
        return (
            "고객이 외부 모델 활용보다 데이터 통제와 책임 범위를 먼저 확인할 수 있으므로, "
            "구축 방식과 운영 거버넌스를 오퍼링 필수 조건과 리스크 승인 기준으로 "
            "함께 정해야 합니다."
        )
    if any(token in normalized for token in ("운영 시나리오", "운영 패키지", "통합")):
        return (
            "개별 기능보다 도입 후 운영 흐름을 먼저 보여주면 고객이 적용 범위, "
            "리스크 감소 방식, 성과 확인 지점을 더 빠르게 판단할 수 있습니다."
        )
    return (
        "이 시사점은 기술 설명을 회사 행동으로 바꾸는 부분이므로, 고객군, "
        "적용 범위, 책임 조직, 기대 효과를 한 번에 판단할 수 있게 결정 기준을 정해야 합니다."
    )


def _grounded_sk_ax_description(
    title: str,
    result: dict[str, Any] | None,
) -> str:
    if not result:
        return ""
    basis = _detailed_basis_sentence_from_result(
        result,
        focus_text=title,
    ) or _basis_signal_sentence_from_result(result)
    actions = _recommended_action_sentences_from_result(result)
    if not basis:
        return ""
    normalized = str(title or "")
    if any(token in normalized for token in ("운영 성과", "리스크", "실행 근거")):
        return (
            f"{basis} 이 근거는 고객의 관심이 기능 보유 여부보다 도입 후 "
            "운영 불확실성을 얼마나 낮출 수 있는지로 옮겨가고 있음을 보여줍니다. "
            "그래서 임원 의사결정에서는 기능 목록보다 줄일 운영 문제, 책임 범위, "
            "성과 측정 기준을 오퍼링 조건으로 먼저 확정해야 합니다."
        )
    if any(token in normalized for token in ("운영 시나리오", "운영 설계", "제안서 메시지")):
        return (
            f"{basis} 이 신호는 고객이 단일 기능보다 도입 후 운영 흐름과 "
            "책임 범위를 함께 판단한다는 뜻입니다. 따라서 SK AX는 기술 항목을 "
            "나열하기보다 데이터 수집, 이상 감지, 현장 적용, 성과 확인 책임을 "
            "오퍼링과 책임 조직에 함께 배정해야 합니다."
        )
    if any(token in normalized for token in ("프라이빗", "보안", "데이터 통제")):
        return (
            f"{basis} 따라서 데이터 통제 방식, 책임 범위, 운영 거버넌스를 "
            "리스크 승인 게이트로 정해야 고객이 도입 리스크를 판단할 수 있습니다."
        )
    if any(token in normalized for token in ("성과 수치", "KPI", "지표", "적용 현장")):
        return (
            f"{basis} 이 신호는 고객이 구축 여부보다 적용 현장에서 어떤 문제가 "
            "줄고 어떤 지표로 개선을 확인할 수 있는지를 보려 한다는 뜻입니다. "
            "따라서 메시지는 플랫폼 기능보다 운영 장면, 측정 지표, 안착 기준을 "
            "앞세워야 합니다."
        )
    if any(token in normalized for token in ("레퍼런스", "사례")):
        return (
            f"{basis} 이미 가진 사례도 구축 사실보다 적용 현장, 운영 전환 과정, "
            "확산 결과 순서로 보여줄 때 신뢰 가능한 운영 실적으로 읽힙니다."
        )
    if actions:
        return (
            f"{basis} 이 근거를 회사 행동으로 옮길 때는 기능명보다 고객의 "
            "운영 판단에 필요한 적용 범위, 책임 구조, 성과 확인 방식을 먼저 정해야 합니다."
        )
    return basis


def _detailed_basis_sentence_from_result(
    result: dict[str, Any],
    *,
    focus_text: str = "",
) -> str:
    phrases = _company_detail_phrases_from_result(result, focus_text=focus_text)
    if not phrases:
        return ""
    joined = _join_korean(phrases[:2])
    return f"구체적으로는 {joined}{_subject_particle(joined)} 확인됩니다."


def _company_detail_phrases_from_result(
    result: dict[str, Any],
    *,
    focus_text: str = "",
) -> list[str]:
    phrases: list[str] = []
    for detail in _json_list(result.get("hidden_details")):
        if not isinstance(detail, dict):
            continue
        package = _json_dict(detail.get("analysis_package"))
        company = _first_text(
            _nested_get(package, "implication", "peer_implication", "company_name_ko"),
            detail.get("company_label"),
        )
        signal = _business_signal_phrase(package)
        number_context = _key_number_context_phrase(package, signal)
        if company and signal and number_context:
            phrases.append(f"{company}의 {number_context}와 연결된 {signal}")
        elif company and signal:
            phrases.append(f"{company}의 {signal}")
    phrases = _dedupe_keep_order(phrases)
    focused = _filter_focus_phrases(phrases, focus_text)
    return focused or phrases


def _business_signal_phrase(package: dict[str, Any]) -> str:
    integrated = _json_dict(package.get("integrated_issue"))
    signals = [
        item for item in _json_list(integrated.get("business_signals")) if isinstance(item, dict)
    ]
    first_signal = signals[0] if signals else {}
    return _brief_noun_phrase(
        _first_text(
            first_signal.get("signal"),
            first_signal.get("description"),
            _nested_get(package, "analysis", "market_signal"),
        ),
        max_chars=58,
    )


def _key_number_context_phrase(package: dict[str, Any], signal: str = "") -> str:
    integrated = _json_dict(package.get("integrated_issue"))
    key_numbers = [
        item for item in _json_list(integrated.get("key_numbers")) if isinstance(item, dict)
    ]
    preferred = [
        item for item in key_numbers if _shares_keyword(signal, _first_text(item.get("context")))
    ]
    values: list[str] = []
    for item in preferred[:3]:
        if not isinstance(item, dict):
            continue
        value = _first_text(item.get("value"))
        context = _first_text(item.get("context"))
        if value and _looks_like_key_number(value):
            values.append(_format_key_number_context(value, context))
    return "·".join(_dedupe_keep_order(values))


def _format_key_number_context(value: str, context: str) -> str:
    if context:
        return f"{context} {value}"
    return value


def _looks_like_key_number(value: str) -> bool:
    text = str(value or "").strip()
    if len(text) > 24:
        return False
    return bool(re.search(r"\d", text))


def _filter_focus_phrases(phrases: list[str], focus_text: str) -> list[str]:
    focus = str(focus_text or "")
    if not focus:
        return []
    keyword_groups = [
        ("프라이빗", "보안", "데이터 통제", "AI", "클라우드"),
        ("로봇", "자동화", "제조", "스마트팩토리", "SW", "운영"),
        ("성과", "KPI", "수치", "지표", "리스크"),
    ]
    active = [group for group in keyword_groups if any(token in focus for token in group)]
    if not active:
        return []
    tokens = {token for group in active for token in group}
    return [phrase for phrase in phrases if any(token in phrase for token in tokens)]


def _shares_keyword(left: str, right: str) -> bool:
    left_text = str(left or "")
    right_text = str(right or "")
    if not left_text or not right_text:
        return False
    tokens = ("로봇", "AI", "클라우드", "스마트", "매출", "생산", "투입", "프라이빗")
    return any(token in left_text and token in right_text for token in tokens)


def _basis_signal_sentence_from_result(result: dict[str, Any]) -> str:
    signals = _basis_signal_phrases_from_result(result)
    if not signals:
        return ""
    joined = _join_korean(signals[:2])
    return f"{joined}{_subject_particle(joined)} 근거로 확인되고 있습니다."


def _basis_signal_phrases_from_result(result: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for package in _analysis_packages_from_result(result):
        integrated = _json_dict(package.get("integrated_issue"))
        analysis = _json_dict(package.get("analysis"))
        package_phrases: list[str] = []
        for signal in _json_list(integrated.get("business_signals")):
            if isinstance(signal, dict):
                package_phrases.append(_brief_noun_phrase(signal.get("signal"), max_chars=44))
        package_phrases.append(_brief_noun_phrase(analysis.get("market_signal"), max_chars=58))
        package_phrases.append(_brief_noun_phrase(analysis.get("impact_reason"), max_chars=58))
        phrases.extend([phrase for phrase in package_phrases if phrase][:1])
    return _dedupe_keep_order([phrase for phrase in phrases if phrase])


def _recommended_action_sentences_from_result(result: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for package in _analysis_packages_from_result(result):
        skax = _json_dict(_nested_get(package, "implication", "skax_implication"))
        actions.extend(
            str(action).strip() for action in _json_list(skax.get("recommended_actions"))
        )
    return _dedupe_keep_order(
        [_brief_sentence(action, max_chars=100) for action in actions if action]
    )


def _analysis_packages_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    for detail in _json_list(result.get("hidden_details")):
        if isinstance(detail, dict):
            package = _json_dict(detail.get("analysis_package"))
            if package:
                packages.append(package)
    return packages


def _brief_noun_phrase(value: object, *, max_chars: int) -> str:
    phrase = _brief_sentence(value, max_chars=max_chars)
    return _strip_terminal_punctuation(phrase)


def _subject_particle(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "이"
    code = ord(text[-1])
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 == 0:
        return "가"
    return "이"


def _is_vague_display_text(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    vague_markers = (
        "기능 설명보다",
        "실행 근거",
        "수요 변화",
        "사업 방향",
        "고객 설득 메시지",
        "중시하고 있습니다",
        "포지셔닝",
        "제안 초반에는 기능 목록보다",
        "어떤 운영 문제가 줄고",
    )
    return any(marker in text for marker in vague_markers) and len(text) < 90


def _find_display_item_source(
    source_items: list[Any],
    current_item: dict[str, Any],
    fallback_index: int,
) -> dict[str, Any] | None:
    current_seq = current_item.get("seq")
    current_label = str(
        current_item.get("label")
        or current_item.get("insight_type")
        or current_item.get("use_case")
        or ""
    )
    for source in source_items:
        if not isinstance(source, dict):
            continue
        if current_seq is not None and source.get("seq") == current_seq:
            return source
        source_label = str(
            source.get("label") or source.get("insight_type") or source.get("use_case") or ""
        )
        if current_label and source_label == current_label:
            return source
    if fallback_index < len(source_items) and isinstance(source_items[fallback_index], dict):
        return source_items[fallback_index]
    return None


def _strip_terminal_punctuation(value: str) -> str:
    return str(value or "").rstrip(" .。!?！？")


def _is_too_similar(value: str, others: list[str], *, threshold: float = 0.82) -> bool:
    normalized = _normalize_similarity_text(value)
    if not normalized:
        return False
    for other in others:
        other_normalized = _normalize_similarity_text(other)
        if not other_normalized:
            continue
        if normalized == other_normalized:
            return True
        if SequenceMatcher(None, normalized, other_normalized).ratio() >= threshold:
            return True
    return False


def _normalize_similarity_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _description_needs_detail(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return len(text) < 95 or len(re.split(r"(?<=[.!?。！？])\s+", text)) <= 1


def _compact_visible_item(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: item[key] for key in keys if item.get(key) not in (None, "", [])}
