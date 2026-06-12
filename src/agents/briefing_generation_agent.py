"""BriefingGenerationAgent.

기간별 integrated_issues를 선택한 뒤, 통합 이슈 저장소의 사실 근거와
분석/시사점 결과로 브리핑 화면 payload를 생성한다.

card_news는 화면과 저장 매핑에 필요한 card id anchor로 사용하고, 분석/시사점이
아직 정규화 테이블로 분리되지 않은 경우에만 legacy analysis_package 보조 경로로
참조한다.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import sys
from datetime import UTC, date, datetime, time, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

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
    from langchain_openai import ChatOpenAI

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from src.config.env_loader import load_profile  # noqa: E402
from src.db.briefing_reports import load_briefing_report, save_briefing_report  # noqa: E402
from src.services.analysis_units import (  # noqa: E402
    QUALITY_SUMMARY_ONLY_FALLBACK,
    analysis_units_from_cards,
    card_like_from_units,
    confidence_penalty_for_flags,
    quality_flags_for_units,
    source_integrated_issue_ids,
)

log = logging.getLogger(__name__)

BriefingType = Literal["daily", "weekly", "monthly"]

_PROMPT_VERSION = "briefing-generation-v0.3-period-briefing"
_BRIEFING_SYNTHESIS_PROMPT_VERSION = "briefing-synthesis-v0.4-integrated-issue-frontend-contract"
_DISPLAY_COPY_PROMPT_VERSION = "briefing-display-copy-v0.24-patterned-llm-guarded"
_LLM_MODEL = os.getenv("BRIEFING_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o"
_MAX_DISPLAY_CARDS = 3
_MAX_MARKET_ITEMS = 3
_MAX_SKAX_ITEMS = 3

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.1,
            max_completion_tokens=2400,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


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
        if save or cacheable_request:
            # 기본형 요청은 save 플래그와 무관하게 저장 — 다음 조회가 재사용하도록
            # 캐시를 채운다 (id 단위 UPSERT 라 중복 적재 없음).
            await asyncio.to_thread(save_briefing_report, report, selected_cards=selected_cards)
        return report


# 진행 중 기간 브리핑의 재사용 허용 시간. 수집 파이프라인이 1시간 주기라
# 30분이면 최신 카드 반영 지연이 수집 주기의 절반을 넘지 않는다.
_REUSE_TTL_CURRENT_PERIOD = timedelta(minutes=30)


def _saved_report_is_fresh(completed_at: Any, period: dict[str, Any]) -> bool:
    """저장된 브리핑을 재사용해도 되는지 판단한다.

    기간이 끝난 브리핑(과거 일/주/월)은 근거 데이터가 더 늘지 않으므로 항상
    재사용하고, 오늘이 포함된 진행 중 기간은 TTL 안에서만 재사용한다.
    """

    date_to = period.get("date_to")
    if isinstance(date_to, date) and date_to < datetime.now(KST).date():
        return True
    if not isinstance(completed_at, datetime):
        return False
    completed = completed_at if completed_at.tzinfo else completed_at.replace(tzinfo=UTC)
    return datetime.now(UTC) - completed.astimezone(UTC) <= _REUSE_TTL_CURRENT_PERIOD


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
        end = start + timedelta(days=6)
    elif briefing_type == "monthly":
        anchor = _parse_date(anchor_date) or today
        start = anchor.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = next_month - timedelta(days=1)
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


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(KST).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text_value = str(value).strip()
    if len(text_value) == 7:
        year, month = text_value.split("-", 1)
        return date(int(year), int(month), 1)
    return datetime.fromisoformat(text_value).date()


def _clean_ids(values: list[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _provenance_base(
    *,
    requested_card_ids: list[str],
    requested_integrated_issue_ids: list[str],
    selected_card_ids: list[str],
    selected_integrated_issue_ids: list[str],
    excluded_card_ids: list[str],
    excluded_integrated_issue_ids: list[str],
    quality_flags: list[str],
    source_mode: str,
) -> dict[str, Any]:
    provenance = {
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "source_card_ids": selected_card_ids,
        "source_integrated_issue_ids": selected_integrated_issue_ids,
        "requested_card_ids": requested_card_ids,
        "requested_integrated_issue_ids": requested_integrated_issue_ids,
        "selected_card_ids": selected_card_ids,
        "selected_integrated_issue_ids": selected_integrated_issue_ids,
        "excluded_card_ids": excluded_card_ids,
        "excluded_integrated_issue_ids": excluded_integrated_issue_ids,
        "quality_flags": quality_flags,
        "source_mode": source_mode,
        "briefing_analysis_basis": (
            "integrated_issues primary facts; card_news supplies card id anchor and "
            "legacy analysis_package fallback"
        ),
    }
    if excluded_card_ids or excluded_integrated_issue_ids:
        provenance["exclusion_reason"] = "out_of_period"
    return provenance


def _briefing_basis_from_analysis_packages(
    *,
    selected_cards: list[dict[str, Any]],
    period: dict[str, Any],
    user_context: str | None,
) -> dict[str, Any]:
    card_ids = [card["id"] for card in selected_cards]
    units = analysis_units_from_cards(selected_cards)
    integrated_issue_ids = source_integrated_issue_ids(units)
    quality_flags = quality_flags_for_units(units)
    entries = _analysis_package_entries(selected_cards)
    analysis_summaries = [entry["analysis_summary"] for entry in entries]
    market_signals = [entry["market_signal"] for entry in entries]
    strategic_meanings = [
        text
        for entry in entries
        for text in _json_list(entry.get("strategic_meaning"))
        if str(text).strip()
    ]
    peer_moves = [entry["peer_meaning"] for entry in entries]
    sk_reasons = [entry["sk_why"] for entry in entries]
    sk_impacts = [entry["sk_impact"] for entry in entries]
    action_details = _action_details_from_analysis_packages(entries)
    recommended_actions = [
        str(item.get("action") or "").strip()
        for item in action_details
        if str(item.get("action") or "").strip()
    ]
    confidence_values = [
        value
        for entry in entries
        for value in (
            entry.get("analysis_confidence"),
            entry.get("implication_confidence"),
            entry.get("validation_score"),
        )
        if isinstance(value, (int, float))
    ]
    confidence = (
        round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.0
    )
    if quality_flags:
        confidence = round(
            max(0.0, confidence - confidence_penalty_for_flags(quality_flags)),
            2,
        )
    fallback = f"{period['label']} 기간에 확인된 카드뉴스 기반 브리핑입니다."
    lead_finding = _combine_blocks(
        market_signals or analysis_summaries,
        fallback,
        max_items=2,
        max_chars=220,
    )
    common_finding = _combine_blocks(
        market_signals or analysis_summaries,
        fallback,
        max_items=2,
        max_chars=180,
    )
    common_rationale = _combine_blocks(analysis_summaries, common_finding, max_items=2)
    comparison_finding = _comparison_finding(entries)
    comparison_rationale = _combine_blocks(peer_moves or analysis_summaries, comparison_finding)
    hidden_finding = _combine_blocks(
        strategic_meanings or market_signals,
        common_finding,
        max_items=2,
        max_chars=180,
    )
    hidden_rationale = _combine_blocks(strategic_meanings or analysis_summaries, hidden_finding)
    strategy_values: list[object] = [
        item for item in (recommended_actions or sk_impacts or sk_reasons)
    ]
    strategy_finding = _combine_blocks(
        strategy_values,
        _display_sk_ax_title(selected_cards),
        max_items=2,
        max_chars=220,
    )
    strategy_rationale = _combine_blocks(sk_reasons or sk_impacts, strategy_finding)
    if user_context and user_context.strip():
        strategy_rationale = _first_text(strategy_rationale, user_context.strip())
    return {
        "basis_id": f"BR-BASIS-{period['date_from']:%Y%m%d}",
        "briefing_insight": lead_finding,
        "lead": {
            "finding": lead_finding,
            "evidence_card_ids": card_ids,
        },
        "core_change": {
            "finding": comparison_finding,
            "rationale": comparison_rationale,
            "evidence": _basis_evidence(entries),
            "evidence_card_ids": card_ids,
        },
        "common_pattern": _analysis_basis_block(common_finding, common_rationale, entries),
        "comparison_point": _analysis_basis_block(
            comparison_finding,
            comparison_rationale,
            entries,
        ),
        "hidden_conclusion": _analysis_basis_block(hidden_finding, hidden_rationale, entries),
        "strategy_implication": {
            "finding": strategy_finding,
            "rationale": strategy_rationale,
            "evidence": _basis_evidence(entries),
            "evidence_card_ids": card_ids,
        },
        "recommended_action_basis": _dedupe_keep_order(
            [str(item).strip() for item in sk_reasons + sk_impacts if str(item).strip()]
        ),
        "action_details": action_details,
        "recommended_actions": recommended_actions,
        "confidence": confidence,
        "sources_used": card_ids,
        "source_integrated_issue_ids": integrated_issue_ids,
        "quality_flags": quality_flags,
        "provenance": {
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "source_integrated_issue_ids": integrated_issue_ids,
            "quality_flags": quality_flags,
            "analysis_basis": (
                "integrated_issues.id -> card_news.evidence_payload.analysis_package "
                "integrated_issue+analysis+implication+classification+validation"
            ),
        },
    }


def _analysis_package_entries(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for card in cards:
        package = _analysis_package(card)
        analysis = _json_dict(package.get("analysis"))
        implication = _json_dict(package.get("implication"))
        peer = _json_dict(implication.get("peer_implication"))
        skax = _json_dict(implication.get("skax_implication"))
        validation = _json_dict(package.get("validation"))
        integrated = _json_dict(package.get("integrated_issue"))
        entries.append(
            {
                "card_id": card.get("id"),
                "integrated_issue_id": card.get("integrated_issue_id")
                or package.get("integrated_issue_id")
                or integrated.get("integrated_issue_id"),
                "company_label": _company_label(card),
                "main_issue": _first_text(integrated.get("main_issue"), card.get("title")),
                "analysis_summary": _first_text(
                    analysis.get("analysis_summary"),
                    integrated.get("integrated_text"),
                    card.get("title"),
                ),
                "market_signal": _first_text(analysis.get("market_signal")),
                "strategic_meaning": _json_list(analysis.get("strategic_meaning")),
                "peer_meaning": _first_text(
                    peer.get("peer_meaning"),
                    peer.get("capability_change"),
                ),
                "sk_why": _first_text(skax.get("why_important")),
                "sk_impact": _first_text(skax.get("potential_impact")),
                "recommended_actions": _json_list(skax.get("recommended_actions")),
                "analysis_confidence": _safe_float(
                    analysis.get("confidence"),
                    default=-1.0,
                ),
                "implication_confidence": _safe_float(
                    implication.get("confidence"),
                    default=-1.0,
                ),
                "validation_score": _safe_float(validation.get("sc_score"), default=-1.0),
            }
        )
    return entries


def _analysis_basis_block(
    finding: str,
    rationale: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "finding": finding,
        "rationale": rationale,
        "evidence": _basis_evidence(entries),
        "evidence_card_ids": [
            str(entry.get("card_id")) for entry in entries if entry.get("card_id")
        ],
    }


def _basis_evidence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for entry in entries:
        text_value = _first_text(entry.get("market_signal"), entry.get("analysis_summary"))
        if entry.get("card_id") and text_value:
            evidence.append(
                {
                    "card_id": entry["card_id"],
                    "integrated_issue_id": entry.get("integrated_issue_id"),
                    "text": text_value,
                }
            )
    return evidence


def _comparison_finding(entries: list[dict[str, Any]]) -> str:
    phrases = []
    for entry in entries[:3]:
        company = str(entry.get("company_label") or "").strip()
        signal = _brief_sentence(
            _first_text(entry.get("peer_meaning"), entry.get("analysis_summary")),
            max_chars=80,
        )
        if company and signal:
            phrases.append(f"{company}: {signal}")
    if phrases:
        return " / ".join(phrases)
    return _combine_blocks(
        [entry.get("analysis_summary") for entry in entries],
        "기간 내 카드뉴스에서 경쟁사별 움직임이 확인되었습니다.",
        max_items=2,
        max_chars=180,
    )


def _action_details_from_analysis_packages(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not entries:
        return []
    return _executive_action_details_from_entries(entries)


def _executive_action_details_from_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = _basis_evidence(entries)
    evidence_card_ids = _front_evidence_card_ids(entries)
    focus = _briefing_decision_focus(entries)
    focus_clause = _briefing_clause(focus)
    customer_scope = _briefing_customer_scope(entries)
    comparison = _briefing_clause(
        _comparison_finding(entries)
        or _combine_blocks(
            [entry.get("peer_meaning") for entry in entries],
            "고객군별 의사결정 기준 차이",
            max_items=2,
            max_chars=120,
        )
    )
    return [
        {
            "action": _clip_text(
                (
                    f"SK AX는 {customer_scope} 고객군을 우선 공략 범위로 두고, "
                    f"{focus_clause}를 기준으로 오퍼링 책임 조직과 "
                    "리스크 승인 권한을 지정한다."
                ),
                max_chars=300,
            ),
            "why": _briefing_action_reason(entries, fallback=focus),
            "use_case": "사업 우선순위",
            "evidence": evidence,
            "evidence_card_ids": evidence_card_ids,
        },
        {
            "action": _clip_text(
                (
                    "SK AX는 사업 라인별 오퍼링을 하나의 AX 상품으로 묶지 말고 "
                    f"{comparison}에 맞춰 분리 상품화한다. 고객군별 영업 우선순위, "
                    "가격/계약 조건, 보안·데이터 거버넌스 기준을 다르게 둔다."
                ),
                max_chars=300,
            ),
            "why": _briefing_action_reason(
                entries,
                fallback="카드별 고객군과 경쟁 신호가 서로 다른 의사결정 기준을 보여준다.",
            ),
            "use_case": "오퍼링/상품화",
            "evidence": evidence,
            "evidence_card_ids": evidence_card_ids,
        },
        {
            "action": _clip_text(
                (
                    "SK AX는 브리핑 안건을 정보 공유가 아니라 자원 배분 의사결정으로 다룬다. "
                    f"{focus_clause}와 연결된 수주 전환, 규제 일정, 운영 KPI가 확인되면 "
                    "전담 인력, 파트너십 후보, 레퍼런스 확보 예산을 재배분한다."
                ),
                max_chars=300,
            ),
            "why": _combine_blocks(
                [entry.get("market_signal") for entry in entries],
                "기간 내 반복 신호를 사업 자원 배분 기준으로 반영할 필요가 있다.",
                max_items=2,
                max_chars=240,
            ),
            "use_case": "자원 배분/시장 대응",
            "evidence": evidence,
            "evidence_card_ids": evidence_card_ids,
        },
    ]


def _briefing_decision_focus(entries: list[dict[str, Any]]) -> str:
    return _combine_blocks(
        [
            *[entry.get("sk_why") for entry in entries],
            *[entry.get("sk_impact") for entry in entries],
            *[entry.get("market_signal") for entry in entries],
            *[text for entry in entries for text in _json_list(entry.get("strategic_meaning"))],
        ],
        "입력에서 확인된 고객 평가 기준 변화",
        max_items=2,
        max_chars=140,
    )


def _briefing_action_reason(entries: list[dict[str, Any]], *, fallback: str) -> str:
    return _combine_blocks(
        [
            *[entry.get("sk_why") for entry in entries],
            *[entry.get("sk_impact") for entry in entries],
            *[entry.get("analysis_summary") for entry in entries],
        ],
        fallback,
        max_items=2,
        max_chars=240,
    )


def _briefing_customer_scope(entries: list[dict[str, Any]]) -> str:
    text_value = " ".join(
        str(value or "")
        for entry in entries
        for value in (
            entry.get("main_issue"),
            entry.get("analysis_summary"),
            entry.get("market_signal"),
            entry.get("sk_why"),
            entry.get("sk_impact"),
            " ".join(str(item) for item in _json_list(entry.get("strategic_meaning"))),
        )
    )
    scopes: list[str] = []
    if any(token in text_value for token in ("금융", "토큰증권", "디지털자산", "결제", "정산")):
        scopes.append("금융")
    if any(token in text_value for token in ("공공", "행정", "부처", "교육", "학교", "기관")):
        scopes.append("공공/교육")
    if any(token in text_value for token in ("제조", "공장", "설비", "물류", "로봇")):
        scopes.append("제조/운영")
    if any(token in text_value for token in ("보안", "데이터 통제", "프라이빗", "거버넌스")):
        scopes.append("보안·데이터 통제")
    scope = _join_korean(_dedupe_keep_order(scopes[:3]))
    return scope or "입력에서 확인된"


def _briefing_clause(value: object) -> str:
    text_value = re.sub(r"\s+", " ", str(value or "")).strip()
    return text_value.rstrip(".。!！?？")


def _refine_briefing_basis_with_llm(
    *,
    briefing_basis: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    period: dict[str, Any],
    user_context: str | None,
    llm: Any | None,
) -> dict[str, Any]:
    """Use LLM only to synthesize briefing-level judgment from integrated issues.

    The deterministic basis remains the source of truth for ids, provenance, and
    fallback content. The model may rewrite judgment structure, but cannot add
    new evidence ids or facts outside the supplied analysis units.
    """

    if llm is None and not os.getenv("OPENAI_API_KEY"):
        log.info("Briefing basis synthesis skipped: OPENAI_API_KEY is not set")
        return briefing_basis

    context = _briefing_synthesis_context(
        briefing_basis=briefing_basis,
        selected_cards=selected_cards,
        period=period,
        user_context=user_context,
    )
    messages = [
        ("system", _briefing_synthesis_system_prompt()),
        ("human", _briefing_synthesis_user_prompt(context)),
    ]
    try:
        response = (llm or _get_llm()).invoke(messages)
    except Exception as exc:  # pragma: no cover - external API safety net
        log.warning("Briefing basis synthesis failed | error=%s", exc)
        return briefing_basis

    parsed = _parse_json_object(getattr(response, "content", response))
    if not parsed:
        return briefing_basis
    issues = _briefing_synthesis_quality_issues(parsed, selected_cards)
    if issues:
        revision_messages = [
            ("system", _briefing_synthesis_system_prompt()),
            ("human", _briefing_synthesis_revision_prompt(context, parsed, issues)),
        ]
        try:
            revision_response = (llm or _get_llm()).invoke(revision_messages)
        except Exception as exc:  # pragma: no cover - external API safety net
            log.warning("Briefing basis synthesis revision failed | error=%s", exc)
        else:
            revised = _parse_json_object(getattr(revision_response, "content", revision_response))
            if revised:
                parsed = revised
        remaining_issues = _briefing_synthesis_quality_issues(parsed, selected_cards)
        if remaining_issues:
            log.info(
                "Briefing basis synthesis rejected | issues=%s",
                remaining_issues,
            )
            return briefing_basis
    return _merge_briefing_basis_synthesis(briefing_basis, parsed, selected_cards)


def _briefing_synthesis_revision_prompt(
    context: dict[str, Any],
    draft: dict[str, Any],
    issues: list[str],
) -> str:
    return "\n".join(
        [
            "# Task",
            "아래 draft_basis에서 감지된 품질 이슈만 고쳐 다시 JSON 객체로 반환하세요.",
            "",
            "# Detected Issues",
            json.dumps(issues, ensure_ascii=False, indent=2),
            "",
            "# Rules",
            "- Output Schema는 최초 요청과 동일합니다.",
            "- 입력 source_card_ids 밖의 id를 쓰지 않습니다.",
            "- integrated_issues/analysis/implication 근거 밖의 내용을 만들지 않습니다.",
            "- immediate/watch 분류 이유가 드러나게 reason을 고칩니다.",
            "",
            "# Source Analysis Units",
            json.dumps(context.get("analysis_units") or [], ensure_ascii=False, indent=2),
            "",
            "# Draft Basis",
            json.dumps(draft, ensure_ascii=False, indent=2),
        ]
    )


def _briefing_synthesis_quality_issues(
    draft: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> list[str]:
    allowed_ids = {str(card.get("id")) for card in selected_cards if card.get("id")}
    issues: list[str] = []
    if not _first_text(draft.get("executive_summary"), draft.get("briefing_insight")):
        issues.append("executive_summary 또는 briefing_insight가 비어 있습니다.")
    invalid_ids = _invalid_synthesis_card_ids(draft, allowed_ids)
    if invalid_ids:
        issues.append(f"입력 source_card_ids 밖의 id가 사용되었습니다: {invalid_ids}")
    if not _json_list(draft.get("immediate_trends")) and not _json_list(draft.get("watch_trends")):
        issues.append("immediate_trends 또는 watch_trends 중 최소 하나가 필요합니다.")
    return issues


def _invalid_synthesis_card_ids(value: object, allowed_ids: set[str]) -> list[str]:
    invalid: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"evidence_card_ids", "related_card_ids"}:
                ids = [str(card_id) for card_id in _json_list(item) if str(card_id).strip()]
                invalid.extend(card_id for card_id in ids if card_id not in allowed_ids)
            elif key == "related_card_id":
                card_id = str(item or "").strip()
                if card_id and card_id not in allowed_ids:
                    invalid.append(card_id)
            else:
                invalid.extend(_invalid_synthesis_card_ids(item, allowed_ids))
    elif isinstance(value, list):
        for item in value:
            invalid.extend(_invalid_synthesis_card_ids(item, allowed_ids))
    return _dedupe_keep_order(invalid)


def _merge_briefing_basis_synthesis(
    base: dict[str, Any],
    synthesis: dict[str, Any],
    selected_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(base)
    allowed_ids = {str(card.get("id")) for card in selected_cards if card.get("id")}
    summary = _first_text(synthesis.get("executive_summary"), synthesis.get("briefing_insight"))
    if summary:
        updated["briefing_insight"] = _brief_sentences(
            summary,
            max_sentences=2,
            max_chars=240,
        )
    for key in (
        "lead",
        "core_change",
        "common_pattern",
        "comparison_point",
        "hidden_conclusion",
        "strategy_implication",
    ):
        block = _normalize_synthesis_block(
            synthesis.get(key),
            fallback=updated.get(key),
            allowed_ids=allowed_ids,
        )
        if block:
            updated[key] = block
    for key in (
        "recommended_action_basis",
        "recommended_actions",
        "immediate_trends",
        "watch_trends",
        "sections",
        "evidence_summary",
    ):
        values = _json_list(synthesis.get(key))
        if values:
            updated[key] = _sanitize_synthesis_list(values, allowed_ids=allowed_ids)
    action_details = _normalize_synthesis_action_details(
        synthesis.get("action_details"),
        allowed_ids=allowed_ids,
    )
    if action_details:
        updated["action_details"] = action_details
        updated["recommended_actions"] = [
            str(item.get("action") or "").strip()
            for item in action_details
            if str(item.get("action") or "").strip()
        ]
    confidence = _safe_float(synthesis.get("confidence"), default=-1.0)
    if 0 <= confidence <= 1:
        updated["confidence"] = round(confidence, 2)
    provenance = _json_dict(updated.get("provenance"))
    provenance["briefing_synthesis_prompt_version"] = _BRIEFING_SYNTHESIS_PROMPT_VERSION
    provenance["briefing_synthesis_model"] = _LLM_MODEL
    updated["provenance"] = provenance
    return updated


def _normalize_synthesis_block(
    value: object,
    *,
    fallback: object,
    allowed_ids: set[str],
) -> dict[str, Any]:
    source = _json_dict(value)
    fallback_block = _json_dict(fallback)
    finding = _brief_sentences(
        _first_text(source.get("finding"), source.get("title"), source.get("summary")),
        max_sentences=2,
        max_chars=240,
    )
    if not finding:
        return fallback_block
    evidence_ids = _valid_card_ids(
        source.get("evidence_card_ids") or source.get("related_card_ids"),
        allowed_ids=allowed_ids,
        fallback=fallback_block.get("evidence_card_ids"),
    )
    return {
        **fallback_block,
        "finding": finding,
        "rationale": _brief_sentences(
            _first_text(source.get("rationale"), source.get("reason"), source.get("description")),
            max_sentences=3,
            max_chars=320,
        )
        or fallback_block.get("rationale")
        or finding,
        "evidence_card_ids": evidence_ids,
    }


def _sanitize_synthesis_list(values: list[Any], *, allowed_ids: set[str]) -> list[Any]:
    sanitized: list[Any] = []
    for value in values:
        if isinstance(value, dict):
            current = copy.deepcopy(value)
            if "evidence_card_ids" in current:
                current["evidence_card_ids"] = _valid_card_ids(
                    current.get("evidence_card_ids"),
                    allowed_ids=allowed_ids,
                )
            if "related_card_ids" in current:
                current["related_card_ids"] = _valid_card_ids(
                    current.get("related_card_ids"),
                    allowed_ids=allowed_ids,
                )
            if "related_card_id" in current:
                card_id = str(current.get("related_card_id") or "").strip()
                if card_id not in allowed_ids:
                    current.pop("related_card_id", None)
            sanitized.append(current)
            continue
        text_value = str(value or "").strip()
        if text_value:
            sanitized.append(text_value)
    return sanitized


def _normalize_synthesis_action_details(
    value: object,
    *,
    allowed_ids: set[str],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        action = _brief_sentences(item.get("action"), max_sentences=2, max_chars=280)
        if not action:
            continue
        actions.append(
            {
                "action": action,
                "why": _brief_sentences(
                    item.get("why") or item.get("reason"),
                    max_sentences=2,
                    max_chars=260,
                ),
                "use_case": _first_text(item.get("use_case"), "사업 우선순위"),
                "evidence_card_ids": _valid_card_ids(
                    item.get("evidence_card_ids"),
                    allowed_ids=allowed_ids,
                ),
            }
        )
    return actions[:3]


def _valid_card_ids(
    value: object,
    *,
    allowed_ids: set[str],
    fallback: object = None,
) -> list[str]:
    ids = [
        card_id
        for card_id in (str(item).strip() for item in _json_list(value))
        if card_id and card_id in allowed_ids
    ]
    if not ids and fallback is not None:
        ids = [
            card_id
            for card_id in (str(item).strip() for item in _json_list(fallback))
            if card_id and card_id in allowed_ids
        ]
    return _dedupe_keep_order(ids)


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
    messages = [
        ("system", _display_copy_system_prompt()),
        ("human", _display_copy_user_prompt(context)),
    ]
    try:
        response = (llm or _get_llm()).invoke(messages)
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
            revision_response = (llm or _get_llm()).invoke(revision_messages)
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


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text_value = str(value or "").strip()
    if not text_value:
        return {}
    try:
        parsed = json.loads(text_value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _company_issue_sentence_from_result(result: dict[str, Any]) -> str:
    phrases = _company_issue_phrases_from_result(result)
    if not phrases:
        return ""
    joined = _join_korean(phrases[:3])
    return f"근거로는 {joined}{_subject_particle(joined)} 함께 확인됩니다."


def _company_signal_map_from_result(result: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for detail in _json_list(result.get("hidden_details")):
        if not isinstance(detail, dict):
            continue
        package = _json_dict(detail.get("analysis_package"))
        company = _first_text(
            _nested_get(package, "implication", "peer_implication", "company_name_ko"),
            detail.get("company_label"),
        )
        signal = _first_text(
            _business_signal_phrase(package),
            _nested_get(package, "analysis", "market_signal"),
            _nested_get(package, "analysis", "analysis_summary"),
        )
        if company and signal:
            pairs.append((company, _brief_noun_phrase(signal, max_chars=76)))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _first_company_signal(
    pairs: list[tuple[str, str]],
    tokens: tuple[str, ...],
) -> str:
    for company, signal in pairs:
        if any(token in signal for token in tokens):
            return f"{company}의 {signal}"
    return ""


def _company_issue_phrases_from_result(result: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for detail in _json_list(result.get("hidden_details")):
        if not isinstance(detail, dict):
            continue
        package = _json_dict(detail.get("analysis_package"))
        company = _first_text(
            _nested_get(package, "implication", "peer_implication", "company_name_ko"),
            detail.get("company_label"),
        )
        issue = _first_text(
            _business_signal_phrase(package),
            _nested_get(package, "integrated_issue", "main_issue"),
            _nested_get(package, "analysis", "analysis_summary"),
        )
        issue = _brief_noun_phrase(issue, max_chars=76)
        if company and issue.startswith(company):
            issue = issue.removeprefix(company).lstrip("의 ·:-")
        if company and issue:
            phrases.append(f"{company}의 {issue}")
    return _dedupe_keep_order(phrases)


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


def _average_confidence(selected_cards: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for card in selected_cards:
        package = _analysis_package(card)
        for value in (
            _nested_get(package, "analysis", "confidence"),
            _nested_get(package, "implication", "confidence"),
            _nested_get(package, "validation", "sc_score"),
            _nested_get(package, "integrated_issue", "confidence"),
        ):
            parsed = _safe_float(value, default=-1.0)
            if 0 <= parsed <= 1:
                values.append(parsed)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _action_values(value: object) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in _json_list(value):
        if isinstance(item, dict):
            actions.append(item)
            continue
        text_value = str(item or "").strip()
        if text_value:
            actions.append({"action": text_value})
    return actions


def _aggregate_text(
    values: list[object],
    fallback: str,
    *,
    max_items: int = 2,
    max_chars: int = 220,
) -> str:
    texts = _unique_texts(values)
    if not texts:
        return _brief_sentence(fallback, max_chars=max_chars)
    if len(texts) == 1:
        return _brief_sentence(texts[0], max_chars=max_chars)
    joined = " ".join(_brief_sentence(text, max_chars=110) for text in texts[:max_items])
    return _clip_text(joined, max_chars=max_chars)


def _combine_blocks(
    values: list[object],
    fallback: str,
    *,
    max_items: int = 2,
    max_chars: int = 240,
) -> str:
    seen: set[str] = set()
    blocks: list[str] = []
    for value in values:
        text_value = str(value or "").strip()
        key = re.sub(r"\s+", " ", text_value)
        if not key or key in seen:
            continue
        seen.add(key)
        blocks.append(key)
        if len(blocks) >= max_items:
            break
    if not blocks:
        return _brief_sentence(fallback, max_chars=max_chars)
    return _clip_text(" ".join(blocks), max_chars=max_chars)


def _first_distinct_text(values: list[object]) -> str:
    texts = _unique_texts(values)
    return texts[0] if texts else ""


def _unique_texts(values: list[object]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for value in values:
        text_value = _limit_sentences(str(value or "").strip(), max_sentences=1)
        key = re.sub(r"\s+", " ", text_value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        texts.append(key)
    return texts


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


def _split_trend_cards(
    selected_cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not selected_cards:
        return [], []
    immediate: list[dict[str, Any]] = []
    watch: list[dict[str, Any]] = []
    for card in selected_cards:
        importance = str(card.get("importance") or "").lower()
        score = _safe_float(card.get("importance_score"), default=0.0)
        if importance in {"high", "critical"} or score >= 0.75:
            immediate.append(card)
        else:
            watch.append(card)
    if not immediate:
        immediate = selected_cards[: min(2, len(selected_cards))]
        immediate_ids = {card.get("id") for card in immediate}
        watch = [card for card in selected_cards if card.get("id") not in immediate_ids]
    return immediate[:3], watch[:3]


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
                "title": _brief_sentence(item.get("title") or trend.get("title"), max_chars=120),
                "reason": _brief_sentences(
                    item.get("reason") or trend.get("reason"),
                    max_sentences=2,
                    max_chars=220,
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


def _trend_title(card: dict[str, Any]) -> str:
    package = _analysis_package(card)
    issue = _json_dict(package.get("integrated_issue"))
    headline = _first_text(
        issue.get("main_issue"),
        issue.get("headline"),
        issue.get("one_line_summary"),
        card.get("title"),
    )
    company = _company_label(card)
    if company and headline and company not in headline:
        headline = f"{company} - {headline}"
    return _brief_sentence(headline, max_chars=120)


def _trend_reason(card: dict[str, Any], *, role: Literal["immediate", "watch"]) -> str:
    package = _analysis_package(card)
    analysis = _json_dict(package.get("analysis"))
    implication = _json_dict(package.get("implication"))
    skax = _json_dict(implication.get("skax_implication"))
    if role == "immediate":
        reason = _first_text(
            skax.get("why_important"),
            analysis.get("market_signal"),
            analysis.get("analysis_summary"),
        )
        fallback = "경쟁 구도나 고객 평가 기준에 바로 반영할 신호입니다."
    else:
        reason = _first_text(
            analysis.get("impact_reason"),
            skax.get("potential_impact"),
            analysis.get("analysis_summary"),
        )
        fallback = "후속 수주, 고객 확산, 실행 근거를 이어서 확인할 필요가 있습니다."
    return _brief_sentences(reason or fallback, max_sentences=2, max_chars=220)


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
        title = _brief_sentence(item.get("title"), max_chars=80)
        if not title:
            continue
        related_ids = _valid_card_ids(
            item.get("related_card_ids"),
            allowed_ids=allowed_ids,
            fallback=source_card_ids,
        )
        bullets = [
            _brief_sentence(bullet, max_chars=120)
            for bullet in _json_list(item.get("bullets"))
            if str(bullet or "").strip()
        ][:4]
        sections.append(
            {
                "title": title,
                "summary": _brief_sentences(
                    item.get("summary") or executive_summary,
                    max_sentences=2,
                    max_chars=240,
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
            "summary": _brief_sentences(executive_summary, max_sentences=2, max_chars=240),
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
        _brief_sentence(item, max_chars=120)
        for item in _dedupe_keep_order([str(value or "").strip() for value in candidates])
        if item
    ][:4]


def _briefing_trends_summary(trends: list[dict[str, Any]]) -> str:
    reasons: list[object] = [
        str(item.get("reason") or "").strip() for item in trends if item.get("reason")
    ]
    return _combine_blocks(
        reasons,
        "통합 이슈 근거 기준으로 우선순위를 나눠 정리했습니다.",
        max_items=2,
        max_chars=220,
    )


def _evidence_summary_from_basis(
    briefing_basis: dict[str, Any],
    *,
    selected_cards: list[dict[str, Any]],
    source_card_ids: list[str],
) -> list[str]:
    provided = [
        _brief_sentence(item, max_chars=140)
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
            "summary": _brief_sentence(executive_summary, max_chars=160),
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
        "summary": _brief_sentence(summary or executive_summary, max_chars=180),
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


def _frontend_daily_title(period: dict[str, Any]) -> str:
    target = period["date_from"]
    return f"{target.year}년 {target.month}월 {target.day}일 Peer Intelligence 일간 브리핑"


def _frontend_weekly_title(period: dict[str, Any]) -> str:
    start = period["date_from"]
    week_no = ((start.day - 1) // 7) + 1
    return f"{start.year}년 {start.month}월 {week_no}주차 Peer Intelligence 주간 브리핑"


def _frontend_section_item(trend: dict[str, Any]) -> dict[str, str]:
    return {
        "headline": _brief_sentence(trend.get("title"), max_chars=120),
        "source": _trend_source_label(trend),
    }


def _trend_source_label(trend: dict[str, Any]) -> str:
    source_name = _first_text(trend.get("source_name"), "통합 이슈 근거")
    published_at = _parse_datetime(trend.get("published_at"))
    if published_at:
        return f"{source_name}, {published_at.astimezone(KST):%Y.%m.%d %H:%M}"
    return source_name


def _primary_source(card: dict[str, Any]) -> dict[str, Any]:
    sources = _json_list(card.get("sources")) or _json_list(
        _nested_get(card, "evidence_payload", "source_links")
    )
    for source in sources:
        if isinstance(source, dict):
            return source
    return {}


def _source_name(source: dict[str, Any]) -> str:
    return _first_text(
        source.get("source_name"),
        source.get("publisher"),
        source.get("provider"),
        source.get("name"),
        "통합 이슈 근거",
    )


def _source_published_at(card: dict[str, Any], source: dict[str, Any]) -> str:
    value = _first_text(source.get("published_at"), card.get("basis_at"), card.get("created_at"))
    parsed = _parse_datetime(value)
    return _iso_or_none(parsed) or value


def _evidence_source_names(selected_cards: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for card in selected_cards:
        for source in _json_list(card.get("sources")):
            if isinstance(source, dict):
                name = _source_name(source)
                if name and name != "통합 이슈 근거":
                    names.append(name)
    return _dedupe_keep_order(names)


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
        max_chars=220,
    )
    key_summary = _brief_sentence(
        _display_core_title(selected_cards, briefing_basis),
        max_chars=140,
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


def _front_briefing_label(briefing_type: BriefingType) -> str:
    return {"daily": "일간", "weekly": "주간", "monthly": "월간"}[briefing_type]


def _front_briefing_count(briefing_type: BriefingType) -> int:
    return {"daily": 4, "weekly": 6, "monthly": 8}[briefing_type]


def _front_signal_title(signal_cards: list[dict[str, Any]]) -> str:
    for card in signal_cards:
        title = _first_text(card.get("title"))
        if title:
            return title
    return ""


def _front_signal_summary(signal_cards: list[dict[str, Any]]) -> str:
    for card in signal_cards:
        summary = _first_text(card.get("summary"), card.get("reason"))
        if summary:
            return summary
    return ""


def _front_card_news_item(card: dict[str, Any]) -> dict[str, Any]:
    source = _primary_source(card)
    source_url = _first_text(source.get("url"), source.get("link"), source.get("source_url"), "#")
    source_name = _source_name(source)
    title = _card_display_title(card)
    summary = _front_card_summary_lines(card)
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
        "detailDescription": _card_summary(card),
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


def _front_briefing_signal_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index, item in enumerate(_json_list(result.get("key_change_cards")), 1):
        if not isinstance(item, dict):
            continue
        title = _first_text(item.get("title"), item.get("description"))
        summary = _first_text(item.get("description"), item.get("summary"))
        reason = _first_text(item.get("why_important"), summary)
        if not title and not summary and not reason:
            continue
        cards.append(
            {
                "label": _first_text(
                    item.get("display_label"),
                    item.get("peer_label"),
                    f"핵심 변화 {index}",
                ),
                "title": title,
                "summary": summary,
                "reason": reason,
                "relatedCardIds": [
                    str(card_id)
                    for card_id in _json_list(item.get("evidence_card_ids"))
                    if str(card_id).strip()
                ]
                or [str(card_id) for card_id in _json_list(result.get("related_card_ids"))],
            }
        )
    return cards


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


def _front_flow_detail(item: dict[str, Any]) -> str:
    title = _first_text(item.get("title"))
    description = _first_text(item.get("description"))
    if title and description:
        return f"{title} {description}"
    return title or description


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
            "title": _brief_sentence(title, max_chars=140),
            "reason": _brief_sentence(reason, max_chars=180),
        }
        for title, reason in candidates
        if _first_text(title)
    ][:4]


def _front_briefing_benchmark(result: dict[str, Any]) -> list[dict[str, str]]:
    basis = _json_dict(result.get("briefing_basis"))
    candidates = [
        (
            _block_text(basis.get("strategy_implication"), "finding"),
            _block_text(basis.get("strategy_implication"), "rationale"),
        ),
        *[(str(item), "") for item in _json_list(basis.get("recommended_actions"))],
        *[(str(item), "") for item in _json_list(result.get("evidence_summary"))],
    ]
    return [
        {
            "title": _brief_sentence(title, max_chars=140),
            "reason": _brief_sentence(reason, max_chars=180),
        }
        for title, reason in candidates
        if _first_text(title)
    ][:4]


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


def _recommended_action_pairs(selected_cards: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for card in selected_cards:
        package = _analysis_package(card)
        skax = _json_dict(_nested_get(package, "implication", "skax_implication"))
        why = _first_text(skax.get("why_important"), skax.get("potential_impact"))
        for action in _json_list(skax.get("recommended_actions")):
            action_text = str(action or "").strip()
            if action_text:
                pairs.append((action_text, why))
    return _dedupe_action_pairs(pairs)


def _dedupe_action_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for action, why in pairs:
        key = re.sub(r"\s+", " ", action).strip()
        if key and key not in seen:
            seen.add(key)
            result.append((action, why))
    return result


def _action_use_case(_action: str) -> str:
    return "SK AX 관점"


def _display_core_title(
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> str:
    _ = selected_cards
    core_value = briefing_basis.get("core_change")
    core: dict[str, Any] = core_value if isinstance(core_value, dict) else {}
    return _first_text(core.get("finding"), briefing_basis.get("briefing_insight"))


def _display_core_summary(briefing_basis: dict[str, Any]) -> str:
    return _block_text(briefing_basis.get("core_change"), "rationale")


def _display_flow_steps(
    briefing_basis: dict[str, Any],
    default_evidence: list[Any],
) -> list[dict[str, Any]]:
    evidence_ids = [str(item) for item in default_evidence if str(item).strip()]
    steps = [
        (
            "관찰된 변화",
            _block_text(briefing_basis.get("common_pattern"), "finding"),
        ),
        (
            "평가축의 이동",
            _block_text(briefing_basis.get("comparison_point"), "finding"),
        ),
        (
            "경쟁 구도 영향",
            _block_text(briefing_basis.get("hidden_conclusion"), "finding"),
        ),
        (
            "전략 시사",
            _block_text(briefing_basis.get("strategy_implication"), "finding")
            or _action_text(briefing_basis.get("action_details"), "action"),
        ),
    ]
    return [
        {
            "seq": index,
            "label": label,
            "one_liner": sentence,
            "evidence_card_ids": evidence_ids,
            "evidence_refs": evidence_ids,
            "langfuse_observation_id": None,
        }
        for index, (label, sentence) in enumerate(steps, 1)
        if sentence
    ]


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
        finding = _clip_text(str(block.get("finding") or "").strip(), max_chars=140)
        if not finding:
            continue
        items.append(
            (
                label,
                finding,
                _brief_sentences(block.get("rationale"), max_sentences=2, max_chars=180),
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


def _display_sk_ax_view(
    briefing_basis: dict[str, Any],
    default_evidence: list[Any],
) -> list[dict[str, Any]]:
    evidence_ids = [str(item) for item in default_evidence if str(item).strip()]
    actions = briefing_basis.get("action_details")
    if not isinstance(actions, list):
        return []
    return [
        {
            "seq": index,
            "use_case": action.get("use_case") or _action_use_case(str(action.get("action") or "")),
            "title": _brief_sentence(action.get("action")),
            "description": _brief_sentence(action.get("why")),
            "evidence_card_ids": action.get("evidence_card_ids") or evidence_ids,
        }
        for index, action in enumerate(actions, 1)
        if isinstance(action, dict) and str(action.get("action") or "").strip()
    ][:_MAX_SKAX_ITEMS]


def _display_sk_ax_title(selected_cards: list[dict[str, Any]]) -> str:
    for action, _why in _recommended_action_pairs(selected_cards):
        title = _brief_sentence(action)
        if title:
            return title
    return ""


def _display_evidence_ids(
    selected_cards: list[dict[str, Any]],
    default_evidence: list[Any],
) -> list[str]:
    evidence_ids = [str(item) for item in default_evidence if str(item).strip()]
    if evidence_ids:
        return evidence_ids
    return [str(card["id"]) for card in selected_cards if card.get("id")]


def _join_korean(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]}{_and_particle(cleaned[0])} {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, {cleaned[-1]}"


def _and_particle(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "와"
    code = ord(text[-1])
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 == 0:
        return "와"
    return "과"


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
        max_chars=140,
    )
    market_summary = _select_distinct_sentence(
        [
            _market_signal_reason_summary(entries),
            *_market_signal_summary_candidates(entries, briefing_basis),
        ],
        avoid=[lead_text, market_title],
        fallback="선택된 카드들의 분석 결과에서 공통 수요와 평가 기준 변화가 확인되었습니다.",
        max_chars=180,
    )
    market_so_what = _select_distinct_sentence(
        _so_what_candidates(entries, briefing_basis),
        avoid=[market_title, market_summary],
        fallback=(
            "이 변화는 고객 제안과 경쟁사 대응에서 확인해야 할 평가 기준을 바꿀 수 있습니다."
        ),
        max_chars=180,
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
        max_chars=140,
    )
    competitor_summary = _select_distinct_sentence(
        [
            _competitor_move_flow_summary(entries),
            *_competitor_move_summary_candidates(entries, briefing_basis, selected_cards),
        ],
        avoid=[market_title, market_summary, market_so_what, competitor_title],
        fallback=_competitor_move_summary(selected_cards),
        max_chars=180,
    )
    competitor_so_what = _select_distinct_sentence(
        _competitor_so_what_candidates(entries, briefing_basis),
        avoid=[market_title, market_summary, market_so_what, competitor_title, competitor_summary],
        fallback=(
            "경쟁사 움직임을 함께 보면 개별 이슈보다 "
            "경쟁 방식과 고객 설득 기준의 변화가 더 분명해집니다."
        ),
        max_chars=180,
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


def _key_change_source_entries(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for card in cards:
        package = _analysis_package(card)
        classification = _json_dict(package.get("classification"))
        integrated = _json_dict(package.get("integrated_issue"))
        analysis = _json_dict(package.get("analysis"))
        implication = _json_dict(package.get("implication"))
        peer = _json_dict(implication.get("peer_implication"))
        skax = _json_dict(implication.get("skax_implication"))
        entries.append(
            {
                "card_id": card.get("id"),
                "company_label": _company_label(card),
                "sector": _first_text(
                    classification.get("sector"),
                    _first_from_list(classification.get("sectors")),
                ),
                "event_type": _first_text(classification.get("event_type")),
                "main_issue": _first_text(integrated.get("main_issue"), card.get("title")),
                "integrated_text": _first_text(integrated.get("integrated_text")),
                "analysis_summary": _first_text(analysis.get("analysis_summary")),
                "market_signal": _first_text(analysis.get("market_signal")),
                "strategic_meaning": _json_list(analysis.get("strategic_meaning")),
                "impact_reason": _first_text(analysis.get("impact_reason")),
                "peer_meaning": _first_text(
                    peer.get("peer_meaning"),
                    peer.get("capability_change"),
                ),
                "sk_why": _first_text(skax.get("why_important")),
                "sk_impact": _first_text(skax.get("potential_impact")),
                "opportunities": _json_list(skax.get("opportunities")),
                "threats": _json_list(skax.get("threats")),
                "recommended_actions": _json_list(skax.get("recommended_actions")),
            }
        )
    return entries


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


def _competitor_move_title_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[object]:
    return [
        _block_text(briefing_basis.get("comparison_point"), "finding"),
        _combine_company_signals(entries, "peer_meaning"),
        _combine_company_signals(entries, "analysis_summary"),
        _combine_blocks([entry.get("main_issue") for entry in entries], "", max_items=3),
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


def _competitor_so_what_candidates(
    entries: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[object]:
    return [
        _competitor_importance_sentence(entries),
        _block_text(briefing_basis.get("strategy_implication"), "finding"),
        _block_text(briefing_basis.get("strategy_implication"), "rationale"),
        *[entry.get("sk_why") for entry in entries],
        *[entry.get("sk_impact") for entry in entries],
        *[text for entry in entries for text in _json_list(entry.get("recommended_actions"))],
        *[entry.get("impact_reason") for entry in entries],
    ]


def _repeated_signal_candidate(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        return "여러 카드에서 같은 방향의 시장 변화와 수요 신호가 함께 확인되었습니다."
    return "선택된 카드에서 시장 변화 신호가 확인되었습니다."


def _competitor_move_group_title(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        return "경쟁사들은 감지된 수요 변화에 맞춰 사업 방향과 실행 메시지를 구체화하고 있습니다."
    return "경쟁사들은 기간 내 감지된 변화에 맞춰 사업과 기술 메시지를 조정하고 있습니다."


def _market_signal_reason_summary(entries: list[dict[str, Any]]) -> str:
    reasons = []
    for entry in entries[:3]:
        reason = _first_text(
            entry.get("impact_reason"),
            _first_from_list(entry.get("strategic_meaning")),
            entry.get("analysis_summary"),
        )
        reason = _strip_terminal_punctuation(_brief_sentence(reason, max_chars=76))
        if reason:
            reasons.append(reason)
    reasons = _dedupe_keep_order(reasons)
    if len(reasons) >= 2:
        return "구체 근거로는 " + " / ".join(reasons[:2])
    if reasons:
        return f"구체 근거를 보면 이 변화는 단발 이슈가 아니라 {reasons[0]} 흐름과 연결됩니다."
    if len(entries) >= 2:
        return (
            "선택된 카드들의 분석 결과가 같은 방향의 수요 변화와 "
            "평가 기준 변화를 함께 가리키고 있습니다."
        )
    return "선택된 카드의 분석 결과가 이후 수요 변화와 평가 기준을 확인할 시장 신호로 해석됩니다."


def _competitor_move_flow_summary(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        return (
            "선택된 카드들은 경쟁사들이 감지된 수요 변화에 맞춰 기술·사업 역량을 "
            "실행 근거와 고객 설득 메시지로 연결하고 있음을 보여줍니다."
        )
    signal = _first_text(
        _first_from_list([entry.get("peer_meaning") for entry in entries]),
        _first_from_list([entry.get("analysis_summary") for entry in entries]),
    )
    return _brief_sentence(signal, max_chars=180)


def _cross_card_importance_sentence(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        return (
            "따라서 임원 의사결정에서는 기능 설명보다 고객이 평가할 운영 성과, "
            "리스크 감소, 실행 근거를 사업 우선순위 기준으로 먼저 봐야 합니다."
        )
    return "이 변화는 수요 확산 여부와 실제 성과 근거에 따라 자원 배분을 조정해야 합니다."


def _competitor_importance_sentence(entries: list[dict[str, Any]]) -> str:
    if len(entries) >= 2:
        return (
            "따라서 경쟁사 메시지가 실제 수주, 고객 사례, 성과 지표로 "
            "이어지는지까지 오퍼링과 시장 대응 우선순위에서 함께 봐야 합니다."
        )
    return "이 움직임은 같은 방향의 경쟁 신호가 반복될 때 고객군 우선순위를 바꿀 수 있습니다."


def _strip_terminal_punctuation(value: str) -> str:
    return str(value or "").rstrip(" .。!?！？")


def _combine_company_signals(entries: list[dict[str, Any]], key: str) -> str:
    phrases = []
    for entry in entries[:3]:
        company = str(entry.get("company_label") or "").strip()
        signal = _brief_sentence(entry.get(key), max_chars=80)
        if company and signal:
            phrases.append(f"{company}: {signal}")
    return " / ".join(phrases)


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


def _generic_distinct_sentence(avoid: list[str], *, max_chars: int) -> str:
    candidates = [
        "이 신호는 이후 고객 평가 기준과 경쟁사 대응 방향을 함께 확인해야 하는 변화입니다.",
        "여러 카드의 근거를 함께 보면 개별 사건보다 시장의 우선순위 변화가 더 중요해집니다.",
        "이 변화는 후속 카드에서 수요 확산 여부와 실제 성과 근거를 계속 확인해야 합니다.",
    ]
    for candidate in candidates:
        if not _is_too_similar(candidate, avoid):
            return _brief_sentence(candidate, max_chars=max_chars)
    return _brief_sentence(candidates[0], max_chars=max_chars)


def _competitor_move_summary(selected_cards: list[dict[str, Any]]) -> str:
    phrases: list[str] = []
    for card in selected_cards[:3]:
        signal = _brief_sentence(_card_summary(card), max_chars=80)
        if signal:
            phrases.append(signal)
    if phrases:
        return " / ".join(phrases)
    return ""


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


def _card_summary(card: dict[str, Any]) -> str:
    package = _analysis_package(card)
    package_summary = _first_text(
        _nested_get(package, "analysis", "analysis_summary"),
        _nested_get(package, "integrated_issue", "integrated_text"),
    )
    if package_summary:
        return package_summary
    line_value = card.get("summary_lines")
    lines: list[Any] = line_value if isinstance(line_value, list) else []
    return " ".join(str(line).strip() for line in lines[:2] if str(line).strip())


def _card_display_title(card: dict[str, Any]) -> str:
    package = _analysis_package(card)
    return _brief_sentence(
        _first_text(
            _nested_get(package, "integrated_issue", "main_issue"),
            card.get("title"),
        )
    )


def _briefing_lead(
    period: dict[str, Any],
    key_summary: str,
    selected_cards: list[dict[str, Any]] | None = None,
) -> str:
    _ = selected_cards
    if not key_summary:
        return f"{period['label']} 동안 확인된 카드뉴스 기반 흐름입니다."
    return _brief_sentence(key_summary, max_chars=180)


def _period_label(briefing_type: BriefingType, start: date, end: date) -> str:
    if briefing_type == "daily":
        return f"{start:%Y. %m. %d} 일간"
    if briefing_type == "weekly":
        return f"{start:%Y. %m. %d}~{end:%m. %d} 주간"
    if briefing_type == "monthly":
        return f"{start:%Y. %m} 월간"
    raise ValueError(f"unsupported briefing_type: {briefing_type}")


def _briefing_id(briefing_type: BriefingType, start: date) -> str:
    return f"BR-{briefing_type.upper()}-{start:%Y%m%d}"


def _flow_item(seq: int, label: str, one_liner: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "label": label,
        "one_liner": one_liner,
        "evidence_card_ids": [],
        "evidence_refs": [],
        "langfuse_observation_id": None,
    }


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
        max_chars=220,
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


def _block_text(value: object, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get(key) or "").strip()


def _action_text(value: object, key: str) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        if isinstance(item, dict) and str(item.get(key) or "").strip():
            return str(item.get(key)).strip()
    return ""


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
                "evidence_text": _evidence_texts(package),
                "analysis_package": package,
            }
        )
    return details


def _evidence_texts(package: dict[str, Any]) -> list[str]:
    integrated = _json_dict(package.get("integrated_issue"))
    fact_basis = _json_list(integrated.get("fact_basis"))
    ledger = _json_list(integrated.get("evidence_ledger"))
    texts = []
    for item in [*fact_basis, *ledger]:
        if isinstance(item, dict) and str(item.get("evidence_text") or "").strip():
            texts.append(str(item["evidence_text"]).strip())
    return texts[:5]


def _brief_sentence(value: object, max_chars: int = 120) -> str:
    text_value = _limit_sentences(str(value or "").strip(), max_sentences=1)
    return _clip_text(text_value, max_chars=max_chars)


def _brief_sentences(value: object, *, max_sentences: int, max_chars: int) -> str:
    text_value = _limit_sentences(str(value or "").strip(), max_sentences=max_sentences)
    return _clip_text(text_value, max_chars=max_chars)


def _limit_sentences(value: str, max_sentences: int) -> str:
    text_value = " ".join(str(value or "").split())
    if not text_value:
        return ""
    sentences = re.split(r"(?<=[.!?。！？])\s+", text_value)
    selected = [sentence.strip() for sentence in sentences if sentence.strip()][:max_sentences]
    return " ".join(selected) if selected else text_value


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


def _interpretation_reasoning_summary(
    result: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "display_title": "해석 흐름 — 관찰부터 시사까지",
        "disclosure_level": "summarized_intermediate_artifacts",
        "note": (
            "원시 모델 사고 과정이 아니라 integrated_issues/analysis/implication "
            "근거에서 화면 결론까지의 중간 산출물을 요약한 감사용 trace입니다."
        ),
        "source_card_ids": _front_evidence_card_ids(entries),
        "source_integrated_issue_ids": _dedupe_keep_order(
            [
                str(entry.get("integrated_issue_id"))
                for entry in entries
                if entry.get("integrated_issue_id")
            ]
        )
        or _json_list(result.get("source_integrated_issue_ids")),
        "source_mode": _nested_get(result, "provenance", "source_mode"),
        "basis": _nested_get(result, "provenance", "briefing_analysis_basis"),
    }


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
                "title": _brief_sentence(item.get("title"), max_chars=120),
                "description": _brief_sentences(
                    item.get("description"),
                    max_sentences=2,
                    max_chars=220,
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
                    max_chars=220,
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


def _reasoning_source_bundle_sentence(source_inputs: list[dict[str, Any]]) -> str:
    if not source_inputs:
        return "이 단계에 사용할 수 있는 명시적 근거가 부족해 기본 브리핑 근거를 사용했습니다."
    phrases = []
    for item in source_inputs[:3]:
        company = _first_text(item.get("company"), "출처")
        evidence = _brief_sentence(item.get("evidence"), max_chars=80)
        if company and evidence:
            phrases.append(f"{company}: {evidence}")
    return _clip_text(" / ".join(phrases), max_chars=260)


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


def _reasoning_stage_source_field(step_seq: int) -> str:
    fields = {
        1: "integrated_issue.main_issue + business_signals + analysis.market_signal",
        2: "analysis.market_signal + analysis.impact_reason + skax_implication",
        3: "peer_implication + analysis_summary + market_signal",
        4: "skax_implication + recommended_actions",
    }
    return fields.get(step_seq, "analysis_package")


def _reasoning_decision_rule(step_seq: int) -> str:
    rules = {
        1: "여러 근거에서 반복되거나 중요도가 높은 사실 신호를 관찰 결과로 압축합니다.",
        2: "관찰된 신호가 고객 평가 기준, 도입 조건, 리스크 기준을 바꾸는지 확인합니다.",
        3: "경쟁사 움직임이 개별 사건인지, 경쟁 방식 변화로 묶이는지 확인합니다.",
        4: "SK AX가 고객군, 오퍼링, 책임 조직, 자원 배분에서 바꿔야 할 결정을 도출합니다.",
    }
    return rules.get(step_seq, "입력 근거와 화면 결과 사이의 연결고리를 요약합니다.")


def _reasoning_output_sentence(items: list[dict[str, Any]]) -> str:
    titles = [
        _brief_sentence(item.get("title"), max_chars=90)
        for item in items
        if str(item.get("title") or "").strip()
    ]
    if not titles:
        return "화면에 표시할 해석 항목이 생성되지 않았습니다."
    return " / ".join(titles[:3])


def _reasoning_stage_name(step_seq: int) -> str:
    names = {
        1: "observation",
        2: "evaluation_shift",
        3: "competition_impact",
        4: "strategy_implication",
    }
    return names.get(step_seq, "interpretation")


def _reasoning_question(step_seq: int) -> str:
    questions = {
        1: "입력 근거에서 실제로 무엇이 확인됐는가?",
        2: "그 변화가 고객/시장 평가 기준을 어떻게 바꾸는가?",
        3: "경쟁사 메시지와 경쟁 방식에는 어떤 영향이 있는가?",
        4: "SK AX는 어떤 의사결정과 실행 기준을 바꿔야 하는가?",
    }
    return questions.get(step_seq, "이 단계의 화면 결론은 어떤 근거에서 나왔는가?")


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


def _front_step_item(
    *,
    title: str,
    description: str,
    evidence_card_ids: list[Any],
) -> dict[str, Any]:
    return {
        "title": _brief_sentence(title, max_chars=100),
        "description": _brief_sentences(description, max_sentences=3, max_chars=240),
        "evidence_card_ids": [str(item) for item in evidence_card_ids if str(item).strip()],
    }


def _front_limited_step_items(
    candidates: list[dict[str, Any]],
    *,
    fallback: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_titles: list[str] = []
    seen_descriptions: list[str] = []
    for candidate in candidates:
        title = str(candidate.get("title") or "").strip()
        description = str(candidate.get("description") or "").strip()
        if not title or not description:
            continue
        if _is_too_similar(title, seen_titles, threshold=0.82):
            continue
        if _is_too_similar(description, seen_descriptions, threshold=0.78):
            continue
        item = copy.deepcopy(candidate)
        item["seq"] = len(items) + 1
        items.append(item)
        seen_titles.append(title)
        seen_descriptions.append(description)
        if len(items) >= 3:
            break
    if not items:
        fallback_item = copy.deepcopy(fallback)
        fallback_item["seq"] = 1
        items.append(fallback_item)
    return items


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


_ROBOT_OPS_TOKENS = ("로봇", "스마트팩토리", "피지컬")
_PRIVATE_AI_TOKENS = ("프라이빗", "데이터 통제", "거버넌스", "자체 AI")


def _grounded_front_briefing_lead(result: dict[str, Any]) -> str:
    entries = _front_evidence_entries(result)
    if not entries:
        return ""
    clauses = _front_join_company_signal_clauses(entries)
    has_robot_signal = _has_token_entry(entries, _ROBOT_OPS_TOKENS)
    has_private_ai_signal = _has_token_entry(entries, _PRIVATE_AI_TOKENS)
    if not (has_robot_signal or has_private_ai_signal):
        return (
            f"오늘 수집된 경쟁사 신호에서는 {_front_primary_signal_summary(entries)} {clauses}"
        ).strip()
    return (
        "오늘 수집된 경쟁사 신호는 AX 평가 기준이 기술 도입 자체보다 "
        "운영 기반과 데이터 통제 쪽으로 이동하고 있음을 보여줍니다. "
        f"{clauses}"
    )


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


def _front_business_signals(integrated: dict[str, Any]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for item in _json_list(integrated.get("business_signals")):
        if not isinstance(item, dict):
            continue
        signal = _brief_noun_phrase(item.get("signal"), max_chars=56)
        description = _brief_sentence(item.get("description"), max_chars=100)
        if signal or description:
            signals.append({"signal": signal, "description": description})
    return signals


def _front_key_numbers(integrated: dict[str, Any]) -> list[dict[str, str]]:
    numbers: list[dict[str, str]] = []
    for item in _json_list(integrated.get("key_numbers")):
        if not isinstance(item, dict):
            continue
        value = _first_text(item.get("value"))
        context = _first_text(item.get("context"))
        if value and _looks_like_key_number(value):
            numbers.append({"value": value, "context": context})
    return numbers


def _front_evidence_card_ids(entries: list[dict[str, Any]]) -> list[str]:
    return _dedupe_keep_order(
        [str(entry.get("card_id")) for entry in entries if entry.get("card_id")]
    )


def _entry_evidence_card_ids(entry: dict[str, Any]) -> list[str]:
    card_id = str(entry.get("card_id") or "").strip()
    return [card_id] if card_id else []


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


def _entry_has_tokens(entry: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    text = " ".join(
        [
            str(entry.get("main_issue") or ""),
            str(entry.get("integrated_text") or ""),
            str(entry.get("analysis_summary") or ""),
            str(entry.get("market_signal") or ""),
            str(entry.get("impact_reason") or ""),
            str(entry.get("peer_meaning") or ""),
            str(entry.get("capability_change") or ""),
            " ".join(
                f"{signal.get('signal', '')} {signal.get('description', '')}"
                for signal in _json_list(entry.get("business_signals"))
                if isinstance(signal, dict)
            ),
        ]
    )
    return any(token in text for token in tokens)


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
    return _brief_noun_phrase(
        _first_text(
            entry.get("market_signal"),
            entry.get("analysis_summary"),
            entry.get("main_issue"),
        ),
        max_chars=48,
    )


def _front_signal_clause(entry: dict[str, Any], *, tokens: tuple[str, ...] = ()) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=tokens)
    phrase = _front_company_signal_phrase(company, signal)
    if phrase:
        return phrase
    return signal


def _front_company_signal_phrase(company: str, signal: str) -> str:
    if not company or not signal:
        return ""
    if any(token in signal for token in ("수요", "전망", "예상")):
        return f"{company} 관련 {signal}"
    return f"{company}의 {signal}"


def _front_signal_description(
    entry: dict[str, Any],
    *,
    tokens: tuple[str, ...] = (),
) -> str:
    signals = [item for item in _json_list(entry.get("business_signals")) if isinstance(item, dict)]
    for item in signals:
        text = f"{item.get('signal', '')} {item.get('description', '')}"
        if not tokens or any(token in text for token in tokens):
            return _brief_sentence(item.get("description"), max_chars=110)
    return _brief_sentence(entry.get("analysis_summary"), max_chars=110)


def _front_market_change_title(entries: list[dict[str, Any]]) -> str:
    robot_entry = _front_entry_by_tokens(entries, _ROBOT_OPS_TOKENS)
    private_entry = _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS)
    if robot_entry and private_entry and robot_entry is not private_entry:
        return "AX 경쟁의 초점이 운영 인프라와 데이터 통제 중심으로 이동하고 있습니다."
    market_signal = _first_text(*[entry.get("market_signal") for entry in entries])
    return _brief_sentence(
        market_signal or "기간 내 수요 변화가 운영 기반 중심으로 구체화되고 있습니다.",
        max_chars=100,
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
    signals = _front_primary_signal_names(entries)
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
    return _brief_sentence(_block_text(result.get("common_pattern"), "rationale"), max_chars=180)


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
        max_chars=90,
    )


def _front_generic_market_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry)
    impact = _brief_sentence(
        _first_text(entry.get("impact_reason"), entry.get("analysis_summary")),
        max_chars=120,
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
    if len(entries) >= 2:
        return "경쟁사들은 기술 신호를 운영 패키지와 성장 논리로 묶고 있습니다."
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
            f"{joined} 이 움직임은 경쟁사들이 단일 기능보다 운영 역량, 구축 방식, "
            "성과 전망을 함께 묶어 시장 메시지를 만들고 있음을 보여줍니다."
        )
    return "선택된 카드에서 경쟁사의 사업·기술 메시지 변화가 확인됩니다."


def _front_peer_move_sentence(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    peer_meaning = _brief_sentence(
        _first_text(entry.get("peer_meaning"), entry.get("analysis_summary")),
        max_chars=110,
    )
    if company and peer_meaning.startswith(company):
        peer_meaning = peer_meaning.removeprefix(company).lstrip("은 는 이 가 의")
    peer_meaning = _front_polite_sentence(peer_meaning)
    if company and peer_meaning:
        return f"{company}는 {peer_meaning}"
    return peer_meaning


def _front_polite_sentence(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(("습니다.", "합니다.", "입니다.")):
        return text
    text = _strip_terminal_punctuation(text)
    if text.endswith("하고 있다"):
        return f"{text.removesuffix('하고 있다')}하고 있습니다."
    if text.endswith("되고 있다"):
        return f"{text.removesuffix('되고 있다')}되고 있습니다."
    if text.endswith("커지고 있다"):
        return f"{text.removesuffix('커지고 있다')}커지고 있습니다."
    if text.endswith("있다"):
        return f"{text.removesuffix('있다')}있습니다."
    if text.endswith("된다"):
        return f"{text.removesuffix('된다')}됩니다."
    return f"{text}."


def _front_competitor_move_importance(entries: list[dict[str, Any]]) -> str:
    return (
        "따라서 SK AX는 경쟁사 메시지가 실제 수주, 고객 사례, 운영 성과 지표로 "
        "이어지는지 확인하면서 고객군, 오퍼링, 자원 배분 우선순위를 조정해야 합니다."
    )


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


def _is_program_artifact_action(value: object) -> bool:
    text_value = str(value or "")
    return any(
        token in text_value
        for token in (
            "제안서",
            "제안 첫 장",
            "PoC",
            "후속 모니터링",
            "모니터링 항목",
            "대시보드",
            "화면",
        )
    )


def _front_action_title(action: object) -> str:
    title = _brief_sentence(action, max_chars=90)
    if not title:
        return ""
    if title.endswith(("합니다.", "해야 합니다.", "필요가 있습니다.")):
        return title
    return _front_polite_sentence(title)


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


def _front_evaluation_shift_title(entries: list[dict[str, Any]]) -> str:
    if _front_entry_by_tokens(entries, _PRIVATE_AI_TOKENS):
        return "고객 평가는 기능 보유보다 운영 가능성과 데이터 통제로 이동하고 있습니다."
    return "고객 평가는 기술 보유보다 운영 성과 검증으로 이동하고 있습니다."


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


def _front_competition_impact_title(entries: list[dict[str, Any]]) -> str:
    return "경쟁 메시지는 단일 기능보다 운영 패키지 중심으로 재구성되고 있습니다."


def _front_competition_impact_description(entries: list[dict[str, Any]]) -> str:
    clauses = _front_join_company_signal_clauses(entries)
    return (
        f"{clauses} 이 조합은 경쟁사가 개별 기능을 따로 설명하는 단계에서 "
        "벗어나, 운영 인프라·데이터 통제·성장 전망을 하나의 패키지로 묶어 "
        "경쟁력을 설명하는 방향으로 가고 있음을 보여줍니다."
    )


def _front_strategy_implication_title(entries: list[dict[str, Any]]) -> str:
    return "SK AX는 운영 책임과 성과 검증 기준을 의사결정 게이트로 둬야 합니다."


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


def _front_private_ai_market_title(entry: dict[str, Any]) -> str:
    return "프라이빗 AI 수요는 데이터 통제 기준을 끌어올리고 있습니다."


def _front_private_ai_market_description(entry: dict[str, Any]) -> str:
    company = _first_text(entry.get("company"))
    signal = _front_signal_name(entry, tokens=_PRIVATE_AI_TOKENS)
    phrase = _front_company_signal_phrase(company, signal)
    return (
        f"{phrase}{_subject_particle(phrase)} 확인됩니다. "
        "따라서 고객은 AI 기능 자체보다 데이터 위치, 접근 권한, 운영 책임, "
        "거버넌스 체계를 함께 평가하게 됩니다."
    ).strip()


def _front_robot_market_title(entry: dict[str, Any]) -> str:
    return "로봇 운영 데이터는 제조 SW 인프라의 핵심 판단 근거가 됩니다."


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


def _front_reference_skax_description(entries: list[dict[str, Any]]) -> str:
    clauses = _front_join_company_signal_clauses(entries)
    return (
        f"{clauses} 이 근거들을 레퍼런스로 보여줄 때는 무엇을 구축했는지보다 "
        "어떤 운영 문제가 줄었고, 어떤 지표로 안정화됐으며, 어디까지 확산됐는지를 "
        "먼저 정리해야 신뢰 가능한 운영 실적으로 읽힙니다."
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


def _front_key_number_for_tokens(
    entry: dict[str, Any],
    tokens: tuple[str, ...],
) -> str:
    for item in _json_list(entry.get("key_numbers")):
        if not isinstance(item, dict):
            continue
        context = str(item.get("context") or "")
        value = str(item.get("value") or "")
        if value and any(token in context for token in tokens):
            return f"{context} {value}"
    return ""


def _key_change_cards_from_result(result: dict[str, Any]) -> list[Any]:
    items = _json_list(result.get("key_change_cards"))
    if items:
        return items
    return _json_list(_nested_get(result, "core_change", "items"))


def _public_key_change_cards_from_result(
    result: dict[str, Any],
    *,
    repair: bool = True,
) -> list[dict[str, Any]]:
    visible_items: list[dict[str, Any]] = []
    for item in _key_change_cards_from_result(result):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if not normalized.get("description") and normalized.get("summary"):
            normalized["description"] = normalized.get("summary")
        visible_items.append(
            _compact_visible_item(
                normalized,
                ("seq", "display_label", "insight_type", "title", "description", "why_important"),
            )
        )
    if not repair:
        return visible_items
    return _repair_public_display_items(visible_items, result)


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


def _compact_flow_step(step: dict[str, Any]) -> dict[str, Any]:
    visible = _compact_visible_item(step, ("seq", "label"))
    items = [
        _compact_visible_item(item, ("seq", "title", "description"))
        for item in _json_list(step.get("items"))
        if isinstance(item, dict)
    ]
    if items:
        visible["items"] = items[:3]
    reasoning_trace = _compact_reasoning_trace(step.get("reasoning_trace"))
    if reasoning_trace:
        visible["reasoning_trace"] = reasoning_trace
    return visible


def _compact_reasoning_summary(value: object) -> dict[str, Any]:
    summary = _json_dict(value)
    if not summary:
        return {}
    return {
        key: summary.get(key)
        for key in (
            "display_title",
            "disclosure_level",
            "note",
            "source_card_ids",
            "source_integrated_issue_ids",
            "source_mode",
            "basis",
        )
        if summary.get(key) not in (None, "", [], {})
    }


def _compact_reasoning_trace(value: object) -> dict[str, Any]:
    trace = _json_dict(value)
    if not trace:
        return {}
    compact = {
        key: trace.get(key)
        for key in ("stage", "label", "question")
        if trace.get(key) not in (None, "", [], {})
    }
    source_inputs = [
        _compact_visible_item(
            item,
            ("card_id", "integrated_issue_id", "company", "evidence", "source_names"),
        )
        for item in _json_list(trace.get("source_inputs"))
        if isinstance(item, dict)
    ]
    if source_inputs:
        compact["source_inputs"] = source_inputs[:4]
    artifacts = [
        _compact_visible_item(item, ("name", "output", "source_field"))
        for item in _json_list(trace.get("intermediate_artifacts"))
        if isinstance(item, dict)
    ]
    if artifacts:
        compact["intermediate_artifacts"] = artifacts[:4]
    output_items = [
        _compact_visible_item(item, ("seq", "title", "description", "evidence_card_ids"))
        for item in _json_list(trace.get("output_items"))
        if isinstance(item, dict)
    ]
    if output_items:
        compact["output_items"] = output_items[:3]
    return compact


def _public_market_reading_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        _compact_visible_item(item, ("seq", "title", "description"))
        for item in _json_list(result.get("market_reading"))
        if isinstance(item, dict)
    ]
    return _repair_market_reading_items(items, result)


def _public_sk_ax_view_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _compact_visible_item(item, ("seq", "title", "description"))
        for item in _repair_sk_ax_view_descriptions(
            [item for item in _json_list(result.get("sk_ax_view")) if isinstance(item, dict)],
            result,
        )
        if isinstance(item, dict)
    ]


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


def _market_description_is_card_listing(value: str) -> bool:
    text = str(value or "")
    company_names = [
        name for name in ("현대오토에버", "LG CNS", "삼성SDS", "포스코DX") if name in text
    ]
    listing_markers = ("각각", "통해 시장", "사례는", "주도하고 있습니다")
    return len(company_names) >= 2 and any(marker in text for marker in listing_markers)


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


def _join_display_sentences(*sentences: str) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())


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


def _display_date_ko(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return value.replace(".", "")
    return f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일"


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
    signals = _basis_signal_phrases_from_result(result)
    signal_text = _join_korean(signals[:2]) if signals else ""
    company_basis = _company_issue_sentence_from_result(result)
    if insight_type == "market_signal":
        if signal_text:
            return (
                f"{scope}에서 {signal_text}{_subject_particle(signal_text)} 함께 확인됩니다. "
                "이 조합은 개별 기업 이벤트보다 시장의 수요와 평가 기준이 "
                "운영 기반 중심으로 움직이고 있음을 보여줍니다."
            )
        return company_basis
    if insight_type == "competitor_move":
        if company_basis:
            return (
                f"{company_basis} 이 움직임은 경쟁사들이 단일 기술 설명보다 "
                "운영 역량, 구축 방식, 고객 설득 근거를 사업 메시지로 묶고 있음을 보여줍니다."
            )
        if signal_text:
            return (
                f"{scope}에서 {signal_text}{_subject_particle(signal_text)} 확인됩니다. "
                "이는 경쟁사들이 감지된 수요 변화에 맞춰 사업 방향을 구체화하고 있음을 뜻합니다."
            )
    return _brief_sentence(title, max_chars=180)


def _grounded_why_important(value: str, result: dict[str, Any]) -> str:
    basis = _detailed_basis_sentence_from_result(result) or _basis_signal_sentence_from_result(
        result
    )
    if not basis:
        return value
    return f"{basis} 따라서 {str(value).removeprefix('따라서 ').strip()}"


def _description_needs_detail(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return len(text) < 95 or len(re.split(r"(?<=[.!?。！？])\s+", text)) <= 1


def _compact_visible_item(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: item[key] for key in keys if item.get(key) not in (None, "", [])}


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
