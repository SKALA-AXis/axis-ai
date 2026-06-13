"""basis_builder — briefing_generation_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

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
from src.services.analysis_units import (  # noqa: E402
    analysis_units_from_cards,
    confidence_penalty_for_flags,
    quality_flags_for_units,
    source_integrated_issue_ids,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

_PROMPT_VERSION = "briefing-generation-v0.3-period-briefing"


_BRIEFING_SYNTHESIS_PROMPT_VERSION = "briefing-synthesis-v0.4-integrated-issue-frontend-contract"


_LLM_MODEL = os.getenv("BRIEFING_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o"


_llm: ChatOpenAI | None = None


def _llm_max_completion_tokens() -> int:
    # gpt-5 계열은 reasoning 토큰이 max_completion_tokens 안에서 소비되므로 캡 상향.
    # mixer(9000)/today_insight(12000) 패턴과 동일. 브리핑 본문 출력은 ~2.4K 라
    # 8000 이면 reasoning headroom + 출력 모두 충분.
    default = "8000" if str(_LLM_MODEL).startswith("gpt-5") else "2400"
    return int(os.getenv("BRIEFING_MAX_COMPLETION_TOKENS", default))


def _get_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI  # lazy: transformers 체인 회피

    global _llm
    if _llm is None:
        llm_kwargs: dict[str, Any] = {
            "model": _LLM_MODEL,
            "temperature": 0.1,
            "max_completion_tokens": _llm_max_completion_tokens(),
            "model_kwargs": {"response_format": {"type": "json_object"}},
        }
        # gpt-5 계열은 reasoning_effort 지정 (mixer/today_insight 와 동일 패턴).
        if str(_LLM_MODEL).startswith("gpt-5"):
            llm_kwargs["reasoning_effort"] = os.getenv("BRIEFING_REASONING_EFFORT", "low")
        _llm = ChatOpenAI(**llm_kwargs)
    return _llm


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


def _display_sk_ax_title(selected_cards: list[dict[str, Any]]) -> str:
    for action, _why in _recommended_action_pairs(selected_cards):
        title = _brief_sentence(action)
        if title:
            return title
    return ""


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


def _front_evidence_card_ids(entries: list[dict[str, Any]]) -> list[str]:
    return _dedupe_keep_order(
        [str(entry.get("card_id")) for entry in entries if entry.get("card_id")]
    )
