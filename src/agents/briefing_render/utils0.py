"""briefing_render utils0 — extracted from facade (move-only)."""

from __future__ import annotations

import copy
import re
from datetime import UTC, date, datetime
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


def _strip_visual_ellipsis_from_payload(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s*(?:\.{3,}|…|⋯)\s*", " ", value).strip()
    if isinstance(value, list):
        return [_strip_visual_ellipsis_from_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_visual_ellipsis_from_payload(item) for key, item in value.items()}
    return value


def _sanitize_internal_display_terms(value: str) -> str:
    text = str(value or "").strip()
    replacements = (
        ("선택된 카드들의", "이번 기간 근거의"),
        ("선택된 카드들은", "이번 기간 근거는"),
        ("선택된 카드에서", "이번 기간 근거에서"),
        ("선택된 카드의", "이번 기간 근거의"),
        ("선택된 카드", "이번 기간 근거"),
        ("피어 프로필의 기존 역량", "경쟁사의 기존 사업 역량"),
        ("피어 프로필 역량", "경쟁사의 기존 사업 역량"),
        ("피어 프로필", "경쟁사 기존 사업 정보"),
        ("프로필의 기존 역량", "기존 사업 역량"),
        ("프로필 역량", "기존 사업 역량"),
        ("프로필의", "기존 사업 정보의"),
        ("프로필", "기존 사업 정보"),
        ("industry_trend", "산업 동향"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _strip_period_scope_prefix(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"^\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*일간에\s*수집된\s*근거에서\s*",
        "",
        text,
    )
    text = re.sub(
        r"^\d{4}년\s*\d{1,2}월\s*\d{1,2}일부터\s*"
        r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일까지\s*(?:주간|월간|기간)에\s*수집된\s*근거에서\s*",
        "",
        text,
    )
    return text.strip()


def _normalize_duplicate_company_prefix(value: str) -> str:
    text = str(value or "").strip()
    company_names = ("LG CNS", "삼성SDS", "포스코DX", "현대오토에버", "SK AX")
    for company in company_names:
        pattern = rf"^{re.escape(company)}\s*:\s*{re.escape(company)}(?=(?:은|는|이|가|의|,|\s))"
        text = re.sub(pattern, company, text)
    return text


def _normalize_korean_plain_ending(value: str) -> str:
    text = str(value or "").strip()
    replacements = (
        (" 낮췄다.", " 낮췄습니다."),
        (" 높였다.", " 높였습니다."),
        (" 기록했다.", " 기록했습니다."),
        (" 출시했다.", " 출시했습니다."),
        (" 체결했다.", " 체결했습니다."),
        (" 선정됐다.", " 선정됐습니다."),
        (" 내정됐다.", " 내정됐습니다."),
        (" 밝혔다.", " 밝혔습니다."),
        (" 확대했다.", " 확대했습니다."),
        (" 강화했다.", " 강화했습니다."),
    )
    for old, new in replacements:
        if text.endswith(old):
            return f"{text.removesuffix(old)}{new}"
    return text


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
            issue = _strip_leading_company_prefix(issue, company)
        if company and issue:
            phrases.append(f"{company}의 {issue}")
    return _dedupe_keep_order(phrases)


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
    return _brief_sentence(headline, max_chars=_DISPLAY_TITLE_MAX)


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
    return _brief_sentences(reason or fallback, max_sentences=2, max_chars=_DISPLAY_REASON_MAX)


def _briefing_trends_summary(trends: list[dict[str, Any]]) -> str:
    reasons: list[object] = [
        str(item.get("reason") or "").strip() for item in trends if item.get("reason")
    ]
    return _combine_blocks(
        reasons,
        "통합 이슈 근거 기준으로 우선순위를 나눠 정리했습니다.",
        max_items=2,
        max_chars=_DISPLAY_BODY_MAX,
    )


def _frontend_daily_title(period: dict[str, Any]) -> str:
    target = period["date_from"]
    return f"{target.year}년 {target.month}월 {target.day}일 Peer Intelligence 일간 브리핑"


def _frontend_weekly_title(period: dict[str, Any]) -> str:
    start = period["date_from"]
    week_no = ((start.day - 1) // 7) + 1
    return f"{start.year}년 {start.month}월 {week_no}주차 Peer Intelligence 주간 브리핑"


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


def _front_flow_detail(item: dict[str, Any]) -> str:
    title = _first_text(item.get("title"))
    description = _first_text(item.get("description"))
    if title and description:
        return f"{title} {description}"
    return title or description


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
            "title": _brief_sentence(title, max_chars=_DISPLAY_TITLE_MAX),
            "reason": _brief_sentences(reason, max_sentences=2, max_chars=_DISPLAY_REASON_MAX),
        }
        for title, reason in candidates
        if _first_text(title)
    ][:4]


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


def _display_evidence_ids(
    selected_cards: list[dict[str, Any]],
    default_evidence: list[Any],
) -> list[str]:
    evidence_ids = [str(item) for item in default_evidence if str(item).strip()]
    if evidence_ids:
        return evidence_ids
    return [str(card["id"]) for card in selected_cards if card.get("id")]


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
    return "기간 내 시장 변화 신호가 확인되었습니다."


def _competitor_move_group_title(entries: list[dict[str, Any]]) -> str:
    companies = _entry_company_labels(entries)
    if len(entries) >= 2:
        if companies:
            joined = _join_korean(companies[:3])
            return f"{joined}의 사업 메시지가 서로 다른 실행 축으로 나뉘고 있습니다."
        return "경쟁사별 사업 메시지가 고객 적용 범위와 구축 방식으로 나뉘고 있습니다."
    return "경쟁사들은 기간 내 감지된 변화에 맞춰 사업과 기술 메시지를 조정하고 있습니다."


def _market_signal_reason_summary(entries: list[dict[str, Any]]) -> str:
    reasons = []
    for entry in entries[:3]:
        reason = _first_text(
            entry.get("impact_reason"),
            _first_from_list(entry.get("strategic_meaning")),
            entry.get("analysis_summary"),
        )
        reason = _strip_terminal_punctuation(_brief_sentence(reason, max_chars=220))
        if reason:
            reasons.append(reason)
    reasons = _dedupe_keep_order(reasons)
    if len(reasons) >= 2:
        return " ".join(f"{reason}." for reason in reasons[:2])
    if reasons:
        return f"이 변화는 단발 이슈보다 {reasons[0]} 흐름과 연결됩니다."
    if len(entries) >= 2:
        return "수요 변화와 평가 기준 변화가 같은 방향으로 확인되고 있습니다."
    return "이 신호는 이후 수요 변화와 평가 기준을 확인할 시장 변화로 해석됩니다."


def _entry_company_labels(entries: list[dict[str, Any]]) -> list[str]:
    return _dedupe_keep_order(
        [
            str(entry.get("company_label") or "").strip()
            for entry in entries
            if str(entry.get("company_label") or "").strip()
        ]
    )


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


def _combine_company_signals(entries: list[dict[str, Any]], key: str) -> str:
    phrases = []
    for entry in entries[:3]:
        company = str(entry.get("company_label") or "").strip()
        signal = _brief_sentence(entry.get(key), max_chars=_DISPLAY_BODY_MAX)
        if company and signal:
            phrases.append(f"{company}: {signal}")
    return " / ".join(phrases)


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
        signal = _brief_sentence(_card_summary(card), max_chars=_DISPLAY_BODY_MAX)
        if signal:
            phrases.append(signal)
    if phrases:
        return " / ".join(phrases)
    return ""


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
    return _brief_sentence(key_summary, max_chars=_DISPLAY_BODY_MAX)


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


def _evidence_texts(package: dict[str, Any]) -> list[str]:
    integrated = _json_dict(package.get("integrated_issue"))
    fact_basis = _json_list(integrated.get("fact_basis"))
    ledger = _json_list(integrated.get("evidence_ledger"))
    texts = []
    for item in [*fact_basis, *ledger]:
        if isinstance(item, dict) and str(item.get("evidence_text") or "").strip():
            texts.append(str(item["evidence_text"]).strip())
    return texts[:5]


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


def _front_step_item(
    *,
    title: str,
    description: str,
    evidence_card_ids: list[Any],
) -> dict[str, Any]:
    return {
        "title": _brief_sentence(title, max_chars=_DISPLAY_TITLE_MAX),
        "description": _brief_sentences(description, max_sentences=3, max_chars=_DISPLAY_FLOW_MAX),
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


def _front_briefing_lead_prefix(result: dict[str, Any]) -> str:
    briefing_type = str(result.get("briefing_type") or "").strip()
    if briefing_type == "weekly":
        return "이번 주 경쟁사와 산업 신호에서는"
    if briefing_type == "monthly":
        return "이번 달 경쟁사와 산업 신호에서는"
    return "오늘 수집된 경쟁사 신호에서는"


def _front_business_signals(integrated: dict[str, Any]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for item in _json_list(integrated.get("business_signals")):
        if not isinstance(item, dict):
            continue
        signal = _brief_noun_phrase(item.get("signal"), max_chars=56)
        description = _brief_sentence(item.get("description"), max_chars=_DISPLAY_BODY_MAX)
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


def _entry_evidence_card_ids(entry: dict[str, Any]) -> list[str]:
    card_id = str(entry.get("card_id") or "").strip()
    return [card_id] if card_id else []


def _entry_has_tokens(entry: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    text = " ".join(
        [
            str(entry.get("main_issue") or ""),
            str(entry.get("integrated_text") or ""),
            " ".join(
                f"{signal.get('signal', '')} {signal.get('description', '')}"
                for signal in _json_list(entry.get("business_signals"))
                if isinstance(signal, dict)
            ),
        ]
    )
    return any(token in text for token in tokens)


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
            return _brief_sentence(item.get("description"), max_chars=_DISPLAY_BODY_MAX)
    return _brief_sentence(entry.get("analysis_summary"), max_chars=_DISPLAY_BODY_MAX)


def _is_profile_linkage_display_text(value: str) -> bool:
    text = str(value or "")
    markers = (
        "경쟁사 기존 사업 정보",
        "기존 사업 정보",
        "에이엑스씽크",
        "자체 로봇 학습 플랫폼",
        "연결되는 배경",
        "연결되는 신호",
    )
    return any(marker in text for marker in markers)


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


def _strip_leading_company_prefix(value: str, company: str) -> str:
    text = str(value or "").strip()
    company_text = str(company or "").strip()
    if company_text and text.startswith(company_text):
        text = text[len(company_text) :]
    text = re.sub(r"^[\s,，·ㆍ:;/'\"-]+", "", text)
    text = re.sub(r"^(은|는|이|가|의|와|과)\s*", "", text)
    text = re.sub(r"^[\s,，·ㆍ:;/'\"-]+", "", text)
    return text.strip()


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
    title = _brief_sentence(action, max_chars=_DISPLAY_TITLE_MAX)
    if not title:
        return ""
    if title.endswith(("합니다.", "해야 합니다.", "필요가 있습니다.")):
        return title
    return _front_polite_sentence(title)


def _front_competition_impact_title(entries: list[dict[str, Any]]) -> str:
    return "경쟁 메시지는 단일 기능보다 운영 패키지 중심으로 재구성되고 있습니다."


def _front_strategy_implication_title(entries: list[dict[str, Any]]) -> str:
    return "SK AX는 운영 책임과 성과 검증 기준을 의사결정 게이트로 둬야 합니다."


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


def _public_sk_ax_view_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _compact_visible_item(item, ("seq", "title", "description"))
        for item in _repair_sk_ax_view_descriptions(
            [item for item in _json_list(result.get("sk_ax_view")) if isinstance(item, dict)],
            result,
        )
        if isinstance(item, dict)
    ]


def _market_description_is_card_listing(value: str) -> bool:
    text = str(value or "")
    company_names = [
        name for name in ("현대오토에버", "LG CNS", "삼성SDS", "포스코DX") if name in text
    ]
    listing_markers = ("각각", "통해 시장", "사례는", "주도하고 있습니다")
    return len(company_names) >= 2 and any(marker in text for marker in listing_markers)


def _join_display_sentences(*sentences: str) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())


def _period_perspective_tail(result: dict[str, Any], insight_type: str) -> str:
    briefing_type = str(result.get("briefing_type") or "").strip()
    if briefing_type == "weekly":
        if insight_type == "competitor_move":
            return (
                "주간 관점에서는 같은 방향의 메시지가 며칠 간격으로 반복되는지와 "
                "후속 수주·고객 사례로 이어지는지를 함께 봐야 합니다."
            )
        return (
            "주간 관점에서는 이번 주 안에서 반복된 신호와 새로 강해진 평가 기준을 "
            "분리해 보는 것이 중요합니다."
        )
    if briefing_type == "monthly":
        if insight_type == "competitor_move":
            return (
                "월간 관점에서는 개별 발표보다 한 달 동안 누적된 경쟁사별 실행 축과 "
                "고객 설득 방식의 차이를 비교해야 합니다."
            )
        return (
            "월간 관점에서는 단발 이벤트보다 한 달 동안 누적된 수요 변화와 "
            "고객 평가 기준의 이동을 봐야 합니다."
        )
    return ""


def _entry_basis_sort_key(entry: dict[str, Any]) -> str:
    return str(entry.get("basis_at") or entry.get("published_at") or "")


def _looks_like_mostly_english(value: str) -> bool:
    text = str(value or "")
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return False
    english = re.findall(r"[A-Za-z]", text)
    korean = re.findall(r"[가-힣]", text)
    return len(english) > len(korean)


def _display_date_ko(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return value.replace(".", "")
    return f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일"


def _grounded_why_important(value: str, result: dict[str, Any]) -> str:
    basis = _detailed_basis_sentence_from_result(result) or _basis_signal_sentence_from_result(
        result
    )
    if not basis:
        return value
    return f"{basis} 따라서 {str(value).removeprefix('따라서 ').strip()}"
