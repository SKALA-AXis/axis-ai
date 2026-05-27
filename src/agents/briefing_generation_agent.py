"""BriefingGenerationAgent.

기간별 card_news만 선택한 뒤, card_news.evidence_payload에 연결된
integrated_issue / analysis / implication / classification / validation 결과로
브리핑 화면 payload를 생성한다.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_openai import ChatOpenAI  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.agents.mixer_analysis_agent import MixerAnalysisAgent  # noqa: E402
from src.config.env_loader import load_profile  # noqa: E402
from src.db.briefing_reports import save_briefing_report  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402

log = logging.getLogger(__name__)

BriefingType = Literal["daily", "weekly", "monthly"]

KST = ZoneInfo("Asia/Seoul")
_PROMPT_VERSION = "briefing-generation-v0.3-period-mixer-briefing"
_DISPLAY_COPY_PROMPT_VERSION = "briefing-display-copy-v0.1"
_LLM_MODEL = os.getenv("BRIEFING_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o"
_DEFAULT_LIMIT = 20
_SECTOR_FILTER_FETCH_MULTIPLIER = 5
_MAX_DISPLAY_CARDS = 3
_MAX_MARKET_ITEMS = 3
_MAX_SKAX_ITEMS = 3
_DEFAULT_MOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "crawler"
    / "crawler_results"
    / "mixer_mock"
    / "mixer_analysis_packages_20260522.json"
)

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.2,
            max_completion_tokens=2400,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


class BriefingGenerationAgent:
    """기간 단위 카드뉴스 브리핑을 생성하는 오케스트레이터.

    책임:
    - daily / weekly / monthly 기간에 해당하는 card_news만 선택한다.
    - 선택된 card_id를 MixerAnalysisAgent에 전달한다.
    - Mixer 결과를 브리핑 화면 구조에 맞는 payload로 변환한다.

    책임 아님:
    - 카드뉴스 자체 생성
    - 원문 뉴스 재요약
    - 이메일 발송

    TODO: DB migration에서 briefing_type CHECK 제약은 daily / weekly / monthly만
    허용하고 custom은 제외해야 한다.
    """

    prompt_version = _PROMPT_VERSION

    def __init__(self, *, llm: Any | None = None, mixer: Any | None = None) -> None:
        self._llm = llm
        self._mixer = mixer

    async def generate(
        self,
        *,
        briefing_type: BriefingType,
        anchor_date: str | date | None = None,
        card_ids: list[str] | None = None,
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
        refine_display_copy: bool = False,
    ) -> dict[str, Any]:
        """기간에 맞는 카드뉴스를 모아 브리핑 payload를 반환한다.

        ``card_ids``가 들어와도 기간 필터는 유지한다. 즉, 일간 브리핑이면 해당 일자
        범위에 속한 카드만 브리핑 근거로 사용된다.
        """

        load_profile()
        requested_card_ids = _clean_ids(card_ids)
        period = _resolve_period(briefing_type, anchor_date)
        source_mode = (
            "mock_fixture" if use_mock or mock_path or mock_items else "card_news_period_filter"
        )
        if source_mode == "mock_fixture":
            mock_source_items = _load_mock_items(mock_path=mock_path, mock_items=mock_items)
            selected_cards = _fetch_mock_period_cards(
                period=period,
                items=mock_source_items,
                card_ids=requested_card_ids,
                peer_ids=peer_ids,
                sectors=sectors,
                limit=limit,
            )
        else:
            mock_source_items = []
            selected_cards = _fetch_period_cards(
                period=period,
                card_ids=requested_card_ids,
                peer_ids=peer_ids,
                sectors=sectors,
                limit=limit,
            )
        selected_card_ids = [card["id"] for card in selected_cards]
        excluded_card_ids = [
            card_id for card_id in requested_card_ids if card_id not in set(selected_card_ids)
        ]
        report_id = _briefing_id(briefing_type, period["date_from"])
        provenance_base = _provenance_base(
            requested_card_ids=requested_card_ids,
            selected_card_ids=selected_card_ids,
            excluded_card_ids=excluded_card_ids,
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

        mixer_context = _mixer_context(
            briefing_type=briefing_type,
            period=period,
            user_context=user_context,
        )
        mixer_result = await self._run_mixer(
            selected_card_ids=selected_card_ids,
            ratios=ratios,
            user_context=mixer_context,
            use_mock=source_mode == "mock_fixture",
            mock_items=mock_source_items,
        )
        briefing_basis = _briefing_basis_from_mixer(
            mixer_result=mixer_result,
            selected_cards=selected_cards,
            period=period,
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
            report = _refine_display_copy_with_llm(
                report=report,
                selected_cards=selected_cards,
                llm=self._llm,
            )
        if save:
            save_briefing_report(report, selected_cards=selected_cards)
        return report

    async def _run_mixer(
        self,
        *,
        selected_card_ids: list[str],
        ratios: dict[str, Any] | None,
        user_context: str | None,
        use_mock: bool,
        mock_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        mixer = self._mixer
        if mixer is None:
            mixer = MixerAnalysisAgent(mock_items=mock_items, prefer_mock=use_mock)
        return await mixer.analyze(
            card_ids=selected_card_ids,
            ratios=ratios,
            user_context=user_context,
        )


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


def _fetch_period_cards(
    *,
    period: dict[str, Any],
    card_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    fetch_limit = max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))
    if sectors:
        fetch_limit *= _SECTOR_FILTER_FETCH_MULTIPLIER
    params: dict[str, Any] = {
        "start_at": period["start_at"],
        "end_at": period["end_exclusive_at"],
        "limit": fetch_limit,
    }
    where = [
        "COALESCE(ra.published_at, cn.created_at) >= :start_at",
        "COALESCE(ra.published_at, cn.created_at) < :end_at",
    ]
    _append_in_filter(where, params, "cn.id", "card_id", card_ids)
    _append_in_filter(
        where,
        params,
        "COALESCE(cn.peer_company_id, cn.company)",
        "peer_id",
        peer_ids,
    )

    sql = f"""
        SELECT
            cn.id,
            cn.title,
            cn.summary_lines,
            cn.event_type,
            cn.importance,
            cn.importance_score,
            cn.company,
            COALESCE(cn.peer_company_id, cn.company) AS peer_id,
            cn.primary_keyword_category,
            cn.implication,
            cn.sources,
            cn.source_articles,
            cn.source_raw_article_ids,
            cn.primary_raw_article_id,
            cn.evidence_payload,
            cn.validation_pass,
            cn.validation_sc_score,
            cn.created_at,
            COALESCE(ra.published_at, cn.created_at) AS basis_at
        FROM card_news cn
        LEFT JOIN raw_articles ra ON ra.id = cn.primary_raw_article_id
        WHERE {" AND ".join(where)}
        ORDER BY
            cn.importance_score DESC NULLS LAST,
            COALESCE(ra.published_at, cn.created_at) DESC,
            cn.created_at DESC
        LIMIT :limit
    """

    try:
        with SessionLocal() as db:
            rows = db.execute(text(sql), params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.exception("BriefingGenerationAgent card_news 기간 조회 실패 | error=%s", exc)
        return []

    cards = [_normalize_card_row(dict(row)) for row in rows]
    final_limit = max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))
    return _filter_by_sectors(cards, sectors)[:final_limit]


def _load_mock_items(
    *,
    mock_path: str | Path | None,
    mock_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if mock_items is not None:
        return mock_items
    if mock_path is None:
        mock_path = _DEFAULT_MOCK_PATH
    path = Path(mock_path)
    if not path.exists():
        log.warning("BriefingGenerationAgent mock path 없음 | path=%s", path)
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("selected_items") or payload.get("mixer_analysis_packages") or []
        return [item for item in items if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _fetch_mock_period_cards(
    *,
    period: dict[str, Any],
    items: list[dict[str, Any]],
    card_ids: list[str] | None,
    peer_ids: list[str] | None,
    sectors: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    requested = set(card_ids or [])
    wanted_peers = {str(peer_id).strip() for peer_id in peer_ids or [] if str(peer_id).strip()}
    cards = [_normalize_mock_item(item) for item in items]
    filtered: list[dict[str, Any]] = []
    for card in cards:
        if requested and card["id"] not in requested:
            continue
        if wanted_peers and card.get("peer_id") not in wanted_peers:
            continue
        basis_at = _parse_datetime(card.get("basis_at") or card.get("created_at"))
        if basis_at is None:
            continue
        basis_utc = basis_at.astimezone(UTC)
        if not (period["start_at"] <= basis_utc < period["end_exclusive_at"]):
            continue
        filtered.append(card)

    filtered = _filter_by_sectors(filtered, sectors)
    filtered.sort(
        key=lambda card: (
            float(card.get("importance_score") or 0.0),
            card.get("basis_at") or "",
        ),
        reverse=True,
    )
    return filtered[: max(1, min(int(limit or _DEFAULT_LIMIT), _DEFAULT_LIMIT))]


def _normalize_mock_item(item: dict[str, Any]) -> dict[str, Any]:
    card_id = str(item.get("card_id") or item.get("id") or "").strip()
    evidence_payload = _json_dict(item.get("evidence_payload"))
    top_package = _json_dict(item.get("analysis_package"))
    payload_package = _json_dict(evidence_payload.get("analysis_package"))
    package_warning = None
    if top_package and payload_package:
        top_bundle = top_package.get("bundle_id")
        payload_bundle = payload_package.get("bundle_id")
        if top_bundle and payload_bundle and top_bundle != payload_bundle:
            package_warning = "analysis_package_bundle_mismatch"
    analysis_package = payload_package or top_package
    if analysis_package:
        evidence_payload["analysis_package"] = analysis_package

    sources = _json_list(item.get("sources"))
    basis_at = _first_source_published_at(sources) or item.get("basis_at") or item.get("created_at")
    implication = _json_dict(item.get("implication"))
    sectors = _json_list(implication.get("sectors")) or _json_list(
        _nested_get(analysis_package, "classification", "sectors")
    )
    sector = _first_text(
        item.get("primary_keyword_category"),
        implication.get("sector"),
        _nested_get(analysis_package, "classification", "sector"),
        item.get("event_type"),
        "other",
    )
    return {
        "id": card_id,
        "card_id": card_id,
        "title": str(item.get("title") or ""),
        "summary_lines": _json_list(item.get("summary_lines")),
        "event_type": str(item.get("event_type") or ""),
        "importance": str(item.get("importance") or "medium"),
        "importance_score": float(item.get("importance_score") or 0.0),
        "company": str(item.get("company") or ""),
        "peer_id": str(item.get("peer_id") or item.get("company") or ""),
        "sector": sector,
        "sectors": [str(value) for value in sectors if str(value).strip()] or [sector],
        "sources": sources,
        "source_raw_article_ids": _int_list(item.get("source_raw_article_ids")),
        "primary_raw_article_id": _first_int(item.get("source_raw_article_ids")),
        "evidence_payload": evidence_payload,
        "analysis_package": analysis_package,
        "has_analysis_package": bool(analysis_package),
        "evidence_card_ids": [card_id] if card_id else [],
        "validation_pass": _nested_get(analysis_package, "validation", "pass"),
        "validation_sc_score": _nested_get(analysis_package, "validation", "sc_score"),
        "created_at": _iso_or_none(item.get("created_at")),
        "basis_at": _iso_or_none(_parse_datetime(basis_at)),
        "mock_warning": package_warning,
    }


def _append_in_filter(
    where: list[str],
    params: dict[str, Any],
    column_sql: str,
    prefix: str,
    values: list[str] | None,
) -> None:
    cleaned = [str(value).strip() for value in values or [] if str(value).strip()]
    if not cleaned:
        return
    placeholders = []
    for index, value in enumerate(cleaned):
        key = f"{prefix}_{index}"
        placeholders.append(f":{key}")
        params[key] = value
    where.append(f"{column_sql} IN ({', '.join(placeholders)})")


def _normalize_card_row(row: dict[str, Any]) -> dict[str, Any]:
    implication = _json_dict(row.get("implication"))
    evidence_payload = _json_dict(row.get("evidence_payload"))
    analysis_package = _analysis_package_from_sources(row, evidence_payload)
    summary_lines = row.get("summary_lines")
    if isinstance(summary_lines, str):
        summary_lines = [summary_lines]
    elif not isinstance(summary_lines, list):
        summary_lines = list(summary_lines or [])
    source_raw_article_ids = _int_list(row.get("source_raw_article_ids"))
    primary_raw_article_id = _optional_int(row.get("primary_raw_article_id"))
    if primary_raw_article_id is not None and primary_raw_article_id not in source_raw_article_ids:
        source_raw_article_ids = [primary_raw_article_id, *source_raw_article_ids]
    sectors = implication.get("sectors")
    if not isinstance(sectors, list):
        sectors = _json_list(_nested_get(analysis_package, "classification", "sectors"))
    sector = (
        row.get("primary_keyword_category")
        or implication.get("sector")
        or _nested_get(analysis_package, "classification", "sector")
        or row.get("event_type")
        or "other"
    )
    card_id = str(row.get("id"))
    return {
        "id": card_id,
        "card_id": card_id,
        "title": str(row.get("title") or ""),
        "summary_lines": [str(item) for item in summary_lines if str(item).strip()],
        "event_type": str(row.get("event_type") or ""),
        "importance": str(row.get("importance") or "medium"),
        "importance_score": float(row.get("importance_score") or 0.0),
        "company": str(row.get("company") or ""),
        "peer_id": str(row.get("peer_id") or row.get("company") or ""),
        "sector": str(sector),
        "sectors": [str(item) for item in sectors if str(item).strip()] or [str(sector)],
        "sources": _json_list(row.get("sources")) or _json_list(row.get("source_articles")),
        "source_raw_article_ids": source_raw_article_ids,
        "primary_raw_article_id": primary_raw_article_id,
        "evidence_payload": evidence_payload,
        "analysis_package": analysis_package,
        "has_analysis_package": bool(analysis_package),
        "evidence_card_ids": [card_id],
        "validation_pass": row.get("validation_pass"),
        "validation_sc_score": row.get("validation_sc_score"),
        "created_at": _iso_or_none(row.get("created_at")),
        "basis_at": _iso_or_none(row.get("basis_at")),
    }


def _filter_by_sectors(
    cards: list[dict[str, Any]],
    sectors: list[str] | None,
) -> list[dict[str, Any]]:
    wanted = {str(item).strip().lower() for item in sectors or [] if str(item).strip()}
    if not wanted:
        return cards
    filtered = []
    for card in cards:
        card_sectors = {str(card.get("sector", "")).lower()}
        card_sectors.update(str(item).lower() for item in card.get("sectors", []))
        if card_sectors & wanted:
            filtered.append(card)
    return filtered


def _provenance_base(
    *,
    requested_card_ids: list[str],
    selected_card_ids: list[str],
    excluded_card_ids: list[str],
    source_mode: str,
) -> dict[str, Any]:
    provenance = {
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "source_card_ids": selected_card_ids,
        "requested_card_ids": requested_card_ids,
        "selected_card_ids": selected_card_ids,
        "excluded_card_ids": excluded_card_ids,
        "source_mode": source_mode,
        "briefing_analysis_basis": (
            "card_id -> card_news.evidence_payload.analysis_package "
            "integrated_issue+analysis+implication+classification+validation"
        ),
    }
    if excluded_card_ids:
        provenance["exclusion_reason"] = "out_of_period"
    return provenance


def _mixer_context(
    *,
    briefing_type: BriefingType,
    period: dict[str, Any],
    user_context: str | None,
) -> str:
    period_role = {
        "daily": "하루 동안 확인된 핵심 변화",
        "weekly": "해당 주간에 누적된 흐름",
        "monthly": "해당 월간의 큰 방향성",
    }[briefing_type]
    lines = [
        f"브리핑 타입: {briefing_type}",
        f"기간: {period['label']} ({period['date_from']}~{period['date_to']})",
        f"해석 초점: {period_role}",
        "출력은 카드뉴스 요약이 아니라 기간 내 핵심 변화, 시장 해석, SK AX 관점이어야 합니다.",
    ]
    if user_context and user_context.strip():
        lines.append(f"사용자 맥락: {user_context.strip()}")
    return "\n".join(lines)


def _briefing_basis_from_mixer(
    *,
    mixer_result: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    period: dict[str, Any],
) -> dict[str, Any]:
    card_ids = [card["id"] for card in selected_cards]
    common = _json_dict(mixer_result.get("common_pattern"))
    comparison = _json_dict(mixer_result.get("comparison_point"))
    hidden = _json_dict(mixer_result.get("hidden_conclusion"))
    actions = [
        item for item in _json_list(mixer_result.get("action_details")) if isinstance(item, dict)
    ]
    mix_insight = _first_text(
        mixer_result.get("mix_insight"),
        common.get("finding"),
        comparison.get("finding"),
        hidden.get("finding"),
        f"{period['label']} 카드뉴스 기반 브리핑입니다.",
    )
    core_finding = _first_text(
        comparison.get("finding"),
        mixer_result.get("mix_insight"),
        common.get("finding"),
    )
    core_summary = _first_text(
        comparison.get("rationale"),
        common.get("rationale"),
        hidden.get("rationale"),
        core_finding,
    )
    strategy_finding = _first_text(
        _aggregate_action_text(actions, "action"),
        _aggregate_action_text(actions, "why"),
        hidden.get("finding"),
    )
    provenance = _json_dict(mixer_result.get("provenance"))
    mixer_prompt_version = provenance.get("prompt_version")
    confidence = _safe_float(mixer_result.get("confidence"), default=0.0)
    return {
        "basis_id": f"BR-BASIS-{period['date_from']:%Y%m%d}",
        "briefing_insight": mix_insight,
        "lead": {
            "finding": mix_insight,
            "evidence_card_ids": _evidence_ids_from_block(mixer_result, card_ids),
        },
        "core_change": {
            "finding": core_finding,
            "rationale": core_summary,
            "evidence": _json_list(comparison.get("evidence")),
            "evidence_card_ids": _evidence_ids_from_block(comparison, card_ids),
        },
        "common_pattern": _basis_block(common, card_ids),
        "comparison_point": _basis_block(comparison, card_ids),
        "hidden_conclusion": _basis_block(hidden, card_ids),
        "strategy_implication": {
            "finding": strategy_finding,
            "rationale": _first_text(_aggregate_action_text(actions, "why"), strategy_finding),
            "evidence": _action_evidence(actions),
            "evidence_card_ids": _evidence_ids_from_actions(actions, card_ids),
        },
        "recommended_action_basis": _json_list(mixer_result.get("recommended_action_basis")),
        "action_details": actions,
        "recommended_actions": _json_list(mixer_result.get("recommended_actions")),
        "confidence": confidence,
        "sources_used": _json_list(mixer_result.get("sources_used")) or card_ids,
        "mixer_result": mixer_result,
        "provenance": {
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": card_ids,
            "analysis_basis": "MixerAnalysisAgent linked results",
            "mixer_prompt_version": mixer_prompt_version,
            "mixer_analysis_basis": provenance.get("analysis_basis"),
        },
    }


def _basis_block(value: dict[str, Any], default_card_ids: list[str]) -> dict[str, Any]:
    return {
        "finding": _first_text(value.get("finding")),
        "rationale": _first_text(value.get("rationale")),
        "evidence": _json_list(value.get("evidence")),
        "evidence_card_ids": _evidence_ids_from_block(value, default_card_ids),
    }


def _evidence_ids_from_block(value: object, default_card_ids: list[str]) -> list[str]:
    if isinstance(value, dict):
        ids = _json_list(value.get("evidence_card_ids") or value.get("sources_used"))
        clean = [str(item) for item in ids if str(item).strip()]
        if clean:
            return _dedupe_keep_order(clean)
    return list(default_card_ids)


def _evidence_ids_from_actions(
    actions: list[dict[str, Any]],
    default_card_ids: list[str],
) -> list[str]:
    ids: list[str] = []
    for action in actions:
        ids.extend(str(item) for item in _json_list(action.get("evidence_card_ids")))
    return _dedupe_keep_order(ids) or list(default_card_ids)


def _aggregate_action_text(actions: list[dict[str, Any]], key: str) -> str:
    return _combine_blocks([action.get(key) for action in actions], "", max_items=2)


def _action_evidence(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for action in actions:
        evidence.extend(
            item for item in _json_list(action.get("evidence")) if isinstance(item, dict)
        )
    return evidence


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _refine_display_copy_with_llm(
    *,
    report: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    llm: Any | None,
) -> dict[str, Any]:
    """analysis_package 기반 payload의 화면 표시문만 LLM으로 정제한다.

    다른 에이전트와 분리하기 위해 이 단계는 card_id별 analysis_package만 입력으로 사용하고,
    evidence/provenance/hidden_details 같은 추적 필드는 코드가 그대로 보존한다.
    """

    if llm is None and not os.getenv("OPENAI_API_KEY"):
        log.info("Briefing display copy refinement skipped: OPENAI_API_KEY is not set")
        return report

    context = _display_copy_context(report, selected_cards)
    messages = [
        (
            "system",
            "\n".join(
                [
                    "당신은 SK AX 전략 브리핑 화면의 문장 편집자입니다.",
                    "입력된 card_news.evidence_payload.analysis_package만 근거로 사용합니다.",
                    "card_news.summary_lines를 판단 근거로 쓰지 않습니다.",
                    "원문 기사 재요약이 아니라 기간 내 변화, 시장 해석, SK AX 관점으로 씁니다.",
                    (
                        "문장은 짧게 쓰고, 화면 기본 노출 문장은 모두 "
                        "'~합니다' 또는 '~있습니다' 문체로 맞춥니다."
                    ),
                    "근거에 없는 회사 주장, 수치, 사건을 추가하지 않습니다.",
                    "evidence_card_ids는 입력에 있는 실제 card_id만 사용합니다.",
                    "JSON 객체만 반환합니다.",
                ]
            ),
        ),
        (
            "human",
            "\n".join(
                [
                    "아래 입력을 브리핑 화면용 표시문으로 정제해주세요.",
                    "",
                    "출력 JSON 스키마:",
                    json.dumps(_display_copy_schema_hint(), ensure_ascii=False, indent=2),
                    "",
                    "작성 규칙:",
                    "- briefing_lead는 최대 2문장입니다.",
                    "- core_change.title과 summary는 각각 1문장입니다.",
                    "- core_change.items는 market_signal, competitor_move 두 슬롯만 유지합니다.",
                    (
                        "- interpretation_flow는 관찰된 변화, 평가축의 이동, "
                        "경쟁 구도 영향, 전략 시사 4단계입니다."
                    ),
                    "- market_reading과 sk_ax_view는 각각 최대 3개입니다.",
                    "- 근거/원문/confidence/debug 정보는 본문에 풀어 쓰지 않습니다.",
                    "",
                    "입력:",
                    json.dumps(context, ensure_ascii=False, indent=2),
                ]
            ),
        ),
    ]
    try:
        response = (llm or _get_llm()).invoke(messages)
    except Exception as exc:  # pragma: no cover - external API safety net
        log.warning("Briefing display copy refinement failed | error=%s", exc)
        return report

    parsed = _parse_json_object(getattr(response, "content", response))
    if not parsed:
        return report
    return _merge_display_copy(report, parsed)


def _display_copy_schema_hint() -> dict[str, Any]:
    return {
        "key_summary": "string",
        "briefing_lead": "string",
        "core_change": {
            "title": "string",
            "summary": "string",
            "items": [
                {
                    "seq": 1,
                    "display_label": "시장 신호",
                    "insight_type": "market_signal",
                    "title": "string",
                    "summary": "string",
                    "so_what": "string",
                    "evidence_card_ids": ["CN-..."],
                },
                {
                    "seq": 2,
                    "display_label": "경쟁사 움직임",
                    "insight_type": "competitor_move",
                    "title": "string",
                    "summary": "string",
                    "so_what": "string",
                    "evidence_card_ids": ["CN-..."],
                },
            ],
        },
        "interpretation_flow": {
            "steps": [
                {
                    "seq": 1,
                    "label": "관찰된 변화",
                    "one_liner": "string",
                    "evidence_card_ids": ["CN-..."],
                }
            ]
        },
        "market_reading": [
            {
                "seq": 1,
                "label": "시장 변화",
                "title": "string",
                "description": "string",
                "evidence_card_ids": ["CN-..."],
            }
        ],
        "sk_ax_view": [
            {
                "seq": 1,
                "use_case": "string",
                "title": "string",
                "description": "string",
                "evidence_card_ids": ["CN-..."],
            }
        ],
    }


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
        "current_display_payload": _frontend_display_payload(report),
        "analysis_packages": [
            {
                "card_id": card.get("id"),
                "company": card.get("company"),
                "peer_id": card.get("peer_id"),
                "company_label": _company_label(card),
                "title": card.get("title"),
                "source_raw_article_ids": card.get("source_raw_article_ids") or [],
                "analysis_package": _compact_analysis_package(_analysis_package(card)),
            }
            for card in selected_cards
        ],
    }


def _compact_analysis_package(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": package.get("bundle_id"),
        "classification": _json_dict(package.get("classification")),
        "integrated_issue": _json_dict(package.get("integrated_issue")),
        "analysis": _json_dict(package.get("analysis")),
        "implication": _json_dict(package.get("implication")),
        "validation": _json_dict(package.get("validation")),
    }


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
) -> dict[str, Any]:
    updated = copy.deepcopy(report)
    _update_text_field(updated, display_copy, "key_summary")
    _update_text_field(updated, display_copy, "briefing_lead")
    if updated.get("key_summary"):
        updated["sk_implication"] = _first_text(
            _nested_get(display_copy, "sk_implication"),
            updated.get("sk_implication"),
        )

    core_copy = _json_dict(display_copy.get("core_change"))
    core = _json_dict(updated.get("core_change"))
    if core and core_copy:
        _update_text_field(core, core_copy, "title")
        _update_text_field(core, core_copy, "summary")
        _merge_core_items(core, core_copy)
        updated["core_change"] = core

    _merge_flow(updated, display_copy)
    _merge_market_reading(updated, display_copy)
    _merge_sk_ax_view(updated, display_copy)

    provenance = _json_dict(updated.get("provenance"))
    provenance["display_copy_prompt_version"] = _DISPLAY_COPY_PROMPT_VERSION
    provenance["display_copy_model"] = _LLM_MODEL
    updated["provenance"] = provenance
    return updated


def _update_text_field(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    value = str(source.get(key) or "").strip()
    if value:
        target[key] = value


def _merge_core_items(core: dict[str, Any], core_copy: dict[str, Any]) -> None:
    source_items = [item for item in _json_list(core_copy.get("items")) if isinstance(item, dict)]
    if not source_items:
        return
    by_type = {
        str(item.get("insight_type")): item
        for item in source_items
        if str(item.get("insight_type") or "").strip()
    }
    current_items = [item for item in _json_list(core.get("items")) if isinstance(item, dict)]
    for index, item in enumerate(current_items):
        source = by_type.get(str(item.get("insight_type")))
        if source is None and index < len(source_items):
            source = source_items[index]
        if not isinstance(source, dict):
            continue
        for key in ("title", "summary", "so_what", "why_important"):
            _update_text_field(item, source, key)
    core["items"] = current_items


def _merge_flow(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_steps = _json_list(_nested_get(display_copy, "interpretation_flow", "steps"))
    current = _json_dict(updated.get("interpretation_flow"))
    current_steps = [step for step in _json_list(current.get("steps")) if isinstance(step, dict)]
    for index, step in enumerate(current_steps):
        source = _find_display_item_source(source_steps, step, index)
        if isinstance(source, dict):
            _update_text_field(step, source, "one_liner")
    if current_steps:
        current["steps"] = current_steps
        updated["interpretation_flow"] = current


def _merge_market_reading(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_items = _json_list(display_copy.get("market_reading"))
    current_items = [
        item for item in _json_list(updated.get("market_reading")) if isinstance(item, dict)
    ]
    for index, item in enumerate(current_items):
        source = _find_display_item_source(source_items, item, index)
        if isinstance(source, dict):
            _update_text_field(item, source, "title")
            _update_text_field(item, source, "description")
    if current_items:
        updated["market_reading"] = current_items[:_MAX_MARKET_ITEMS]


def _merge_sk_ax_view(updated: dict[str, Any], display_copy: dict[str, Any]) -> None:
    source_items = _json_list(display_copy.get("sk_ax_view"))
    current_items = [
        item for item in _json_list(updated.get("sk_ax_view")) if isinstance(item, dict)
    ]
    for index, item in enumerate(current_items):
        source = _find_display_item_source(source_items, item, index)
        if isinstance(source, dict):
            _update_text_field(item, source, "use_case")
            _update_text_field(item, source, "title")
            _update_text_field(item, source, "description")
    if current_items:
        updated["sk_ax_view"] = current_items[:_MAX_SKAX_ITEMS]


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
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
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


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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

    return {
        "id": report_id,
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "title": report_title,
        "briefing_type": briefing_type,
        "date_from": period["date_from"].isoformat(),
        "date_to": period["date_to"].isoformat(),
        "period_label": period["label"],
        "requested_by_user_id": requested_by_user_id,
        "status": "completed",
        "progress": 1.0,
        "key_summary": key_summary,
        "sk_implication": sk_ax_implication,
        "briefing_lead": briefing_lead or _briefing_lead(period, key_summary, selected_cards),
        "selected_cards": _public_selected_cards(selected_cards),
        "related_card_ids": source_card_ids,
        "primary_card_news_id": source_card_ids[0] if source_card_ids else None,
        "primary_peer_company_id": selected_cards[0].get("peer_id") if selected_cards else None,
        "core_change": _core_change_payload(period, selected_cards, briefing_basis),
        "interpretation_flow": _interpretation_flow_payload(briefing_basis, selected_cards),
        "market_reading": _market_reading_payload(briefing_basis, selected_cards),
        "sk_ax_view": _sk_ax_view_payload(briefing_basis, selected_cards),
        "hidden_details": _hidden_details(selected_cards, briefing_basis),
        "briefing_basis": briefing_basis,
        "confidence": confidence,
        "provenance": {
            **provenance_base,
            "agent": "BriefingGenerationAgent",
            "prompt_version": _PROMPT_VERSION,
            "source_card_ids": source_card_ids,
            "briefing_analysis_basis": (
                "card_id -> card_news.evidence_payload "
                "integrated_issue+analysis+implication+classification+validation"
            ),
        },
    }


def _empty_report(
    *,
    report_id: str,
    briefing_type: BriefingType,
    period: dict[str, Any],
    title: str | None,
    requested_by_user_id: int | None,
    provenance_base: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": report_id,
        "agent": "BriefingGenerationAgent",
        "prompt_version": _PROMPT_VERSION,
        "title": title or f"{period['label']} 브리핑",
        "briefing_type": briefing_type,
        "date_from": period["date_from"].isoformat(),
        "date_to": period["date_to"].isoformat(),
        "period_label": period["label"],
        "requested_by_user_id": requested_by_user_id,
        "status": "failed",
        "progress": 1.0,
        "error_message": "기간 조건에 맞는 카드뉴스가 없습니다.",
        "key_summary": "",
        "briefing_lead": "",
        "selected_cards": [],
        "related_card_ids": [],
        "primary_card_news_id": None,
        "core_change": {"label": "CORE CHANGE", "title": "", "summary": "", "items": []},
        "interpretation_flow": {
            "label": "INTERPRETATION FLOW",
            "title": "해석 흐름",
            "steps": [],
        },
        "market_reading": [],
        "sk_ax_view": [],
        "hidden_details": [],
        "briefing_basis": None,
        "requested_card_ids": provenance_base.get("requested_card_ids", []),
        "confidence": 0.0,
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
    core = (
        briefing_basis.get("core_change")
        if isinstance(briefing_basis.get("core_change"), dict)
        else {}
    )
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
        return f"{cleaned[0]}와 {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, {cleaned[-1]}"


def _core_change_payload(
    period: dict[str, Any],
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> dict[str, Any]:
    common = (
        briefing_basis.get("common_pattern")
        if isinstance(briefing_basis.get("common_pattern"), dict)
        else {}
    )
    title = _clip_text(_display_core_title(selected_cards, briefing_basis), max_chars=140)
    summary = _brief_sentences(
        _display_core_summary(briefing_basis)
        or _first_text(common.get("rationale"), _briefing_lead(period, title)),
        max_sentences=2,
        max_chars=220,
    )
    return {
        "label": "CORE CHANGE",
        "title": title,
        "summary": summary,
        "items": _core_change_insight_items(selected_cards, briefing_basis),
    }


def _core_change_insight_items(
    selected_cards: list[dict[str, Any]],
    briefing_basis: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence_ids = [card["id"] for card in selected_cards if card.get("id")]
    market_block = (
        briefing_basis.get("common_pattern")
        if isinstance(briefing_basis.get("common_pattern"), dict)
        else {}
    )
    competitor_block = (
        briefing_basis.get("hidden_conclusion")
        if isinstance(briefing_basis.get("hidden_conclusion"), dict)
        else {}
    )
    market_title = _clip_text(
        _first_text(
            market_block.get("finding"),
            briefing_basis.get("briefing_insight"),
        ),
        max_chars=140,
    )
    competitor_title = _clip_text(
        _first_text(
            competitor_block.get("finding"),
        ),
        max_chars=140,
    )
    return [
        {
            "seq": 1,
            "display_label": "시장 신호",
            "insight_type": "market_signal",
            "title": market_title,
            "summary": _brief_sentences(
                _first_text(
                    market_block.get("rationale"),
                    _display_core_summary(briefing_basis),
                ),
                max_sentences=2,
                max_chars=180,
            ),
            "so_what": _brief_sentences(
                _first_text(
                    _block_text(briefing_basis.get("hidden_conclusion"), "finding"),
                    market_title,
                ),
                max_sentences=2,
                max_chars=180,
            ),
            "evidence_card_ids": evidence_ids,
        },
        {
            "seq": 2,
            "display_label": "경쟁사 움직임",
            "insight_type": "competitor_move",
            "title": competitor_title,
            "summary": _brief_sentences(
                _first_text(
                    competitor_block.get("rationale"),
                    _competitor_move_summary(selected_cards),
                ),
                max_sentences=2,
                max_chars=180,
            ),
            "so_what": _brief_sentences(
                _first_text(
                    competitor_block.get("rationale"),
                    _block_text(briefing_basis.get("comparison_point"), "finding"),
                    competitor_title,
                ),
                max_sentences=2,
                max_chars=180,
            ),
            "evidence_card_ids": evidence_ids,
        },
    ]


def _competitor_move_summary(selected_cards: list[dict[str, Any]]) -> str:
    phrases: list[str] = []
    for card in selected_cards[:3]:
        company = _company_label(card)
        signal = _brief_sentence(_card_summary(card), max_chars=80)
        if company and signal:
            phrases.append(f"{company}: {signal}")
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
            "title": "해석 흐름",
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
        "title": "해석 흐름",
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
    lines = card.get("summary_lines") if isinstance(card.get("summary_lines"), list) else []
    return " ".join(str(line).strip() for line in lines[:2] if str(line).strip())


def _card_display_title(card: dict[str, Any]) -> str:
    package = _analysis_package(card)
    return _brief_sentence(
        _first_text(
            _nested_get(package, "integrated_issue", "main_issue"),
            card.get("title"),
        )
    )


def _company_label(card: dict[str, Any]) -> str:
    value = str(card.get("peer_id") or card.get("company") or "").strip()
    labels = {
        "hyundai_autoever": "현대오토에버",
        "lg_cns": "LG CNS",
        "samsung_sds": "삼성SDS",
        "posco_dx": "포스코DX",
    }
    return labels.get(value, value)


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
    return {
        "seq": step.get("seq") or step.get("step_idx") or 0,
        "label": str(step.get("label") or step.get("phase") or ""),
        "one_liner": _brief_sentence(step.get("one_liner") or step.get("answer")),
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
    details = []
    for card in cards:
        sources = _json_list(card.get("sources"))
        package = _analysis_package(card)
        details.append(
            {
                "card_id": card.get("id"),
                "evidence_card_ids": card.get("evidence_card_ids") or [card.get("id")],
                "source_raw_article_ids": card.get("source_raw_article_ids") or [],
                "sources": sources,
                "confidence": _first_text(
                    _nested_get(package, "validation", "sc_score"),
                    _nested_get(package, "analysis", "confidence"),
                    _nested_get(package, "implication", "confidence"),
                ),
                "evidence_text": _evidence_texts(package),
                "analysis_package": package,
            }
        )
    if briefing_basis:
        details.append(
            {
                "type": "briefing_basis",
                "evidence_card_ids": _json_list(
                    _nested_get(briefing_basis, "provenance", "source_card_ids")
                ),
                "confidence": briefing_basis.get("confidence"),
                "payload": briefing_basis,
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


def _clip_text(value: str, max_chars: int) -> str:
    text_value = str(value or "").strip()
    if len(text_value) <= max_chars:
        return text_value
    return text_value[: max_chars - 1].rstrip() + "…"


def _analysis_package(card: dict[str, Any]) -> dict[str, Any]:
    return _analysis_package_from_sources(card, _json_dict(card.get("evidence_payload")))


def _analysis_package_from_sources(*sources: object) -> dict[str, Any]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        package = source.get("analysis_package")
        if isinstance(package, dict):
            return package
    return {}


def _nested_get(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _int_list(value: object) -> list[int]:
    out: list[int] = []
    for item in _json_list(value):
        parsed = _optional_int(item)
        if parsed is not None and parsed not in out:
            out.append(parsed)
    return out


def _first_int(value: object) -> int | None:
    values = _int_list(value)
    return values[0] if values else None


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_source_published_at(sources: list[Any]) -> object:
    for source in sources:
        if isinstance(source, dict) and source.get("published_at"):
            return source.get("published_at")
    return None


def _first_from_list(value: object) -> str:
    values = _json_list(value)
    return str(values[0]) if values else ""


def _first_text(*values: object) -> str:
    for value in values:
        text_value = str(value or "").strip()
        if text_value:
            return text_value
    return ""


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(KST).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _frontend_display_payload(result: dict[str, Any]) -> dict[str, Any]:
    visible = {
        "title": result.get("title"),
        "briefing_lead": result.get("briefing_lead"),
        "core_change": result.get("core_change"),
        "interpretation_flow": result.get("interpretation_flow"),
        "market_reading": result.get("market_reading"),
        "sk_ax_view": result.get("sk_ax_view"),
        "related_card_ids": result.get("related_card_ids"),
        "primary_card_news_id": result.get("primary_card_news_id"),
        "hidden_details_count": len(result.get("hidden_details") or []),
    }
    return _strip_default_hidden_fields(visible)


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
        help="Skip BriefingGenerationAgent display-copy refinement.",
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
