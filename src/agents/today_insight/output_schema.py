"""today_insight output_schema — extracted from facade (move-only)."""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Mapping

from sqlalchemy import text

from src.agents.today_insight._config import _LLM_MODEL, _PROMPT_VERSION
from src.agents.today_insight.text_processing import (  # noqa: F401
    _clamp_float,
    _clip,
    _compact_sources,
    _dedupe,
    _fallback_actions,
    _fallback_headline,
    _fallback_implication,
    _fallback_signals,
    _fallback_summary,
    _is_before_window,
    _is_weak_or_mock_text,
    _json_dumps,
    _list,
    _sanitize_public_text,
    _section_actions,
    _slug,
    _string_list,
)
from src.contracts.today_insight_schemas import (
    TodayInsightGenerateResponse,
)
from src.db.postgres import SessionLocal
from src.services.today_insight_comparison_engine import (
    build_ui_change_summary,
    format_evidence_change_lines,
    polish_executive_output,
)

log = logging.getLogger(__name__)


_SIGNAL_LABELS = ("주요 신호", "관찰 포인트", "다음 판단")


def _derive_insight_state(base: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    """today_signal | quiet.

    오늘(anchor) 신호가 있으면 today_signal, 당일 신호가 없거나 placeholder 일 때만 quiet.
    홈은 "오늘 들어온 것 중 중요한 것"을 보여주는 화면이므로 내부 salience 컷으로
    실제 당일 신호를 숨기지 않는다.
    """
    provenance = base.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("is_status_placeholder"):
        return "quiet"
    if bool(context.get("has_current_signal", False)):
        return "today_signal"
    return "quiet"


def _build_week_synthesis(base: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    """조용한 날용 결정적 1줄 종합 (LLM 호출 없음 — quiet 는 LLM 을 타지 않음)."""
    coverage = base.get("coverage_stats")
    reviewed = int(coverage.get("reviewed_last_7d", 0)) if isinstance(coverage, Mapping) else 0
    comparison = base.get("comparison_facts") or context.get("comparison_facts")
    top_label = ""
    if isinstance(comparison, Mapping):
        candidates = comparison.get("salience_candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
            peer = str(candidates[0].get("peer_label") or "").strip()
            event = str(candidates[0].get("event_type") or "").strip()
            if peer:
                top_label = peer + (f"·{event}" if event and event != "-" else "")
    if top_label:
        return (
            f"최근 7일 peer 4사 동향 {reviewed}건을 검토했고 {top_label} 등이 관찰됐으나, "
            "오늘 새로 부상한 주요 전략 신호는 없습니다."
        )
    return (
        f"최근 7일 peer 4사 동향 {reviewed}건을 모니터링했으며, "
        "오늘 새로 부상한 주요 전략 신호는 없습니다."
    )


def _build_coverage_stats(*, window_days: int = 7) -> dict[str, Any]:
    """모니터링 안심 스트립용 결정적 DB 카운트 (실패해도 기본값 반환)."""
    stats: dict[str, Any] = {
        "window_days": window_days,
        "reviewed_last_7d": 0,
        "cards_last_7d": 0,
        "urgent_last_7d": 0,
        "peers_monitored": 4,
        "peer_names": ["삼성SDS", "LG CNS", "현대오토에버", "포스코DX"],
    }
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                        SELECT
                          (SELECT COUNT(*) FROM raw_articles
                             WHERE created_at >= now() - make_interval(days => :d)) AS reviewed,
                          (SELECT COUNT(*) FROM card_news
                             WHERE created_at >= now() - make_interval(days => :d)
                               AND LOWER(status) = 'active') AS cards,
                          (SELECT COUNT(*) FROM sent_alerts
                             WHERE sent_at >= now() - make_interval(days => :d)) AS urgent
                        """
                    ),
                    {"d": window_days},
                )
                .mappings()
                .first()
            )
            if row is not None:
                stats["reviewed_last_7d"] = int(row["reviewed"] or 0)
                stats["cards_last_7d"] = int(row["cards"] or 0)
                stats["urgent_last_7d"] = int(row["urgent"] or 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight coverage stats failed | %s", exc)
    return stats


def _apply_insight_state(base: dict[str, Any], context: Mapping[str, Any]) -> None:
    """base 에 state/signal_date/week_synthesis/coverage_stats 를 얹는다.

    페이로드(signals/insight_sections)는 그대로 두고 state 만 정한다. quiet 면
    프론트가 깊은 3섹션 대신 week_synthesis 를 렌더해 변두리 필러 분석 노출을 막는다.
    """
    base["coverage_stats"] = _build_coverage_stats(window_days=7)
    state = _derive_insight_state(base, context)
    base["state"] = state
    if state == "today_signal":
        base["signal_date"] = (str(base.get("report_date") or "")[:10]) or None
        base["week_synthesis"] = None
        return
    base["signal_date"] = None
    base["week_synthesis"] = _build_week_synthesis(base, context)
    provenance = base.get("provenance")
    if isinstance(provenance, dict):
        provenance.setdefault("depth_gate", "below_bar")


def _normalize_result(
    data: Mapping[str, Any],
    *,
    anchor_date: date,
    context: dict[str, Any],
) -> dict[str, Any]:
    base = dict(data)
    base["report_date"] = anchor_date.isoformat()
    base["generated_at"] = datetime.now(UTC).isoformat()
    base.setdefault("headline", "")
    base.setdefault("executive_summary", "")
    base.setdefault("executive_implication", "")
    comparison_facts = context.get("comparison_facts")
    default_change_summary = build_ui_change_summary(
        default_rows=context["change_stats"]["default_change_summary"],
        comparison_facts=comparison_facts if isinstance(comparison_facts, dict) else None,
    )
    base["change_summary"] = _normalize_change_summary(
        base.get("change_summary"),
        default_change_summary,
    )
    if isinstance(comparison_facts, dict):
        base["comparison_facts"] = comparison_facts
    base["signals"] = _normalize_signals(base.get("signals"), context=context)
    base["response_direction"] = _normalize_actions(base.get("response_direction"), context=context)
    base["sources"] = _normalize_sources(base.get("sources"), context["sources"])
    base["source_integrated_issue_ids"] = _dedupe(
        [
            str(row.get("id"))
            for row in context["current_issues"]
            if isinstance(row, dict) and row.get("id")
        ],
        limit=12,
    )
    base["source_card_ids"] = _dedupe(
        [
            str(card.get("id"))
            for card in context["recent_cards"]
            if isinstance(card, dict) and card.get("id")
        ],
        limit=12,
    )
    base["source_trace"] = _normalize_source_trace(base.get("source_trace"), context=context)
    base["insight_sections"] = _normalize_insight_sections(
        base.get("insight_sections"),
        signals=base["signals"],
        actions=base["response_direction"],
        sources=base["sources"],
        source_trace=base["source_trace"],
        context=context,
    )
    base["peer_ids"] = _dedupe(
        [
            str(value)
            for row in [*context["current_issues"], *context["recent_cards"]]
            if isinstance(row, dict)
            for value in [row.get("main_company") or row.get("peer_id")]
            if value
        ],
        limit=10,
    )
    base["sectors"] = _dedupe(
        [
            str(sector)
            for row in context["current_issues"]
            if isinstance(row, dict)
            for sector in _list(row.get("sectors"))
            if sector
        ],
        limit=10,
    )
    base["confidence"] = _clamp_float(base.get("confidence"), default=0.65)
    raw_provenance = base.get("provenance")
    provenance: dict[str, Any] = raw_provenance if isinstance(raw_provenance, dict) else {}
    base["provenance"] = {
        **provenance,
        "llm_model": _LLM_MODEL,
        "prompt_version": _PROMPT_VERSION,
        "context_counts": {
            "current_issues": len(context["current_issues"]),
            "history_issues": len(context["history_issues"]),
            "recent_cards": len(context["recent_cards"]),
            "prior_reports": len(context["prior_today_insight_memory"]),
            "ledger_items": len(context["analysis_ledger_context"]),
        },
        "current_input_policy": "anchor_date_peer_only",
        "comparison_coverage": (
            (context.get("comparison_facts") or {}).get("coverage")
            if isinstance(context.get("comparison_facts"), dict)
            else {}
        ),
        "llm_context": context.get("llm_context_meta") or {},
    }
    if not base["headline"] or _is_weak_or_mock_text(str(base["headline"])):
        base["headline"] = _fallback_headline(context)
    if not base["executive_summary"] or _is_weak_or_mock_text(str(base["executive_summary"])):
        base["executive_summary"] = _fallback_summary(context)
    if not base["executive_implication"] or _is_weak_or_mock_text(
        str(base["executive_implication"])
    ):
        base["executive_implication"] = _fallback_implication(context)
    base = polish_executive_output(base, context=context)
    if _is_weak_or_mock_text(str(base.get("executive_implication") or "")):
        base["executive_implication"] = _fallback_implication(context)
    base["headline"] = _clip(_sanitize_public_text(base.get("headline")), 120)
    base["executive_summary"] = _clip(_sanitize_public_text(base.get("executive_summary")), 500)
    base["executive_implication"] = _clip(
        _sanitize_public_text(base.get("executive_implication")),
        500,
    )
    base["insight_sections"] = _normalize_insight_sections(
        base.get("insight_sections"),
        signals=base["signals"],
        actions=base["response_direction"],
        sources=base["sources"],
        source_trace=base["source_trace"],
        context=context,
    )
    base["memory_document"] = _build_memory_document(
        base,
        context=context,
        anchor_date=anchor_date,
    )
    _apply_insight_state(base, context)
    response = TodayInsightGenerateResponse.model_validate(base)
    return response.model_dump()


def _normalize_change_summary(value: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    normalized = []
    for item in rows[:3]:
        label = _clip(_sanitize_public_text(item.get("label") or ""), 24)
        val = _clip(_sanitize_public_text(item.get("value") or ""), 36)
        if label and val:
            normalized.append({"label": label, "value": val})
    while len(normalized) < 3:
        normalized.append(fallback[len(normalized)])
    return normalized[:3]


def _normalize_signals(value: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    fallback = _fallback_signals(context)
    normalized: list[dict[str, Any]] = []
    for idx in range(len(_SIGNAL_LABELS)):
        item = rows[idx] if idx < len(rows) else fallback[idx]
        raw_evidence = item.get("evidence")
        evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
        reasoning: list[dict[str, Any]] = [
            step for step in _list(item.get("reasoning")) if isinstance(step, dict)
        ][:4]
        if not reasoning:
            reasoning = fallback[idx]["reasoning"]
        raw_value = _sanitize_public_text(item.get("value") or "")
        if _is_weak_or_mock_text(raw_value):
            raw_value = ""
        signal = {
            "id": _slug(str(item.get("id") or f"signal-{idx + 1}")),
            "label": _SIGNAL_LABELS[idx],
            "value": _clip(raw_value or _sanitize_public_text(fallback[idx]["value"]), 220),
            "reasoning": [
                {
                    "stage": _clip(
                        str(step.get("stage") or fallback[idx]["reasoning"][0]["stage"]), 14
                    ),
                    "detail": _clip(_sanitize_public_text(step.get("detail") or ""), 150),
                }
                for step in reasoning
                if step.get("detail")
            ][:4],
            "evidence": {
                "grounds": _string_list(
                    evidence.get("grounds"), fallback[idx]["evidence"]["grounds"], 3
                ),
                "changes": _string_list(
                    evidence.get("changes"), fallback[idx]["evidence"]["changes"], 3
                ),
                "related_keywords": _string_list(
                    evidence.get("related_keywords") or evidence.get("relatedKeywords"),
                    fallback[idx]["evidence"]["related_keywords"],
                    6,
                ),
                "source_ids": _string_list(
                    evidence.get("source_ids") or evidence.get("sourceIds"),
                    fallback[idx]["evidence"]["source_ids"],
                    6,
                    max_len=80,
                ),
            },
        }
        if not signal["reasoning"]:
            signal["reasoning"] = fallback[idx]["reasoning"]
        signal = _merge_comparison_evidence(signal, context=context, idx=idx)
        normalized.append(signal)
    return normalized


def _merge_comparison_evidence(
    signal: dict[str, Any],
    *,
    context: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    comparison = context.get("comparison_facts")
    if not isinstance(comparison, dict):
        return signal

    computed_changes = format_evidence_change_lines(comparison)
    if not computed_changes:
        return signal

    evidence = signal.get("evidence")
    if not isinstance(evidence, dict):
        return signal

    if computed_changes:
        evidence["changes"] = computed_changes[:3]
    signal["evidence"] = evidence
    return signal


def _normalize_actions(value: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    fallback = _fallback_actions(context)
    normalized = []
    for idx in range(3):
        item = rows[idx] if idx < len(rows) else fallback[idx]
        action = _sanitize_public_text(item.get("action") or "")
        rationale = _sanitize_public_text(item.get("rationale") or "")
        if _is_weak_or_mock_text(action):
            action = ""
        if _is_weak_or_mock_text(rationale):
            rationale = ""
        normalized.append(
            {
                "action": _clip(action or _sanitize_public_text(fallback[idx]["action"]), 180),
                "decision_owner": _clip(
                    _sanitize_public_text(
                        item.get("decision_owner") or fallback[idx]["decision_owner"]
                    ),
                    48,
                ),
                "time_horizon": _clip(
                    _sanitize_public_text(
                        item.get("time_horizon") or fallback[idx]["time_horizon"]
                    ),
                    32,
                ),
                "rationale": _clip(
                    rationale or _sanitize_public_text(fallback[idx]["rationale"]),
                    160,
                ),
                "evidence_refs": _string_list(
                    item.get("evidence_refs") or item.get("evidenceRefs"),
                    fallback[idx]["evidence_refs"],
                    5,
                    max_len=80,
                ),
            }
        )
    return normalized


def _normalize_sources(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Prefer deterministic DB-derived sources because they carry raw article URLs.
    # LLM-returned sources are still kept as supplemental metadata.
    rows = [
        item
        for item in [*fallback, *[row for row in _list(value) if isinstance(row, dict)]]
        if isinstance(item, dict)
    ]
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        source_id = str(item.get("id") or item.get("source_id") or item.get("url") or "")
        title = _sanitize_public_text(item.get("title") or item.get("headline") or "")
        if not source_id and not title:
            continue
        normalized_item = {
            "id": _clip(source_id or f"source-{len(normalized) + 1}", 120),
            "title": _clip(title, 180),
            "source_name": _clip(
                _sanitize_public_text(item.get("source_name") or item.get("source") or ""),
                80,
            ),
            "publisher": _clip(_sanitize_public_text(item.get("publisher") or ""), 80),
            "related_companies": _string_list(
                item.get("related_companies") or item.get("relatedCompanies"),
                [],
                6,
                max_len=40,
            ),
            "url": _clip(str(item.get("url") or ""), 500),
            "published_at": str(item.get("published_at") or item.get("created_at") or "") or None,
        }
        key = str(
            normalized_item.get("url")
            or normalized_item.get("id")
            or normalized_item.get("title")
            or ""
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_item)
    return normalized[:8]


def _normalize_source_trace(value: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    fallback = _source_trace_from_context(context)
    normalized: list[dict[str, Any]] = []
    for item in (rows or fallback)[:16]:
        issue_id = str(
            item.get("source_integrated_issue_id") or item.get("integrated_issue_id") or ""
        )
        card_id = str(item.get("source_card_id") or item.get("card_id") or "")
        raw_ids = [
            str(raw_id)
            for raw_id in _list(item.get("source_raw_article_ids") or item.get("raw_article_ids"))
            if str(raw_id or "").strip()
        ][:8]
        title = _sanitize_public_text(item.get("title") or "")
        url = str(item.get("url") or "")
        if not issue_id and not card_id and not raw_ids and not title:
            continue
        normalized.append(
            {
                "source_integrated_issue_id": _clip(issue_id, 120),
                "source_card_id": _clip(card_id, 120),
                "source_raw_article_ids": raw_ids,
                "title": _clip(title, 180),
                "url": _clip(url, 500),
            }
        )
    return normalized


def _normalize_insight_sections(
    value: Any,
    *,
    signals: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    source_trace: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    sections: list[dict[str, Any]] = []
    for idx in range(len(_SIGNAL_LABELS)):
        signal = signals[idx] if idx < len(signals) else _fallback_signals(context)[idx]
        raw = rows[idx] if idx < len(rows) else {}
        signal_evidence_raw = signal.get("evidence")
        signal_evidence: dict[str, Any] = (
            signal_evidence_raw if isinstance(signal_evidence_raw, dict) else {}
        )
        raw_evidence_raw = raw.get("evidence")
        raw_evidence: dict[str, Any] = (
            raw_evidence_raw if isinstance(raw_evidence_raw, dict) else {}
        )
        evidence: dict[str, Any] = signal_evidence if signal_evidence else raw_evidence
        source_ids = set(
            str(source_id)
            for source_id in _list(evidence.get("source_ids") or evidence.get("sourceIds"))
            if str(source_id or "").strip()
        )
        matched_sources = _match_sources_by_ids(sources, source_ids)
        matched_trace = _match_trace_by_ids(source_trace, source_ids)
        section_actions = _section_actions(
            raw.get("response_direction") if isinstance(raw, dict) else None,
            actions=actions,
            source_ids=source_ids,
            idx=idx,
        )
        summary = _sanitize_public_text(raw.get("summary") or signal.get("value") or "")
        if _is_weak_or_mock_text(summary):
            summary = str(signal.get("value") or "")
        sections.append(
            {
                "id": _slug(str(raw.get("id") or signal.get("id") or f"section-{idx + 1}")),
                "label": _SIGNAL_LABELS[idx],
                "title": _clip(
                    _sanitize_public_text(raw.get("title") or signal.get("value") or ""),
                    220,
                ),
                "summary": _clip(summary, 260),
                "reasoning": signal.get("reasoning") or [],
                "evidence": {
                    "grounds": _string_list(
                        evidence.get("grounds"),
                        signal_evidence.get("grounds", []),
                        4,
                    ),
                    "changes": _string_list(
                        evidence.get("changes"),
                        signal_evidence.get("changes", []),
                        4,
                    ),
                    "related_keywords": _string_list(
                        evidence.get("related_keywords") or evidence.get("relatedKeywords"),
                        signal_evidence.get("related_keywords", []),
                        8,
                    ),
                    "source_ids": _string_list(
                        evidence.get("source_ids") or evidence.get("sourceIds"),
                        signal_evidence.get("source_ids", []),
                        8,
                        max_len=80,
                    ),
                },
                "response_direction": section_actions,
                "sources": matched_sources[:4],
                "source_trace": matched_trace[:6],
            }
        )
    return sections


def _norm_id_set(values: Any) -> set[str]:
    """출처 id 비교용 정규화 집합.

    대소문자 무시 + raw 원문 id 의 ``raw-123`` ↔ ``123`` 표기 차이를 흡수한다.
    (integrated issue 의 source_ids 는 숫자 raw id 인데 source 메타의 id 는 ``raw-N``
    으로 갈려 있어, LLM 이 어느 표기로 인용하든 매칭되도록 양방향 보강.) ``cn-``/``ic-``
    같은 prefix 는 그대로 두어 서로 다른 타입(raw vs card vs issue)이 우연히
    교차 매칭되지 않게 한다.
    """
    if isinstance(values, (set, frozenset, list, tuple)):
        items = list(values)
    else:
        items = _list(values)
    out: set[str] = set()
    for value in items:
        token = str(value or "").strip().lower()
        if not token:
            continue
        out.add(token)
        raw_match = re.match(r"^raw-(\d+)$", token)
        if raw_match:
            out.add(raw_match.group(1))
        elif token.isdigit():
            out.add(f"raw-{token}")
    return out


def _match_sources_by_ids(
    sources: list[dict[str, Any]],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    url_sources = [
        source
        for source in sources
        if str(source.get("url") or "").startswith(("http://", "https://"))
    ]
    if not source_ids:
        return (url_sources or sources)[:4]
    cited = _norm_id_set(source_ids)
    matched = [
        source for source in sources if _norm_id_set([source.get("id"), source.get("url")]) & cited
    ]
    matched_url_sources = [
        source
        for source in matched
        if str(source.get("url") or "").startswith(("http://", "https://"))
    ]
    # 인용된 source_id 와 실제로 매칭된 출처만 노출한다. 매칭 실패 시 임의의 다른
    # 출처를 붙이지 않는다 — 본문과 무관한 "출처 링크"(오링크) 방지.
    return matched_url_sources or matched


def _match_trace_by_ids(
    trace: list[dict[str, Any]],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    if not source_ids:
        return trace[:6]
    cited = _norm_id_set(source_ids)
    matched = []
    for row in trace:
        values = _norm_id_set(
            [
                row.get("source_integrated_issue_id"),
                row.get("source_card_id"),
                *_list(row.get("source_raw_article_ids")),
            ]
        )
        if values & cited:
            matched.append(row)
    # 매칭 실패 시 임의 trace 를 붙이지 않는다(오링크 방지).
    return matched


def _source_trace_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cards = [card for card in context.get("recent_cards", []) if isinstance(card, dict)]
    issues = [issue for issue in context.get("current_issues", []) if isinstance(issue, dict)]
    for card in cards:
        sources = _compact_sources(card.get("sources"), limit=1)
        source = sources[0] if sources else {}
        out.append(
            {
                "source_integrated_issue_id": str(card.get("integrated_issue_id") or ""),
                "source_card_id": str(card.get("id") or ""),
                "source_raw_article_ids": [
                    str(item) for item in _list(card.get("source_raw_article_ids"))
                ],
                "title": str(card.get("title") or source.get("title") or ""),
                "url": str(source.get("url") or ""),
            }
        )
    known_issue_ids = {str(item.get("source_integrated_issue_id") or "") for item in out}
    for issue in issues:
        issue_id = str(issue.get("id") or "")
        if not issue_id or issue_id in known_issue_ids:
            continue
        sources = _compact_sources(issue.get("sources"), limit=1)
        source = sources[0] if sources else {}
        out.append(
            {
                "source_integrated_issue_id": issue_id,
                "source_card_id": "",
                "source_raw_article_ids": [str(item) for item in _list(issue.get("source_ids"))],
                "title": str(
                    issue.get("headline")
                    or issue.get("one_line_summary")
                    or source.get("title")
                    or ""
                ),
                "url": str(source.get("url") or ""),
            }
        )
    return out


def _build_memory_document(
    result: dict[str, Any],
    *,
    context: dict[str, Any],
    anchor_date: date,
) -> dict[str, Any]:
    window_days = int(context.get("window_days") or 60)
    window_start = anchor_date - timedelta(days=max(window_days - 1, 0))
    source_trace = [item for item in _list(result.get("source_trace")) if isinstance(item, dict)][
        :16
    ]
    sections = [item for item in _list(result.get("insight_sections")) if isinstance(item, dict)][
        :3
    ]
    actions = [item for item in _list(result.get("response_direction")) if isinstance(item, dict)][
        :3
    ]
    comparison = context.get("comparison_facts")
    structural = (
        [item for item in _list(comparison.get("structural")) if isinstance(item, dict)]
        if isinstance(comparison, dict)
        else []
    )
    primary_selection = comparison.get("primary_selection") if isinstance(comparison, dict) else {}
    primary_selection = primary_selection if isinstance(primary_selection, dict) else {}
    primary_items = (
        [item for item in _list(primary_selection.get("items")) if isinstance(item, dict)]
        if isinstance(comparison, dict)
        else []
    )

    observed_facts: list[dict[str, Any]] = []
    for section in sections:
        raw_evidence = section.get("evidence")
        evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
        observed_facts.append(
            {
                "label": section.get("label") or "",
                "fact": _clip(
                    str(section.get("title") or section.get("summary") or ""),
                    240,
                ),
                "grounds": _string_list(evidence.get("grounds"), [], 4, max_len=180),
                "changes": _string_list(evidence.get("changes"), [], 4, max_len=180),
                "source_ids": _string_list(evidence.get("source_ids"), [], 8, max_len=120),
            }
        )

    important_memory: list[dict[str, Any]] = []
    if result.get("headline"):
        important_memory.append(
            {
                "type": "headline",
                "content": _clip(str(result.get("headline")), 180),
                "why_it_matters": _clip(str(result.get("executive_implication") or ""), 260),
                "source_ids": _string_list(result.get("source_card_ids"), [], 8, max_len=120),
            }
        )
    for item in primary_items[:3]:
        important_memory.append(
            {
                "type": "primary_selection",
                "content": _clip(str(item.get("title") or ""), 180),
                "peer_id": item.get("peer_id"),
                "sector": item.get("sector"),
                "event_type": item.get("event_type"),
                "label": item.get("label"),
                "salience_score": item.get("salience_score"),
                "exposure_score": item.get("exposure_score"),
                "source_id": item.get("id"),
            }
        )

    next_analysis_hints: list[dict[str, Any]] = []
    for action in actions:
        next_analysis_hints.append(
            {
                "watch_item": _clip(str(action.get("action") or ""), 220),
                "rationale": _clip(str(action.get("rationale") or ""), 220),
                "owner": _clip(str(action.get("decision_owner") or ""), 80),
                "time_horizon": _clip(str(action.get("time_horizon") or ""), 60),
                "evidence_refs": _string_list(action.get("evidence_refs"), [], 8, max_len=120),
            }
        )
    for metric in structural[:4]:
        next_analysis_hints.append(
            {
                "watch_item": _clip(str(metric.get("metric") or ""), 80),
                "rationale": _clip(_json_dumps(metric), 260),
                "owner": "",
                "time_horizon": "다음 분석",
                "evidence_refs": [],
            }
        )

    prior_reports = [
        item for item in _list(context.get("prior_today_insight_memory")) if isinstance(item, dict)
    ]
    pruned_items: list[dict[str, Any]] = [
        {
            "report_date": str(item.get("report_date") or ""),
            "headline": _clip(str(item.get("headline") or ""), 160),
            "reason": f"{window_days}일 분석 창 밖이면 다음 누적 문서에서 제외",
        }
        for item in prior_reports
        if _is_before_window(str(item.get("report_date") or ""), window_start)
    ][:20]

    source_updated_dates = _dedupe(
        [
            str(value)
            for row in [
                *[item for item in _list(context.get("current_issues")) if isinstance(item, dict)],
                *[item for item in _list(context.get("recent_cards")) if isinstance(item, dict)],
            ]
            for value in [
                row.get("published_at"),
                row.get("created_date_kst"),
                row.get("created_at"),
                row.get("updated_at"),
            ]
            if value
        ],
        limit=20,
    )
    return {
        "update_window": {
            "from": window_start.isoformat(),
            "to": anchor_date.isoformat(),
            "window_days": window_days,
            "source_updated_dates": source_updated_dates,
            "retention_rule": f"{window_days}일 초과 항목은 pruned_items 후보로 기록",
        },
        "observed_facts": [item for item in observed_facts if item.get("fact")],
        "important_memory": [
            item for item in important_memory if item.get("content") or item.get("source_id")
        ][:8],
        "next_analysis_hints": [item for item in next_analysis_hints if item.get("watch_item")][
            :10
        ],
        "source_trace": source_trace,
        "pruned_items": pruned_items,
    }
