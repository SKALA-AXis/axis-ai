# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — Today's Insight dual-lane 비교 엔진 구축 및 카피/모델(gpt-5.5) 개선
#   2026-06-10 박진 — mixer 챗봇 today insight 플로우 개선
#   2026-06-12 안가은 — 대시보드 키워드 트렌드 파이프라인 반영
"""Deterministic comparison facts for Today's Insight (dual-lane: salience + volume).

Lane 1 — Event salience: definite strategic triggers (M&A, DART/IR, high-impact keywords)
         independent of media volume.
Lane 2 — Structural volume: peer/sector counts, intensity, recurrence vs rolling baseline.

LLM must narrate only from emitted facts; it must not invent numbers.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import Any, Literal

from sqlalchemy import text

from src.config.companies import COMPANIES
from src.db.postgres import SessionLocal
from src.preprocessing.classification import compute_article_impact

log = logging.getLogger(__name__)

CandidateKind = Literal["card", "integrated_issue"]

_SALIENCE_PRIMARY_THRESHOLD = 0.70
_EXPOSURE_LOW_THRESHOLD = 0.40
_VISIBILITY_GAP_MIN = 0.30
_CLUSTER_SIZE_SATURATION = 5

_OFFICIAL_SOURCE_MARKERS = (
    "dart",
    "ir",
    "ir_pdf",
    "official",
    "company_site",
    "securities_report",
    "공시",
    "전자공시",
)

_STRATEGIC_SECTOR_BOOST = {
    "security": 0.05,
    "ax": 0.03,
    "infra": 0.02,
    "deal": 0.02,
}

# Dashboard keyword chart 와 동일한 네이버 DataLab 그룹 (raw_articles.search_trend).
_DEFAULT_KEYWORD_GROUPS = (
    "AI 에이전트",
    "LLM",
    "인프라",
    "네트워크",
    "AX",
    "생성형 AI",
    "사이버보안",
    "SOC",
    "제조 AX",
    "스마트팩토리",
)

_SECTOR_TO_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "security": ("사이버보안", "SOC", "네트워크"),
    "ax": ("AX", "생성형 AI", "AI 에이전트", "LLM"),
    "infra": ("인프라", "IT 인프라", "네트워크", "GPU"),
    "deal": ("AX", "제조 AX"),
    "other": ("AI 에이전트", "LLM"),
}

_KEYWORD_TREND_HISTORY_DAYS = 7

_EVENT_TYPE_LABELS: dict[str, str] = {
    "low_visibility_definite_event": "노출은 낮지만 내용이 확인된 이벤트",
    "high_salience_visible": "보도 확산이 큰 이벤트",
    "general_update": "일반 업데이트",
    "major_contract": "계약·협약 이벤트",
    "partnership": "제휴",
    "product_launch": "제품·서비스 출시",
    "ma": "M&A",
    "personnel": "인사",
    "tech_release": "기술 출시",
    "regulation": "규제",
    "contract": "계약",
    "financial": "실적·공시",
    "expansion": "사업 확장",
    "company": "사업",
    "investment": "투자·계약",
    "security": "보안 관련 이벤트",
}


def _event_type_label(event_type: str) -> str:
    normalized = str(event_type or "").strip().lower()
    if not normalized or normalized == "-":
        return "관련"
    return _EVENT_TYPE_LABELS.get(normalized, normalized.replace("_", " "))


def _impact_adjective(score: float) -> str:
    if score >= 0.85:
        return "매우 큰"
    if score >= 0.70:
        return "큰"
    if score >= 0.55:
        return "있는"
    return "제한적인"


def _exposure_clause(score: float) -> str:
    if score >= 0.60:
        return "보도 확산도 함께 높습니다"
    if score >= 0.40:
        return "보도는 보통 수준입니다"
    return "보도는 아직 많지 않습니다"


def build_comparison_facts(
    *,
    anchor_date: date,
    window_days: int,
    current_issues: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    change_stats: dict[str, Any],
    prior_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build LLM-ready comparison_facts JSON from structured inputs only."""
    salience_candidates = _build_salience_candidates(
        current_issues=current_issues,
        cards=cards,
        history_issues=history_issues,
        window_days=window_days,
    )
    keyword_groups = _relevant_keyword_groups(current_issues=current_issues, cards=cards)
    keyword_trends = _load_keyword_trend_facts(
        anchor_date=anchor_date,
        groups=keyword_groups,
        history_days=_KEYWORD_TREND_HISTORY_DAYS,
    )
    structural = _build_structural_facts(
        change_stats=change_stats,
        current_issues=current_issues,
        history_issues=history_issues,
        cards=cards,
        window_days=window_days,
        keyword_trends=keyword_trends,
    )
    visibility_gaps = [
        item for item in salience_candidates if item.get("label") == "low_visibility_definite_event"
    ]
    primary_selection = _select_primary_lane(salience_candidates, visibility_gaps)
    context_selection = _select_context_lane(structural, salience_candidates)

    return {
        "anchor_date": anchor_date.isoformat(),
        "window_days": window_days,
        "coverage": _coverage_summary(
            salience_candidates=salience_candidates,
            structural=structural,
            visibility_gaps=visibility_gaps,
            keyword_trends=keyword_trends,
        ),
        "salience_candidates": salience_candidates[:12],
        "keyword_trends": keyword_trends[:6],
        "structural": structural[:12],
        "visibility_gaps": visibility_gaps[:5],
        "primary_selection": primary_selection,
        "context_selection": context_selection,
        "prior_memory": _prior_memory_facts(prior_reports or []),
        "selection_rules": {
            "primary_lane": (
                "salience_score 기준. 확실한 이벤트(M&A, 공시, 고임팩트 키워드)가 "
                "보도량보다 우선한다."
            ),
            "context_lane": (
                "structural 지표는 필터가 아니라 맥락. "
                "노출이 낮아도 primary_lane 후보를 제거하지 않는다."
            ),
            "hidden_gem_rule": (
                "visibility_gaps 가 있으면 headline 또는 signal[0]에 최소 1건을 반드시 반영한다."
            ),
            "number_rule": (
                "comparison_facts 에 없는 수치를 생성하지 않는다. "
                "card_attached_numbers 가 비어 있으면 금액/규모 수치를 쓰지 않는다."
            ),
            "keyword_trend_rule": (
                "keyword_trends 는 raw_articles.search_trend(DB) 상대 검색지수다. "
                "특정 카드뉴스와 1:1 인과 연결하지 말고 시장 관심 맥락으로만 쓴다."
            ),
        },
    }


def _build_salience_candidates(
    *,
    current_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    window_days: int,
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    out: list[dict[str, Any]] = []

    for issue in current_issues:
        issue_id = str(issue.get("id") or "").strip()
        if not issue_id or issue_id in seen_ids:
            continue
        seen_ids.add(issue_id)
        out.append(
            _salience_for_issue(
                issue,
                history_issues=history_issues,
                window_days=window_days,
            )
        )

    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if not card_id or card_id in seen_ids:
            continue
        seen_ids.add(card_id)
        out.append(
            _salience_for_card(
                card,
                history_issues=history_issues,
                window_days=window_days,
            )
        )

    out.sort(
        key=lambda item: (
            float(item.get("salience_score") or 0.0),
            float(item.get("visibility_gap") or 0.0),
        ),
        reverse=True,
    )
    return out


def _salience_for_issue(
    issue: dict[str, Any],
    *,
    history_issues: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    event_type = str(issue.get("event_type") or "").strip().lower()
    headline = str(issue.get("headline") or issue.get("one_line_summary") or "")
    summary = str(issue.get("one_line_summary") or issue.get("content_summary") or "")
    source_type = _normalize_source_type(str(issue.get("source_family") or ""))

    impact = compute_article_impact(
        {"title": headline, "content": summary, "source_type": source_type},
        event_type or "company",
    )
    salience_score = float(impact["impact_score"])
    salience_triggers = list(impact.get("impact_signals") or [])

    sectors = _list(issue.get("sectors"))
    sector = str(sectors[0] if sectors else "").strip().lower()
    sector_boost = _STRATEGIC_SECTOR_BOOST.get(sector, 0.0)
    if sector_boost:
        salience_score = round(min(1.0, salience_score + sector_boost), 3)
        salience_triggers.append(f"sector:{sector}")

    exposure_score = _exposure_for_issue(issue)
    visibility_gap = round(max(0.0, salience_score - exposure_score), 3)
    peer_id = str(issue.get("main_company") or "")
    recurrence = _recurrence_count(
        peer_id=peer_id,
        event_type=event_type,
        history_issues=history_issues,
        exclude_id=str(issue.get("id") or ""),
    )

    return _candidate_payload(
        kind="integrated_issue",
        item_id=str(issue.get("id") or ""),
        title=headline or str(issue.get("id") or ""),
        peer_id=peer_id,
        sector=sector,
        event_type=event_type,
        salience_score=salience_score,
        exposure_score=exposure_score,
        visibility_gap=visibility_gap,
        salience_triggers=salience_triggers,
        recurrence_60d=recurrence,
        window_days=window_days,
        source_type=source_type,
    )


def _salience_for_card(
    card: dict[str, Any],
    *,
    history_issues: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    event_type = str(card.get("event_type") or "").strip().lower()
    title = str(card.get("title") or "")
    summary_lines = " / ".join(str(line) for line in _list(card.get("summary_lines"))[:3])
    source_type = _infer_source_type_from_card(card)

    impact = compute_article_impact(
        {"title": title, "content": summary_lines, "source_type": source_type},
        event_type or "company",
    )
    salience_score = float(impact["impact_score"])
    salience_triggers = list(impact.get("impact_signals") or [])

    sector = (
        str(
            card.get("sector")
            or card.get("primary_keyword_category")
            or _sector_from_implication(card)
            or ""
        )
        .strip()
        .lower()
    )
    sector_boost = _STRATEGIC_SECTOR_BOOST.get(sector, 0.0)
    if sector_boost:
        salience_score = round(min(1.0, salience_score + sector_boost), 3)
        salience_triggers.append(f"sector:{sector}")

    exposure_score = _exposure_for_card(card)
    visibility_gap = round(max(0.0, salience_score - exposure_score), 3)
    peer_id = str(card.get("peer_id") or card.get("company") or "")
    recurrence = _recurrence_count(
        peer_id=peer_id,
        event_type=event_type,
        history_issues=history_issues,
        exclude_id="",
    )

    return _candidate_payload(
        kind="card",
        item_id=str(card.get("id") or ""),
        title=title or str(card.get("id") or ""),
        peer_id=peer_id,
        sector=sector,
        event_type=event_type,
        salience_score=salience_score,
        exposure_score=exposure_score,
        visibility_gap=visibility_gap,
        salience_triggers=salience_triggers,
        recurrence_60d=recurrence,
        window_days=window_days,
        source_type=source_type,
        card_attached_numbers=_card_attached_numbers(card),
    )


def _candidate_payload(
    *,
    kind: CandidateKind,
    item_id: str,
    title: str,
    peer_id: str,
    sector: str,
    event_type: str,
    salience_score: float,
    exposure_score: float,
    visibility_gap: float,
    salience_triggers: list[str],
    recurrence_60d: int,
    window_days: int,
    source_type: str,
    card_attached_numbers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = _visibility_label(
        salience_score=salience_score,
        exposure_score=exposure_score,
        visibility_gap=visibility_gap,
    )
    recurrence_label = (
        "novel" if recurrence_60d == 0 else "recurring" if recurrence_60d >= 2 else "single_prior"
    )
    return {
        "kind": kind,
        "id": item_id,
        "title": title[:180],
        "peer_id": peer_id,
        "peer_label": _company_label(peer_id),
        "sector": sector or "-",
        "event_type": event_type or "-",
        "salience_score": salience_score,
        "exposure_score": exposure_score,
        "visibility_gap": visibility_gap,
        "label": label,
        "salience_triggers": salience_triggers[:8],
        "recurrence_60d": recurrence_60d,
        "recurrence_label": recurrence_label,
        "source_type": source_type or "-",
        "lane": "primary" if salience_score >= _SALIENCE_PRIMARY_THRESHOLD else "context",
        "card_attached_numbers": card_attached_numbers or [],
        "narrative_hint": _narrative_hint(
            label=label,
            recurrence_label=recurrence_label,
            salience_score=salience_score,
            exposure_score=exposure_score,
            event_type=event_type,
            peer_label=_company_label(peer_id),
        ),
    }


def _visibility_label(
    *,
    salience_score: float,
    exposure_score: float,
    visibility_gap: float,
) -> str:
    if (
        salience_score >= _SALIENCE_PRIMARY_THRESHOLD
        and exposure_score < _EXPOSURE_LOW_THRESHOLD
        and visibility_gap >= _VISIBILITY_GAP_MIN
    ):
        return "low_visibility_definite_event"
    if salience_score >= _SALIENCE_PRIMARY_THRESHOLD and exposure_score >= _EXPOSURE_LOW_THRESHOLD:
        return "high_salience_visible"
    if salience_score >= 0.55:
        return "moderate_salience"
    return "volume_driven"


def _narrative_hint(
    *,
    label: str,
    recurrence_label: str,
    salience_score: float,
    exposure_score: float,
    event_type: str,
    peer_label: str,
) -> str:
    event_label = _event_type_label(event_type)
    if label == "low_visibility_definite_event":
        impact = _impact_adjective(salience_score)
        return (
            f"{peer_label} {event_label}은 사업 영향이 {impact} 편인데, "
            f"{_exposure_clause(exposure_score)}. 단건 보도라 놓치기 쉽습니다."
        )
    if label == "high_salience_visible":
        return (
            f"{peer_label} {event_label}은 사업 영향과 보도 확산이 함께 높아 "
            "오늘의 핵심 판단 축입니다."
        )
    if recurrence_label == "novel":
        return f"{peer_label} {event_label} 조합이 이번 기간에 처음 포착됐습니다."
    if recurrence_label == "recurring":
        return f"{peer_label} {event_label} 조합이 반복되는 패턴입니다."
    return "보도량·업종 비중은 참고 맥락으로만 봅니다."


def _relevant_keyword_groups(
    *,
    current_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> list[str]:
    groups: list[str] = list(_DEFAULT_KEYWORD_GROUPS[:4])
    sectors: list[str] = []
    for issue in current_issues:
        sectors.extend(str(item) for item in _list(issue.get("sectors")) if item)
    for card in cards:
        for key in ("primary_keyword_category", "sector"):
            value = str(card.get(key) or "").strip().lower()
            if value:
                sectors.append(value)
    for sector in dict.fromkeys(sectors):
        groups.extend(_SECTOR_TO_KEYWORD_GROUPS.get(sector, ()))
    deduped: list[str] = []
    seen: set[str] = set()
    for group in groups:
        normalized = group.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped[:10]


def _load_keyword_trend_facts(
    *,
    anchor_date: date,
    groups: list[str],
    history_days: int,
) -> list[dict[str, Any]]:
    if not groups:
        return []
    rows = _fetch_keyword_trend_rows(
        anchor_date=anchor_date,
        groups=groups,
        history_days=history_days,
    )
    return compute_keyword_trend_facts(rows)


def _fetch_keyword_trend_rows(
    *,
    anchor_date: date,
    groups: list[str],
    history_days: int,
) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            result = db.execute(
                text(
                    """
                    WITH trend_rows AS (
                        SELECT
                            COALESCE(ra.metadata ->> 'group_name', ra.company::jsonb ->> 0)
                                AS group_name,
                            COALESCE((ra.metadata ->> 'period')::date, ra.published_at::date)
                                AS period,
                            NULLIF(ra.metadata ->> 'ratio', '')::numeric AS ratio,
                            COALESCE(ra.metadata ->> 'time_unit', 'date') AS time_unit,
                            ROW_NUMBER() OVER (
                                PARTITION BY
                                    COALESCE(
                                        ra.metadata ->> 'group_name',
                                        ra.company::jsonb ->> 0,
                                    ),
                                    COALESCE(
                                        (ra.metadata ->> 'period')::date,
                                        ra.published_at::date,
                                    )
                                ORDER BY ra.collected_at DESC NULLS LAST, ra.id DESC
                            ) AS rn
                        FROM raw_articles ra
                        WHERE ra.source_type = 'search_trend'
                          AND COALESCE((ra.metadata ->> 'period')::date, ra.published_at::date)
                              BETWEEN CAST(:anchor_date AS date)
                                  - (:history_days * INTERVAL '1 day')
                              AND CAST(:anchor_date AS date)
                    )
                    SELECT group_name, period::text AS period, ratio
                      FROM trend_rows
                     WHERE rn = 1
                       AND time_unit = 'date'
                       AND group_name = ANY(:groups)
                       AND ratio IS NOT NULL
                     ORDER BY group_name ASC, period DESC
                    """
                ),
                {
                    "anchor_date": anchor_date.isoformat(),
                    "history_days": int(history_days),
                    "groups": groups,
                },
            ).mappings()
            return [dict(row) for row in result.all()]
    except Exception as exc:  # noqa: BLE001
        log.debug("keyword trend lookup skipped | error=%s", exc)
        return []


def compute_keyword_trend_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn deduped search_trend rows into WoW-style ratio deltas (pure, testable)."""
    by_group: dict[str, list[tuple[date, float]]] = {}
    for row in rows:
        group_name = str(row.get("group_name") or "").strip()
        period_raw = str(row.get("period") or "").strip()
        ratio_raw = row.get("ratio")
        if not group_name or not period_raw or ratio_raw is None:
            continue
        try:
            period = date.fromisoformat(period_raw[:10])
            ratio = round(float(ratio_raw), 2)
        except (TypeError, ValueError):
            continue
        by_group.setdefault(group_name, []).append((period, ratio))

    facts: list[dict[str, Any]] = []
    for group_name, points in by_group.items():
        points.sort(key=lambda item: item[0], reverse=True)
        latest_period, latest_ratio = points[0]
        prev_ratio = points[1][1] if len(points) > 1 else latest_ratio
        ratio_delta = round(latest_ratio - prev_ratio, 2)
        facts.append(
            {
                "metric": "keyword_search_ratio_delta",
                "group_name": group_name,
                "latest_period": latest_period.isoformat(),
                "latest_ratio": latest_ratio,
                "prev_ratio": prev_ratio,
                "ratio_delta": ratio_delta,
                "source": "raw_articles.search_trend",
                "note": "네이버 DataLab 상대 검색지수. 특정 뉴스와 직접 매핑하지 않음.",
            }
        )

    facts.sort(key=lambda item: abs(float(item.get("ratio_delta") or 0.0)), reverse=True)
    return facts


def _build_structural_facts(
    *,
    change_stats: dict[str, Any],
    current_issues: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    window_days: int,
    keyword_trends: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []

    facts.append(
        {
            "metric": "today_detected_count",
            "today": change_stats.get("today_issue_count", 0)
            + change_stats.get("recent_card_count", 0),
            "baseline_label": f"최근 {window_days}일",
            "note": "통합 이슈 + 카드뉴스 합산",
        }
    )

    for row in _list(change_stats.get("peer_delta_vs_window"))[:4]:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        if not key:
            continue
        facts.append(
            {
                "metric": "peer_activity_delta",
                "key": key,
                "peer_label": _company_label(key),
                "today_count": row.get("today_count"),
                "historical_daily_avg": row.get("historical_daily_avg"),
                "delta_vs_daily_avg": row.get("delta_vs_daily_avg"),
                "note": "보도량 맥락 — salience 대체 아님",
            }
        )

    for row in _list(change_stats.get("sector_delta_vs_window"))[:3]:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        if not key:
            continue
        facts.append(
            {
                "metric": "sector_activity_delta",
                "key": key,
                "today_count": row.get("today_count"),
                "historical_daily_avg": row.get("historical_daily_avg"),
                "delta_vs_daily_avg": row.get("delta_vs_daily_avg"),
            }
        )

    event_mix = _event_mix_shift(current_issues, history_issues, window_days)
    if event_mix:
        facts.append(event_mix)

    intensity = _intensity_fact(cards, current_issues)
    if intensity:
        facts.append(intensity)

    for item in (keyword_trends or [])[:4]:
        facts.append(item)

    return facts


def _event_mix_shift(
    current_issues: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any] | None:
    current = Counter(
        str(row.get("event_type") or "unknown") for row in current_issues if row.get("event_type")
    )
    if not current:
        return None

    history = Counter(
        str(row.get("event_type") or "unknown") for row in history_issues if row.get("event_type")
    )
    history_total = sum(history.values()) or 1
    current_total = sum(current.values()) or 1

    shifts: list[dict[str, Any]] = []
    for event_type, count in current.most_common(5):
        current_pct = round(100.0 * count / current_total, 1)
        baseline_pct = round(100.0 * history.get(event_type, 0) / history_total, 1)
        shifts.append(
            {
                "event_type": event_type,
                "today_pct": current_pct,
                "baseline_pct": baseline_pct,
                "delta_pp": round(current_pct - baseline_pct, 1),
            }
        )
    shifts.sort(key=lambda item: abs(float(item.get("delta_pp") or 0.0)), reverse=True)
    top = shifts[0] if shifts else None
    if not top:
        return None
    return {
        "metric": "event_type_mix_shift",
        "window_days": window_days,
        "top_shift": top,
        "distribution_today": dict(current),
    }


def _intensity_fact(
    cards: list[dict[str, Any]],
    current_issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scores: list[float] = []
    for card in cards:
        value = card.get("importance_score")
        if value is not None:
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue
    if not scores and not current_issues:
        return None
    avg_importance = round(sum(scores) / len(scores), 3) if scores else None
    return {
        "metric": "card_importance_intensity",
        "avg_importance_score": avg_importance,
        "high_importance_count": sum(1 for score in scores if score >= 0.7),
        "card_count": len(scores),
        "issue_count": len(current_issues),
    }


def _select_primary_lane(
    salience_candidates: list[dict[str, Any]],
    visibility_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_pool = [
        item
        for item in salience_candidates
        if float(item.get("salience_score") or 0.0) >= _SALIENCE_PRIMARY_THRESHOLD
    ]
    if not primary_pool:
        primary_pool = salience_candidates[:1]

    ordered = list(primary_pool[:3])
    if visibility_gaps:
        top_gap = visibility_gaps[0]
        if top_gap.get("id") and not any(item.get("id") == top_gap.get("id") for item in ordered):
            ordered = [top_gap, *[item for item in ordered if item.get("id") != top_gap.get("id")]][
                :3
            ]

    return {
        "lane": "primary",
        "threshold": _SALIENCE_PRIMARY_THRESHOLD,
        "ids": [str(item.get("id") or "") for item in ordered if item.get("id")],
        "items": [
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "title": item.get("title"),
                "label": item.get("label"),
                "salience_score": item.get("salience_score"),
                "exposure_score": item.get("exposure_score"),
                "narrative_hint": item.get("narrative_hint"),
            }
            for item in ordered
        ],
        "must_include_hidden_gem": bool(visibility_gaps),
    }


def _select_context_lane(
    structural: list[dict[str, Any]],
    salience_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    volume_candidates = [
        item
        for item in salience_candidates
        if item.get("label") in {"volume_driven", "moderate_salience", "high_salience_visible"}
    ]
    volume_candidates.sort(
        key=lambda item: float(item.get("exposure_score") or 0.0),
        reverse=True,
    )
    return {
        "lane": "context",
        "structural_metric_ids": [str(item.get("metric") or "") for item in structural[:5]],
        "volume_signal_ids": [
            str(item.get("id") or "") for item in volume_candidates[:2] if item.get("id")
        ],
        "note": "보도 확산·sector 비중 변화는 맥락 설명용",
    }


def _prior_memory_facts(prior_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for report in prior_reports[:3]:
        if not isinstance(report, dict):
            continue
        out.append(
            {
                "report_date": report.get("report_date"),
                "headline": report.get("headline"),
                "signal_values": _list(report.get("signal_values"))[:3],
            }
        )
    return out


def _coverage_summary(
    *,
    salience_candidates: list[dict[str, Any]],
    structural: list[dict[str, Any]],
    visibility_gaps: list[dict[str, Any]],
    keyword_trends: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    keyword_count = len(keyword_trends or [])
    return {
        "salience_candidates": len(salience_candidates),
        "structural_facts": len(structural),
        "keyword_trends": keyword_count,
        "hidden_gem_count": len(visibility_gaps),
        "card_attached_numbers": sum(
            len(_list(item.get("card_attached_numbers"))) for item in salience_candidates
        ),
        "mode": (
            "dual_lane_full"
            if salience_candidates and structural
            else "salience_only"
            if salience_candidates
            else "structural_only"
            if structural
            else "sparse"
        ),
    }


def _exposure_for_card(card: dict[str, Any]) -> float:
    implication = card.get("implication")
    if isinstance(implication, dict):
        raw = implication.get("exposure_score")
        if raw is not None:
            try:
                return round(min(1.0, max(0.0, float(raw))), 3)
            except (TypeError, ValueError):
                pass
        signals = implication.get("signals")
        if isinstance(signals, dict):
            cluster_size = signals.get("cluster_size")
            if cluster_size is not None:
                try:
                    size = max(1, int(cluster_size))
                    return round(min(1.0, size / _CLUSTER_SIZE_SATURATION) * 0.85, 3)
                except (TypeError, ValueError):
                    pass

    source_count = _source_count(card)
    if source_count <= 1:
        return 0.18
    if source_count == 2:
        return 0.28
    if source_count <= 4:
        return 0.38
    return 0.52


def _exposure_for_issue(issue: dict[str, Any]) -> float:
    source_ids = _list(issue.get("source_ids"))
    sources = _list(issue.get("sources"))
    count = max(len(source_ids), len(sources), 1)
    confidence = issue.get("confidence")
    try:
        conf = float(confidence) if confidence is not None else 0.5
    except (TypeError, ValueError):
        conf = 0.5

    base = 0.15 + min(count, 6) * 0.05
    return round(min(1.0, base * (0.75 + 0.25 * conf)), 3)


def _source_count(card: dict[str, Any]) -> int:
    raw_ids = _list(card.get("source_raw_article_ids"))
    if raw_ids:
        return len(raw_ids)
    sources = _list(card.get("sources"))
    if sources:
        return len(sources)
    return 1


def _infer_source_type_from_card(card: dict[str, Any]) -> str:
    for source in _list(card.get("sources")):
        if not isinstance(source, dict):
            continue
        for key in ("source_type", "type", "source_name", "publisher"):
            value = str(source.get(key) or "").strip().lower()
            if value:
                normalized = _normalize_source_type(value)
                if normalized:
                    return normalized
    return _normalize_source_type(str(_sector_from_implication(card) or ""))


def _normalize_source_type(raw: str) -> str:
    text = raw.strip().lower()
    if not text:
        return ""
    if "dart" in text or "공시" in text:
        return "dart"
    if text in {"ir", "ir_pdf"} or "ir" in text:
        return "ir"
    if any(marker in text for marker in ("official", "company_site", "보도자료", "newsroom")):
        return "official"
    if "securities" in text:
        return "securities_report"
    return text


def _sector_from_implication(card: dict[str, Any]) -> str:
    implication = card.get("implication")
    if isinstance(implication, dict):
        sector = implication.get("sector")
        if sector:
            return str(sector)
    return ""


def _card_attached_numbers(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Numbers already stored on the card — cite only, never aggregate."""
    out: list[dict[str, Any]] = []
    implication = card.get("implication")
    if isinstance(implication, dict):
        chain = implication.get("evidence_chain")
        if isinstance(chain, dict):
            for ref in _list(chain.get("financial_refs"))[:3]:
                if not isinstance(ref, dict):
                    continue
                narrative = str(ref.get("narrative") or ref.get("value") or "").strip()
                if narrative:
                    out.append(
                        {
                            "card_id": str(card.get("id") or ""),
                            "claim": narrative[:160],
                            "source": "evidence_chain.financial_refs",
                        }
                    )
    evidence_payload = card.get("evidence_payload")
    if isinstance(evidence_payload, dict):
        package = evidence_payload.get("analysis_package")
        if isinstance(package, dict):
            for key in ("metrics", "numeric_claims"):
                for item in _list(package.get(key))[:2]:
                    if isinstance(item, dict):
                        text = str(item.get("text") or item.get("value") or "").strip()
                    else:
                        text = str(item or "").strip()
                    if text:
                        out.append(
                            {
                                "card_id": str(card.get("id") or ""),
                                "claim": text[:160],
                                "source": f"evidence_payload.{key}",
                            }
                        )
    return out[:4]


def _recurrence_count(
    *,
    peer_id: str,
    event_type: str,
    history_issues: list[dict[str, Any]],
    exclude_id: str,
) -> int:
    if not peer_id or not event_type:
        return 0
    count = 0
    for issue in history_issues:
        if str(issue.get("id") or "") == exclude_id:
            continue
        if str(issue.get("main_company") or "") != peer_id:
            continue
        if str(issue.get("event_type") or "") != event_type:
            continue
        count += 1
    return count


def _company_label(company_id: str) -> str:
    return COMPANIES.get(company_id, {}).get("name_ko") or company_id or "Peer"


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def format_evidence_change_lines(comparison_facts: dict[str, Any] | None) -> list[str]:
    """Human-readable change lines for signal evidence (DB-backed only)."""
    if not isinstance(comparison_facts, dict):
        return []

    lines: list[str] = []
    for item in _list(comparison_facts.get("keyword_trends"))[:2]:
        if not isinstance(item, dict):
            continue
        group_name = str(item.get("group_name") or "")
        ratio_delta = item.get("ratio_delta")
        latest_ratio = item.get("latest_ratio")
        if not group_name or ratio_delta is None:
            continue
        try:
            delta = float(ratio_delta)
        except (TypeError, ValueError):
            continue
        sign = "+" if delta > 0 else ""
        latest = f", 지수 {latest_ratio}" if latest_ratio is not None else ""
        lines.append(f"{group_name} 검색지수 {sign}{delta:.1f}pt (전일 대비{latest})")

    for item in _list(comparison_facts.get("structural")):
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "")
        if metric == "peer_activity_delta":
            peer_label = str(item.get("peer_label") or item.get("key") or "")
            if peer_label.strip().lower() in {"meta", "other", "unknown", "-", "google"}:
                continue
            peer_delta = item.get("delta_vs_daily_avg")
            today_count = item.get("today_count")
            if peer_label and peer_delta is not None:
                peer_delta_value = float(peer_delta)
                try:
                    today_count_value = float(today_count or 0)
                except (TypeError, ValueError):
                    today_count_value = 0.0
                if today_count_value <= 0 or abs(peer_delta_value) < 0.05:
                    continue
                sign = "+" if peer_delta_value > 0 else ""
                lines.append(
                    f"{peer_label} 카드/이슈 {today_count}건 "
                    f"(일평균 대비 {sign}{peer_delta_value:.1f}건)"
                )
        elif metric == "event_type_mix_shift":
            top = item.get("top_shift")
            if isinstance(top, dict) and top.get("event_type"):
                event_label = _event_type_label(str(top.get("event_type") or ""))
                lines.append(
                    f"{event_label} 비중 {top.get('today_pct')}% "
                    f"(최근 평균 {top.get('baseline_pct')}%, "
                    f"{top.get('delta_pp'):+}pp)"
                )

    for gap in _list(comparison_facts.get("visibility_gaps"))[:1]:
        if not isinstance(gap, dict):
            continue
        title = str(gap.get("title") or "")
        peer_label = str(gap.get("peer_label") or gap.get("peer_id") or "")
        subject = _short_title(title) if title else peer_label
        if subject:
            lines.append(f"{subject}: 영향은 크지만 보도는 아직 적음")

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line[:160])
    return deduped[:5]


def build_ui_change_summary(
    *,
    default_rows: list[dict[str, str]],
    comparison_facts: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Merge deterministic comparison chips for the home meta row."""
    rows = [dict(item) for item in default_rows[:3]]
    if not isinstance(comparison_facts, dict):
        return rows[:3]

    coverage = comparison_facts.get("coverage")
    hidden_count = 0
    if isinstance(coverage, dict):
        try:
            hidden_count = int(coverage.get("hidden_gem_count") or 0)
        except (TypeError, ValueError):
            hidden_count = 0

    keyword_rows = [
        item for item in _list(comparison_facts.get("keyword_trends")) if isinstance(item, dict)
    ]
    if keyword_rows:
        top = keyword_rows[0]
        group_name = str(top.get("group_name") or "")
        ratio_delta = top.get("ratio_delta")
        if group_name and ratio_delta is not None:
            try:
                delta = float(ratio_delta)
                sign = "+" if delta > 0 else ""
                rows[1] = {
                    "label": "검색지수 변화",
                    "value": f"{group_name} {sign}{delta:.1f}pt",
                }
            except (TypeError, ValueError):
                pass
    elif hidden_count > 0 and len(rows) > 1:
        rows[1] = {"label": "단건·고임팩트", "value": f"{hidden_count}건"}

    primary = comparison_facts.get("primary_selection")
    if isinstance(primary, dict):
        lead = next(
            (item for item in _list(primary.get("items")) if isinstance(item, dict)),
            None,
        )
        if lead and len(rows) > 2:
            event_type = str(lead.get("label") or lead.get("event_type") or "")
            if event_type == "low_visibility_definite_event":
                rows[2] = {"label": "핵심 축", "value": "단건 전략 이벤트"}
            elif lead.get("title"):
                rows[2] = {
                    "label": "핵심 축",
                    "value": _clip(str(lead.get("title") or ""), 36),
                }

    return rows[:3]


def _clip(value: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    return cleaned[:limit].rstrip()


def _short_title(title: str, *, limit: int = 42) -> str:
    cleaned = " ".join(str(title or "").split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


_GENERIC_EXECUTIVE_MARKERS = (
    "경쟁 환경",
    "경쟁력 강화",
    "경쟁 심화",
    "시장 확대",
    "전략 강화",
    "중요한 변화를 예고",
    "직접적인 영향",
    "영향을 미칠",
    "대응이 필요",
    "주목할 필요",
)

_GENERIC_SIGNAL_MARKERS = (
    "보도량·sector 비중 맥락 확인",
    "고객 대응·PoC·운영모델에서 무엇을 바꿀지 오늘 결정",
    "Peer 신호가",
    "AX 신호가 단건",
)


def is_generic_executive_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    return any(marker in cleaned or marker in lowered for marker in _GENERIC_EXECUTIVE_MARKERS)


def _primary_lead(comparison_facts: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(comparison_facts, dict):
        return None
    primary = comparison_facts.get("primary_selection")
    if not isinstance(primary, dict):
        return None
    return next(
        (item for item in _list(primary.get("items")) if isinstance(item, dict)),
        None,
    )


def build_primary_headline(comparison_facts: dict[str, Any] | None) -> str:
    lead = _primary_lead(comparison_facts)
    if not lead:
        return ""
    title = str(lead.get("title") or "").strip()
    if not title:
        return ""
    return _clip(title, 120)


def _execution_terms_for_title(title: str) -> tuple[str, str]:
    lowered = str(title or "").lower()
    if "gpu" in lowered:
        return (
            "GPU 확보, 클라우드 운영, 보안 운영을 묶은 실행 역량",
            "GPU 기반 서비스 개시 일정, 운영 SLA, 보안 모니터링 체계",
        )
    if any(token in lowered for token in ("클라우드", "cloud", "msp")):
        return (
            "클라우드 운영 안정성, 비용 구조, 보안 책임 범위",
            "서비스 전환 일정, 운영 SLA, 보안·비용 검증 지표",
        )
    if any(token in lowered for token in ("보안", "security", "제로트러스트")):
        return (
            "보안 운영 역량, 탐지·대응 체계, 감사 대응 기준",
            "모니터링 범위, 대응 SLA, 고객별 보안 운영 기준",
        )
    if any(
        token in lowered
        for token in (
            "ai",
            "llm",
            "생성형",
            "에이전트",
            "agent",
            "anthropic",
            "openai",
            "클로드",
        )
    ):
        return (
            "AI 기능보다 운영 책임, 데이터 보안, 검증 지표",
            "PoC 검증 지표, 운영 SLA, 보안·감사 대응 기준",
        )
    return (
        "고객 대응 레퍼런스, 운영 KPI, 검증 지표",
        "고객별 대응 기준, 운영 SLA, PoC 검증 항목",
    )


def build_executive_summary_from_facts(
    comparison_facts: dict[str, Any] | None,
    *,
    change_stats: dict[str, Any] | None = None,
) -> str:
    lead = _primary_lead(comparison_facts)
    if not lead:
        return ""
    title = str(lead.get("title") or "")
    short_title = _short_title(title, limit=56)
    label = str(lead.get("label") or "")
    del change_stats
    customer_terms, action_terms = _execution_terms_for_title(title)
    if label == "low_visibility_definite_event":
        return _clip(
            (
                f"'{short_title}'은 고객 비교 기준을 실행 역량 중심으로 "
                f"옮길 수 있습니다: {customer_terms}. "
                f"SK AX는 {action_terms}를 오늘 고객별 대응 기준에 반영해야 합니다."
            ),
            320,
        )
    elif label == "high_salience_visible":
        return _clip(
            f"'{short_title}'은 보도 확산과 사업 영향이 함께 큰 신호입니다. "
            f"SK AX는 {action_terms}가 고객 대응 패키지에 반영돼 있는지 확인해야 합니다.",
            320,
        )
    else:
        return _clip(
            f"'{short_title}'은 오늘 대응 우선순위를 다시 좁힐 신호입니다. "
            f"SK AX는 다음 실행 기준으로 고객별 확인 항목을 정해야 합니다: {customer_terms}.",
            320,
        )


def build_executive_implication_from_facts(comparison_facts: dict[str, Any] | None) -> str:
    lead = _primary_lead(comparison_facts)
    if not lead:
        return ""
    title = _short_title(str(lead.get("title") or ""), limit=48)
    label = str(lead.get("label") or "")
    _, action_terms = _execution_terms_for_title(title)
    if label == "low_visibility_definite_event":
        return _clip(
            f"SK AX는 '{title}'를 단순 뉴스가 아니라 고객 비교 기준 변화로 봐야 합니다. "
            f"{action_terms}를 포함한 고객별 대응 기준을 오늘 정해야 합니다.",
            280,
        )
    if label == "high_salience_visible":
        return _clip(
            f"SK AX는 '{title}'를 범용 AX 메시지가 아니라 실행 산출물 기준으로 확인해야 합니다. "
            f"{action_terms}를 고객 대응 패키지에 반영할지 오늘 결정해야 합니다.",
            280,
        )
    return _clip(
        f"SK AX는 '{title}'를 고객 대응 기준 변화로 볼지 먼저 판단해야 합니다. "
        f"{action_terms} 중심으로 대응 우선순위를 정해야 합니다.",
        280,
    )


def build_primary_signal_value(comparison_facts: dict[str, Any] | None) -> str:
    lead = _primary_lead(comparison_facts)
    if not lead:
        return ""
    title = _short_title(str(lead.get("title") or ""), limit=36)
    label = str(lead.get("label") or "")
    if label == "low_visibility_definite_event":
        return _clip(f"고객 비교 기준 변화 — {title}", 96)
    if label == "high_salience_visible":
        return _clip(f"핵심 판단 축 — {title}", 96)
    return _clip(title, 96)


def build_context_signal_value(comparison_facts: dict[str, Any] | None) -> str:
    if not isinstance(comparison_facts, dict):
        return ""
    changes = format_evidence_change_lines(comparison_facts)
    for line in changes:
        if "영향은 크지만" in line:
            return _clip(line, 96)
    for line in changes:
        if "검색지수" in line or "일평균" in line or "비중" in line:
            return _clip(line, 96)
    trends = _list(comparison_facts.get("keyword_trends"))
    if trends and isinstance(trends[0], dict):
        group_name = str(trends[0].get("group_name") or "")
        ratio_delta = trends[0].get("ratio_delta")
        if group_name and ratio_delta is not None:
            sign = "+" if float(ratio_delta) > 0 else ""
            return _clip(
                f"{group_name} 검색지수 {sign}{float(ratio_delta):.1f}pt — 시장 관심 참고", 96
            )
    return "보도량·업종 비중은 핵심 신호를 대체하지 않음"


def build_next_judgment_signal_value(
    comparison_facts: dict[str, Any] | None,
    *,
    lead: dict[str, Any] | None = None,
) -> str:
    item = lead or _primary_lead(comparison_facts)
    if not item:
        return "고객 대응·PoC·운영 KPI 중 무엇을 바꿀지 오늘 확정"
    title = _short_title(str(item.get("title") or ""), limit=40)
    if str(item.get("label") or "") == "low_visibility_definite_event":
        return _clip(f"'{title}' 후속 대응 반영 여부를 이번 주 안에 확정", 96)
    return _clip(f"'{title}' 기준 고객 대응 산출물·검증 지표 변경 여부 결정", 96)


def polish_executive_output(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic post-process: generic LLM prose → comparison_facts 기반 문장."""
    out = dict(payload)
    comparison = context.get("comparison_facts")
    if not isinstance(comparison, dict):
        return out

    raw_change_stats = context.get("change_stats")
    change_stats: dict[str, Any] = raw_change_stats if isinstance(raw_change_stats, dict) else {}
    default_summary = change_stats.get("default_change_summary")
    default_rows: list[dict[str, str]] = (
        default_summary
        if isinstance(default_summary, list)
        else [dict(row) for row in _list(out.get("change_summary")) if isinstance(row, dict)]
    )
    out["change_summary"] = build_ui_change_summary(
        default_rows=default_rows,
        comparison_facts=comparison,
    )

    primary_headline = build_primary_headline(comparison)
    if primary_headline:
        out["headline"] = primary_headline

    summary_rewrite = build_executive_summary_from_facts(comparison, change_stats=change_stats)
    if summary_rewrite:
        out["executive_summary"] = summary_rewrite

    implication = str(out.get("executive_implication") or "")
    lead = _primary_lead(comparison)
    implication_rewrite = build_executive_implication_from_facts(comparison)
    if (
        lead
        and implication_rewrite
        and (
            is_generic_executive_text(implication)
            or "salience" in implication.lower()
            or "노출 0." in implication
            or implication.strip() == str(lead.get("narrative_hint") or "").strip()
        )
    ):
        out["executive_implication"] = implication_rewrite

    signals = [item for item in _list(out.get("signals")) if isinstance(item, dict)]
    if len(signals) >= 3:
        polished: list[dict[str, Any]] = []
        replacements = [
            (build_primary_signal_value(comparison) or str(signals[0].get("value") or "")),
            build_context_signal_value(comparison) or str(signals[1].get("value") or ""),
            build_next_judgment_signal_value(comparison, lead=lead),
        ]
        for idx, signal in enumerate(signals):
            item = dict(signal)
            if idx >= len(replacements):
                polished.append(item)
                continue
            current = str(item.get("value") or "")
            replacement = str(replacements[idx] or "")
            if replacement and (
                not current.strip()
                or any(marker in current for marker in _GENERIC_SIGNAL_MARKERS)
                or (idx == 0 and is_generic_executive_text(current))
            ):
                item["value"] = _clip(replacement, 96)
            polished.append(item)
        out["signals"] = polished

    provenance = out.get("provenance")
    if isinstance(provenance, dict):
        out["provenance"] = {
            **provenance,
            "postprocess": "comparison_facts_v1",
        }
    return out


def trim_comparison_facts_for_prompt(comparison_facts: dict[str, Any] | None) -> dict[str, Any]:
    """Keep salience/structural facts intact while dropping verbose candidate payloads."""
    if not isinstance(comparison_facts, dict):
        return {}

    slim_candidates: list[dict[str, Any]] = []
    for item in _list(comparison_facts.get("salience_candidates"))[:8]:
        if not isinstance(item, dict):
            continue
        numbers = [
            row for row in _list(item.get("card_attached_numbers")) if isinstance(row, dict)
        ][:2]
        slim_candidates.append(
            {
                "kind": item.get("kind"),
                "id": item.get("id"),
                "title": item.get("title"),
                "peer_label": item.get("peer_label"),
                "sector": item.get("sector"),
                "event_type": item.get("event_type"),
                "label": item.get("label"),
                "recurrence_label": item.get("recurrence_label"),
                "narrative_hint": item.get("narrative_hint"),
                "salience_score": item.get("salience_score"),
                "exposure_score": item.get("exposure_score"),
                "card_attached_numbers": numbers,
            }
        )

    return {
        "anchor_date": comparison_facts.get("anchor_date"),
        "window_days": comparison_facts.get("window_days"),
        "coverage": comparison_facts.get("coverage"),
        "primary_selection": comparison_facts.get("primary_selection"),
        "context_selection": comparison_facts.get("context_selection"),
        "visibility_gaps": _list(comparison_facts.get("visibility_gaps"))[:3],
        "keyword_trends": _list(comparison_facts.get("keyword_trends"))[:6],
        "structural": _list(comparison_facts.get("structural"))[:8],
        "salience_candidates": slim_candidates,
    }


__all__ = [
    "build_comparison_facts",
    "build_ui_change_summary",
    "build_context_signal_value",
    "build_executive_implication_from_facts",
    "build_executive_summary_from_facts",
    "build_primary_headline",
    "build_primary_signal_value",
    "compute_keyword_trend_facts",
    "format_evidence_change_lines",
    "is_generic_executive_text",
    "polish_executive_output",
    "trim_comparison_facts_for_prompt",
]
