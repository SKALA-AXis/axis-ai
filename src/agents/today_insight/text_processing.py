"""today_insight text_processing — extracted from facade (move-only)."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime
from typing import Any, Mapping

from src.config.companies import COMPANIES
from src.config.company_tiers import DOMESTIC_COMPANY_IDS
from src.services.peer_id_aliases import PEER_ID_ALIASES, normalize_to_canonical_id

_WEAK_OR_MOCK_MARKERS = (
    "금융 AX",
    "제안",
    "제안서",
    "제안 자료",
    "고객 제안",
    "AX 제안",
    "제안 우선순위",
    "전략 강화",
    "전략 수립",
    "전략 분석",
    "대응 전략",
    "시장 확대",
    "경쟁 환경",
    "경쟁력 강화",
    "벤치마킹",
    "가능성 검토",
    "디지털 전환",
    "중요한 사례",
)


_PUBLIC_TEXT_REPLACEMENTS = (
    ("low_visibility_definite_event", "노출은 낮지만 내용이 확인된 이벤트"),
    ("high_salience_visible", "보도 확산이 큰 이벤트"),
    ("general_update", "일반 업데이트"),
    ("event_type_mix_shift", "이벤트 유형 변화"),
    ("peer_activity_delta", "Peer 활동 변화"),
    ("baseline_pct", "최근 평균 비중"),
    ("today_pct", "오늘 비중"),
    ("delta_pp", "변화폭"),
    ("ratio_delta", "검색 관심도 변화"),
    ("latest_ratio", "최근 검색 관심도"),
    ("baseline", "최근 평균"),
    ("today_insight_reports", "저장 리포트"),
    ("integrated_issues", "통합 이슈"),
    ("card_news", "카드뉴스"),
    ("raw_articles", "원문"),
    ("analysis_ledger", "분석 기록"),
    ("source id", "근거"),
    ("raw id", "원문 근거"),
    ("source_ids", "근거"),
    ("source_id", "근거"),
    ("source_raw_article_ids", "원문 근거"),
    ("source_integrated_issue_id", "통합 이슈 근거"),
    ("source_card_id", "카드뉴스 근거"),
    ("primary_selection", "대표 신호"),
    ("comparison_facts", "비교 근거"),
)


def _section_actions(
    value: Any,
    *,
    actions: list[dict[str, Any]],
    source_ids: set[str],
    idx: int,
) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    if rows:
        return rows[:2]
    matched = []
    for action in actions:
        refs = {
            str(ref)
            for ref in _list(action.get("evidence_refs") or action.get("evidenceRefs"))
            if str(ref or "").strip()
        }
        if not source_ids or refs.intersection(source_ids):
            matched.append(action)
    if matched:
        return matched[:2]
    if idx < len(actions):
        return [actions[idx]]
    return actions[:1]


def _is_before_window(raw_date: str, window_start: date) -> bool:
    if not raw_date:
        return False
    try:
        return date.fromisoformat(raw_date[:10]) < window_start
    except ValueError:
        return False


def _fallback_headline(context: dict[str, Any]) -> str:
    comparison = context.get("comparison_facts")
    if isinstance(comparison, dict):
        primary = comparison.get("primary_selection")
        if isinstance(primary, dict):
            items = [item for item in _list(primary.get("items")) if isinstance(item, dict)]
            if items:
                lead = items[0]
                title = str(lead.get("title") or "").strip()
                if title:
                    return _clip(title, 120)

    first = _first_issue(context)
    if first:
        company = _company_label(str(first.get("main_company") or ""))
        headline = str(first.get("headline") or first.get("one_line_summary") or "")
        return _clip(
            f"{company} 신호를 기준으로 오늘의 AX 대응 판단을 재정렬해야 합니다: {headline}", 120
        )
    first_card = _first_card(context)
    if first_card:
        company = _company_label(str(first_card.get("peer_id") or ""))
        title = str(first_card.get("title") or "")
        return _clip(
            f"{company} 카드뉴스 신호를 기준으로 오늘의 AX 대응 판단을 재점검해야 합니다: {title}",
            120,
        )
    return "오늘 비교 가능한 신규 신호가 제한적이어서 기존 AX 관찰 기준을 유지합니다."


def _fallback_summary(context: dict[str, Any]) -> str:
    stats = context.get("change_stats") or {}
    count = stats.get("today_issue_count", 0)
    card_count = stats.get("recent_card_count", 0)
    axis = _top_axis_from_stats(stats)
    if count:
        return (
            f"오늘 확인된 통합 이슈에 {axis or 'AX 실행'} 관련 변화가 포함됐습니다. "
            "과거 누적 결과와 실제 사건 내용을 함께 비교해 고객 대응·PoC·운영 "
            "책임 범위를 확인할 필요가 있습니다."
        )
    if card_count:
        return (
            f"최근 카드뉴스에 {axis or 'AX 실행'} 관련 신호가 포함됐습니다. "
            "통합 이슈가 비어 있어도 원문 내용과 고객 대응 가능 항목을 함께 확인합니다."
        )
    return (
        "오늘 기준 신규 통합 이슈가 충분하지 않습니다. 홈 인사이트는 "
        "최근 누적 분석과 기존 SK AX 관점 유지로 표시합니다."
    )


def _fallback_implication(context: dict[str, Any]) -> str:
    stats = context.get("change_stats") or {}
    axis = _top_axis_from_stats(stats) or "산업 AX"
    return (
        f"SK AX는 {axis} 신호를 범용 AI 메시지로 처리하지 말고, "
        "고객 대응 패키지의 운영 KPI·보안 책임·검증 지표 중 어느 항목을 "
        "바꿀지까지 확인해야 합니다."
    )


def _fallback_signals(context: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = context.get("comparison_facts")
    if isinstance(comparison, dict):
        primary = comparison.get("primary_selection")
        if isinstance(primary, dict):
            lead = next(
                (item for item in _list(primary.get("items")) if isinstance(item, dict)),
                None,
            )
            if lead:
                lead_id = str(lead.get("id") or "")
                source_ids = [lead_id] if lead_id else _default_source_ids(context)
                label = str(lead.get("label") or "")
                title = str(lead.get("title") or "")
                hint = str(lead.get("narrative_hint") or "")
                structural = [
                    str(item.get("metric") or "")
                    for item in _list(comparison.get("structural"))
                    if isinstance(item, dict) and item.get("metric")
                ][:2]
                return [
                    {
                        "id": "signal-primary-salience",
                        "label": "관찰 포인트",
                        "value": _clip(
                            _sanitize_public_text(title) or "오늘 확인된 대표 신호",
                            220,
                        ),
                        "reasoning": [
                            {
                                "stage": "관찰",
                                "detail": _sanitize_public_text(hint or title) or "대표 신호 기준",
                            },
                            {
                                "stage": "비교",
                                "detail": (
                                    ", ".join(structural) or "대표 신호와 보조 맥락을 함께 확인"
                                ),
                            },
                            {
                                "stage": "의미",
                                "detail": (
                                    "입력 근거상 확인된 개별 이벤트로 분류"
                                    if label == "low_visibility_definite_event"
                                    else "오늘 확인할 판단 축"
                                ),
                            },
                            {"stage": "판단", "detail": "고객 대응·PoC·운영 책임 범위 재점검"},
                        ],
                        "evidence": {
                            "grounds": [hint] if hint else _default_grounds(context)[:2],
                            "changes": structural or _default_changes(context),
                            "related_keywords": _keywords_from_context(context),
                            "source_ids": source_ids,
                        },
                    },
                    {
                        "id": "signal-implication",
                        "label": "시사점",
                        "value": "고객 대응·PoC·운영모델에서 확인할 항목을 구체화",
                        "reasoning": [
                            {
                                "stage": "의미",
                                "detail": "primary 이벤트 기준으로 고객 대응 항목을 좁혀 봅니다.",
                            },
                        ],
                        "evidence": {
                            "grounds": _default_grounds(context)[:2],
                            "changes": _default_changes(context)[:2],
                            "related_keywords": _keywords_from_context(context),
                            "source_ids": source_ids,
                        },
                    },
                    {
                        "id": "signal-response-direction",
                        "label": "대응방향",
                        "value": "관련 고객·사업·운영 KPI를 확인해 대응 범위를 정리",
                        "reasoning": [
                            {
                                "stage": "판단",
                                "detail": "입력 근거에 기반해 확인할 대응 범위를 정리합니다.",
                            },
                        ],
                        "evidence": {
                            "grounds": _default_grounds(context)[:2],
                            "changes": _default_changes(context)[:2],
                            "related_keywords": _keywords_from_context(context),
                            "source_ids": source_ids,
                        },
                    },
                ]

    first = _first_issue(context)
    first_card = _first_card(context)
    source_ids = _default_source_ids(context)
    headline = (
        str(first.get("headline") or first.get("one_line_summary") or "")
        if first
        else str(first_card.get("title") or "")
        if first_card
        else ""
    )
    company = (
        _company_label(str(first.get("main_company") or ""))
        if first
        else _company_label(str(first_card.get("peer_id") or ""))
        if first_card
        else "Peer"
    )
    stats = context.get("change_stats") or {}
    top_axis = _top_axis_from_stats(stats) or "AX"
    signal_axis = "AX 대응" if top_axis == company else top_axis
    grounds = _default_grounds(context)
    changes = _default_changes(context)
    return [
        {
            "id": "signal-primary-change",
            "label": "관찰 포인트",
            "value": _clip(f"{company} 신호가 {signal_axis} 판단 축과 연결됨", 220),
            "reasoning": [
                {
                    "stage": "관찰",
                    "detail": headline or "오늘 통합 이슈와 카드뉴스를 함께 조회했습니다.",
                },
                {
                    "stage": "비교",
                    "detail": (
                        f"최근 {stats.get('window_days', 14)}일 카드/이슈와 같은 축을 비교했습니다."
                    ),
                },
                {
                    "stage": "의미",
                    "detail": "단순 보도량보다 고객 대응 산출물에 반영할 판단 기준을 우선했습니다.",
                },
                {
                    "stage": "판단",
                    "detail": "오늘 회의에서는 운영 KPI와 검증 책임 범위를 먼저 확인해야 합니다.",
                },
            ],
            "evidence": {
                "grounds": grounds,
                "changes": changes,
                "related_keywords": _keywords_from_context(context),
                "source_ids": source_ids,
            },
        },
        {
            "id": "signal-implication",
            "label": "시사점",
            "value": "고객 대응·PoC·운영모델에서 확인할 항목을 구체화",
            "reasoning": [
                {"stage": "관찰", "detail": "SK AX 공식 관점과 피어 프로필을 함께 검토했습니다."},
                {
                    "stage": "비교",
                    "detail": (
                        "범용 AX 메시지가 아니라 고객 평가 항목 변화 여부를 기준으로 삼았습니다."
                    ),
                },
                {"stage": "의미", "detail": "고객 대응 산출물의 구조 변경 여부를 확인합니다."},
            ],
            "evidence": {
                "grounds": grounds[:2],
                "changes": changes[:2],
                "related_keywords": _keywords_from_context(context),
                "source_ids": source_ids,
            },
        },
        {
            "id": "signal-response-direction",
            "label": "대응방향",
            "value": "관련 고객·사업·운영 KPI를 확인해 대응 범위를 정리",
            "reasoning": [
                {"stage": "관찰", "detail": "입력 근거의 고객·사업·운영 항목을 함께 봅니다."},
                {
                    "stage": "판단",
                    "detail": (
                        "뉴스 건수만이 아니라 실제 사건 내용과 대응 가능 항목을 "
                        "기준으로 점검합니다."
                    ),
                },
            ],
            "evidence": {
                "grounds": grounds[:2],
                "changes": changes[:2],
                "related_keywords": _keywords_from_context(context),
                "source_ids": source_ids,
            },
        },
    ]


def _fallback_actions(context: dict[str, Any]) -> list[dict[str, Any]]:
    source_ids = _default_source_ids(context)
    axis = _proposal_axis_from_stats(context.get("change_stats") or {})
    return [
        {
            "action": (
                f"{axis} 고객 대응 체크리스트에 운영 KPI, 보안 책임 범위, "
                "PoC 검증 지표를 별도 항목으로 분리합니다."
            ),
            "decision_owner": "사업전략/영업 리더",
            "time_horizon": "이번 주",
            "rationale": "오늘 신호가 제품 기능보다 고객 평가 기준 변화에 더 가깝기 때문입니다.",
            "evidence_refs": source_ids,
        },
        {
            "action": (
                "카드뉴스 원문별 고객명·사업명·검증단계를 표로 재정리해 "
                "기존 SK AX 레퍼런스와 맞닿는 항목만 임원 브리핑에 올립니다."
            ),
            "decision_owner": "전략기획",
            "time_horizon": "오늘",
            "rationale": (
                "근거가 약한 시장 확대 단정을 제거하고, 실제 고객 대응에 쓸 수 "
                "있는 근거만 남기기 위함입니다."
            ),
            "evidence_refs": source_ids,
        },
        {
            "action": (
                "후속 모니터링은 같은 peer·sector에서 수주, PoC, 운영 책임 "
                "확대가 반복되는지로 제한합니다."
            ),
            "decision_owner": "인텔리전스 담당",
            "time_horizon": "이번 달",
            "rationale": "단건 보도와 구조적 변화의 의사결정 가치를 구분해야 하기 때문입니다.",
            "evidence_refs": source_ids,
        },
    ]


def _compact_sources(value: Any, *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(_list(value)):
        if not isinstance(item, dict):
            continue
        source_id = (
            item.get("id")
            or item.get("source_id")
            or item.get("raw_article_id")
            or item.get("url")
            or f"source-{idx + 1}"
        )
        out.append(
            {
                "id": str(source_id),
                "title": _clip(str(item.get("title") or item.get("headline") or ""), 180),
                "source_name": _clip(
                    str(
                        item.get("source_name") or item.get("source") or item.get("publisher") or ""
                    ),
                    80,
                ),
                "publisher": _clip(str(item.get("publisher") or ""), 80),
                "url": _clip(str(item.get("url") or ""), 500),
                "published_at": str(item.get("published_at") or item.get("created_at") or "")
                or None,
            }
        )
        if len(out) >= limit:
            break
    return out


def _related_company_labels_from_source(source: dict[str, Any]) -> list[str]:
    text_blob = " ".join(
        [
            str(source.get("title") or ""),
            str(source.get("content") or ""),
        ]
    )
    related_ids: list[str] = []

    normalized_text = text_blob.lower().replace(" ", "")
    for company_id, aliases in PEER_ID_ALIASES.items():
        if company_id in related_ids:
            continue
        if any(alias.lower().replace(" ", "") in normalized_text for alias in aliases):
            related_ids.append(company_id)

    for raw_company in _list(source.get("company")):
        canonical_id = normalize_to_canonical_id(str(raw_company)) or str(raw_company)
        if canonical_id and canonical_id not in related_ids:
            related_ids.append(canonical_id)

    return [_company_label(company_id) for company_id in related_ids[:6]]


def _is_domestic_company_id(value: object) -> bool:
    """도메스틱 4사(samsung_sds·lg_cns·hyundai_autoever·posco_dx)면 True.

    글로벌 IT peer(google 등 overseas)·self(sk_ax)·미상은 모두 False.
    Today's Insight 는 국내 peer 만 헤드라인/신호에 노출한다.
    """
    if not value:
        return False
    canonical = normalize_to_canonical_id(str(value)) or str(value)
    return canonical in DOMESTIC_COMPANY_IDS


def _is_domestic_issue(issue: dict[str, Any]) -> bool:
    return _is_domestic_company_id(issue.get("main_company"))


def _is_domestic_card(card: dict[str, Any]) -> bool:
    return _is_domestic_company_id(card.get("peer_id"))


def _compact_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    refs = value.get("evidence_refs") or value.get("references") or value.get("items") or []
    compact_refs = []
    for item in _list(refs)[:5]:
        if isinstance(item, dict):
            compact_refs.append(
                {
                    "id": item.get("evidence_ref_id") or item.get("id"),
                    "text": _clip(str(item.get("evidence_text") or item.get("text") or ""), 220),
                    "source_ids": _list(item.get("source_ids"))[:5],
                }
            )
    return {"evidence_refs": compact_refs}


def _first_issue(context: dict[str, Any]) -> dict[str, Any] | None:
    issues = context.get("current_issues") or []
    return issues[0] if issues and isinstance(issues[0], dict) else None


def _first_card(context: dict[str, Any]) -> dict[str, Any] | None:
    cards = context.get("recent_cards") or []
    return cards[0] if cards and isinstance(cards[0], dict) else None


def _default_source_ids(context: dict[str, Any]) -> list[str]:
    ids = [
        str(row.get("id"))
        for row in context.get("current_issues", [])
        if isinstance(row, dict) and row.get("id")
    ]
    ids.extend(
        str(card.get("id"))
        for card in context.get("recent_cards", [])
        if isinstance(card, dict) and card.get("id")
    )
    return _dedupe(ids, limit=6)


def _default_grounds(context: dict[str, Any]) -> list[str]:
    grounds = []
    for issue in context.get("current_issues", [])[:3]:
        if not isinstance(issue, dict):
            continue
        label = _company_label(str(issue.get("main_company") or ""))
        summary = issue.get("one_line_summary") or issue.get("headline") or ""
        if summary:
            grounds.append(_clip(f"{label}: {summary}", 140))
    for card in context.get("recent_cards", [])[:3]:
        if not isinstance(card, dict):
            continue
        label = _company_label(str(card.get("peer_id") or ""))
        summary_lines = " / ".join(str(item) for item in _list(card.get("summary_lines"))[:3])
        summary = summary_lines or card.get("title") or ""
        if summary:
            grounds.append(_clip(f"{label}: {summary}", 140))
    return grounds or ["오늘 기준 통합 이슈와 카드뉴스 근거가 제한적으로 조회되었습니다."]


def _default_changes(context: dict[str, Any]) -> list[str]:
    stats = context.get("change_stats") or {}
    changes = []
    if stats.get("top_peer"):
        top = stats["top_peer"]
        changes.append(
            f"{_company_label(top['key'])} 관련 카드/이슈가 "
            f"최근 입력에서 {top['count']}건 포착되었습니다."
        )
    if stats.get("top_sector"):
        top = stats["top_sector"]
        changes.append(f"{top['key']} sector가 오늘 주요 비교 축으로 올라왔습니다.")
    changes.append(
        f"최근 {stats.get('window_days', 14)}일 누적 흐름 대비 오늘 신호를 분리했습니다."
    )
    return changes[:3]


def _keywords_from_context(context: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    for issue in context.get("current_issues", [])[:5]:
        if not isinstance(issue, dict):
            continue
        keywords.extend(str(item) for item in _list(issue.get("sectors")) if item)
        event_type = issue.get("event_type")
        if event_type:
            keywords.append(str(event_type))
    for card in context.get("recent_cards", [])[:5]:
        if not isinstance(card, dict):
            continue
        if card.get("event_type"):
            keywords.append(str(card["event_type"]))
        if card.get("peer_id"):
            keywords.append(_company_label(str(card["peer_id"])))
    stats = context.get("change_stats") or {}
    for key in ("top_sector", "top_event_type"):
        item = stats.get(key)
        if isinstance(item, dict) and item.get("key"):
            keywords.append(str(item["key"]))
    return _dedupe(keywords, limit=6) or ["AX", "Agentic AI", "운영 KPI"]


def _top_axis_from_stats(stats: dict[str, Any]) -> str:
    for key in ("top_sector", "top_peer", "top_event_type"):
        item = stats.get(key)
        if isinstance(item, dict) and item.get("key"):
            value = str(item["key"])
            if value.lower() in {"other", "unknown", "-"}:
                continue
            return _company_label(value) if key == "top_peer" else value
    return ""


def _proposal_axis_from_stats(stats: dict[str, Any]) -> str:
    item = stats.get("top_sector")
    if isinstance(item, dict) and item.get("key"):
        value = str(item["key"])
        if value.lower() not in {"other", "unknown", "-"}:
            return value
    return "AX"


def _top_axis(*counters: Counter[str]) -> str:
    for counter in counters:
        top = _counter_top(counter)
        if top:
            return _company_label(top["key"]) if top["key"] in COMPANIES else top["key"]
    return ""


def _counter_top(counter: Counter[str]) -> dict[str, Any] | None:
    if not counter:
        return None
    key, count = counter.most_common(1)[0]
    return {"key": key, "count": count}


def _delta_rows(
    current: Counter[str], history: Counter[str], window_days: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = set(current) | set(history)
    for key in keys:
        today = current.get(key, 0)
        historical_daily = history.get(key, 0) / max(window_days - 1, 1)
        rows.append(
            {
                "key": key,
                "today_count": today,
                "historical_daily_avg": round(historical_daily, 2),
                "delta_vs_daily_avg": round(today - historical_daily, 2),
            }
        )
    rows.sort(key=lambda item: item["delta_vs_daily_avg"], reverse=True)
    return rows[:5]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, default=str)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def _string_list(value: Any, fallback: list[str], limit: int, *, max_len: int = 140) -> list[str]:
    out = [
        _clip(_sanitize_public_text(item), max_len)
        for item in _list(value)
        if str(item or "").strip() and not _is_weak_or_mock_text(str(item))
    ]
    out = _dedupe(out, limit=limit)
    if out:
        return out
    return fallback[:limit]


def _is_weak_or_mock_text(value: str) -> bool:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return False
    return any(marker in cleaned for marker in _WEAK_OR_MOCK_MARKERS)


def _sanitize_public_text(value: Any) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    for source, replacement in _PUBLIC_TEXT_REPLACEMENTS:
        text_value = text_value.replace(source, replacement)
    text_value = _qualify_internal_score_text(text_value)
    text_value = re.sub(r"\b(?:IC|CN|raw)-[A-Za-z0-9_.:-]+\b", "근거", text_value)
    text_value = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "근거",
        text_value,
    )
    text_value = re.sub(r"\b[a-z]+_[a-z0-9_]+\b", "", text_value)
    text_value = re.sub(r"\s{2,}", " ", text_value).strip()
    return text_value


def _qualify_internal_score_text(text_value: str) -> str:
    """Keep internal scores available in JSON, but avoid exposing raw scoring math in copy."""
    replacements: tuple[tuple[str, str], ...] = (
        (
            r"(?:영향도|영향|중요도|salience|impact|importance)(?:\s*(?:score|점수|스코어))?"
            r"\s*(?:는|은|:|=)?\s*0?\.\d+\s*(?:로|으로|이고|이며|,)?",
            "내용 영향은 큰 편이고",
        ),
        (
            r"(?:노출도|노출|exposure)(?:\s*(?:score|점수|스코어))?"
            r"\s*(?:는|은|:|=)?\s*0?\.\d+\s*(?:로|으로|이고|이며|,)?",
            "보도 확산은 아직 낮은 편이고",
        ),
        (
            r"(?:visibility\s*gap|가시성\s*격차|노출\s*격차)"
            r"\s*(?:는|은|:|=)?\s*0?\.\d+\s*(?:로|으로|이고|이며|,)?",
            "내용 영향과 보도 확산 사이의 차이가 있어",
        ),
    )
    for pattern, replacement in replacements:
        text_value = re.sub(pattern, replacement, text_value, flags=re.IGNORECASE)
    text_value = re.sub(
        r"(?:내부\s*)?(?:점수|스코어)\s*(?:기준|상)?\s*0?\.\d+\s*(?:로|으로)?",
        "내부 판단 기준상",
        text_value,
    )
    text_value = re.sub(r"\s*,\s*", ", ", text_value)
    text_value = re.sub(r"\s{2,}", " ", text_value).strip(" ,")
    return text_value


def _int_list(value: Any) -> list[int]:
    out = []
    for item in _list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _dedupe(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _dedupe_int(values: list[int], *, limit: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _clip(value: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    return cleaned[:limit].rstrip()


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _company_label(company_id: str) -> str:
    return COMPANIES.get(company_id, {}).get("name_ko") or company_id or "Peer"


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:80] or "signal"
