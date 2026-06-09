"""TodayInsightAgent — home dashboard executive daily signal synthesis.

The agent compares today's integrated issues against accumulated JSON context
(`today_insight_reports.output_payload`), integrated issue history, company
profiles, and SK AX official context. The public output is intentionally compact
and UI-ready: key signal, watch point, next judgment, evidence, response, sources.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from datetime import UTC, date, datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.api.today_insight_schemas import (
    TodayInsightGenerateRequest,
    TodayInsightGenerateResponse,
)
from src.config.companies import COMPANIES
from src.config.company_tiers import SELF_COMPANY_IDS
from src.db.postgres import SessionLocal
from src.observability.langfuse_client import tracing_config
from src.services.profile_context_loader import ProfileContextLoader
from src.services.skax_profile_context_loader import SKAXProfileLoader
from src.services.today_insight_comparison_engine import (
    build_comparison_facts,
    build_ui_change_summary,
    format_evidence_change_lines,
    polish_executive_output,
)

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
_LLM_MODEL = os.getenv("TODAY_INSIGHT_MODEL", "gpt-4o")
_PROMPT_VERSION = "today-insight-v1.2-dual-lane-postprocess"
_MAX_PROMPT_JSON_CHARS = 44_000
_SIGNAL_LABELS = ("주요 신호", "관찰 포인트", "다음 판단")

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=0.18,
            max_completion_tokens=3200,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


_TODAY_INSIGHT_PROMPT = """\
당신은 SK AX CEO/임원 홈 대시보드의 Today's Insight를 작성하는 전략 인텔리전스 에이전트입니다.

목표:
- comparison_facts.primary_selection(이벤트 salience)을 먼저 말하고,
  comparison_facts.structural(보도량·비중 맥락)은 보조로 씁니다.
- 독자는 SK 임원입니다. 일반론, 캠페인 문구, "경쟁 환경 영향"만 있는 권고를 쓰지 않습니다.
- 답은 홈 화면 첫 영역에 노출되므로 짧지만, 근거-변화-판단의 밀도가 높아야 합니다.

Dual-lane 판단:
- Lane 1 (primary): salience_score 높은 확실한 이벤트 — M&A, 공시, 고임팩트 키워드.
  label=low_visibility_definite_event 이면 보도 1건이어도 headline/signal[0]에 반드시 반영.
- Lane 2 (context): structural 지표(건수, peer/sector delta, event mix)는
  "왜 놓치기 쉬운지/시장이 어떻게 보는지" 설명용. primary 후보를 제거하는 필터가 아님.

내부 판단 방식:
- primary_selection → structural 맥락 → SK AX 제안/운영 판단 변수 → 다음 확인 항목.
- reasoning step label: "관찰", "비교", "의미", "판단" 중에서만 사용.

절대 규칙:
1. comparison_facts 에 없는 수치를 만들지 않습니다.
   card_attached_numbers 가 비어 있으면 금액/규모 수치를 쓰지 않습니다.
2. "시장 확대", "경쟁 심화", "전략 강화", "경쟁 환경에 영향" 같은
   넓은 결론은 단독으로 쓰지 않습니다.
   반드시 event_type/sector/운영 KPI/검증 항목/제안 산출물로 좁힙니다.
3. response_direction은 실행 산출물이 보여야 합니다.
4. signal은 정확히 3개이며 label은 순서대로 "주요 신호", "관찰 포인트", "다음 판단"입니다.
5. signal[0]은 primary_selection.items[0]을 우선 반영합니다.
6. evidence.changes에는 comparison_facts.structural 또는 salience recurrence 중
   확인 가능한 항목만 씁니다.
7. sources는 실제 입력 id/url/title 만 사용합니다.

입력 JSON:
{context_json}

출력 JSON schema:
{{
  "headline": "오늘 홈 화면 H2 아래에 놓일 핵심 판단 1문장",
  "executive_summary": "오늘 변화가 왜 중요한지 2문장 이내",
  "executive_implication": "SK AX 임원 관점의 시사점 2문장 이내. 일반론 금지",
  "change_summary": [
    {{"label": "오늘 감지된 변화", "value": "N건"}},
    {{"label": "비교 기준", "value": "최근 N일"}},
    {{"label": "핵심 축", "value": "구체 peer/sector/keyword"}}
  ],
  "signals": [
    {{
      "id": "signal-key",
      "label": "주요 신호",
      "value": "홈 카드 버튼에 들어갈 1문장",
      "reasoning": [
        {{"stage": "관찰", "detail": "입력 근거를 좁혀 쓴 설명"}},
        {{"stage": "비교", "detail": "과거/프로필 대비 달라진 점"}},
        {{"stage": "의미", "detail": "SK AX 판단 기준에 주는 의미"}},
        {{"stage": "판단", "detail": "오늘 취할 판단"}}
      ],
      "evidence": {{
        "grounds": ["근거 1", "근거 2"],
        "changes": ["달라진 점 1", "달라진 점 2"],
        "related_keywords": ["키워드"],
        "source_ids": ["IC-... 또는 CN-... 또는 raw-..."]
      }}
    }}
  ],
  "response_direction": [
    {{
      "action": "구체 대응 산출물/판단 기준",
      "decision_owner": "임원/조직 후보",
      "time_horizon": "오늘/이번 주/이번 달",
      "rationale": "왜 이 액션인지",
      "evidence_refs": ["입력 source id"]
    }}
  ],
  "sources": [
    {{
      "id": "source id",
      "title": "제목",
      "source_name": "출처명",
      "publisher": "발행처",
      "url": "https://...",
      "published_at": "ISO 또는 문자열"
    }}
  ],
  "confidence": 0.0
}}

JSON만 출력하세요.
"""


class TodayInsightAgent:
    """Generate the home dashboard Today's Insight block."""

    def __init__(self) -> None:
        self.prompt_version = _PROMPT_VERSION

    async def generate(self, request: TodayInsightGenerateRequest | None = None) -> dict[str, Any]:
        req = request or TodayInsightGenerateRequest()
        anchor_date = req.anchor_date or datetime.now(KST).date()

        if req.preload_model:
            _preload_llm_client()

        if req.use_cached and not req.force_refresh:
            cached_record = _load_latest_report_record(anchor_date)
            if cached_record:
                return cached_record["payload"]

        if req.cache_only:
            return _scheduled_cache_pending_result(anchor_date)

        context = _build_context(req, anchor_date)
        if not context["current_issues"] and not context["recent_cards"]:
            result = _fallback_result(
                anchor_date=anchor_date,
                context=context,
                warning="today insight source data unavailable",
            )
            if req.save:
                _save_report(result, input_snapshot=context)
            return result

        prompt = _TODAY_INSIGHT_PROMPT.replace(
            "{context_json}",
            _compact_json(context, max_chars=_MAX_PROMPT_JSON_CHARS),
        )

        try:
            response = await asyncio.to_thread(
                _get_llm().invoke,
                prompt,
                tracing_config(
                    agent="TodayInsightAgent",
                    phase="generate",
                    prompt_version=_PROMPT_VERSION,
                ),
            )
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            parsed = json.loads(content)
            result = _normalize_result(parsed, anchor_date=anchor_date, context=context)
        except Exception as exc:  # noqa: BLE001
            log.exception("TodayInsightAgent LLM generation failed | error=%s", exc)
            result = _fallback_result(
                anchor_date=anchor_date,
                context=context,
                warning=f"LLM generation failed: {type(exc).__name__}",
            )

        if req.save:
            _save_report(result, input_snapshot=context)
        return result


def _build_context(req: TodayInsightGenerateRequest, anchor_date: date) -> dict[str, Any]:
    issues = _fetch_integrated_issues(
        anchor_date=anchor_date,
        window_days=req.window_days,
        limit=max(req.max_issues * 3, req.max_issues),
    )
    current_issues = [
        row for row in issues if row.get("created_date_kst") == anchor_date.isoformat()
    ]
    if not current_issues:
        current_issues = issues[: min(req.max_issues, len(issues))]
    history_issues = [
        row for row in issues if row.get("id") not in {item.get("id") for item in current_issues}
    ][: max(req.max_issues * 2, 6)]

    issue_ids = [str(row["id"]) for row in current_issues if row.get("id")]
    cards = _fetch_cards_for_issues(
        issue_ids,
        anchor_date=anchor_date,
        window_days=req.window_days,
        limit=req.max_cards,
    )
    if len(cards) < req.max_cards:
        supplemental_cards = _fetch_recent_cards(
            anchor_date=anchor_date,
            window_days=req.window_days,
            limit=req.max_cards - len(cards),
            exclude_ids=[
                str(card.get("id")) for card in cards if str(card.get("id") or "").strip()
            ],
        )
        cards = [*cards, *supplemental_cards][: req.max_cards]
    peer_ids = _dedupe(
        [
            str(value)
            for value in [
                *[row.get("main_company") for row in current_issues + history_issues],
                *[card.get("peer_id") for card in cards],
            ]
            if value and str(value) not in SELF_COMPANY_IDS
        ],
        limit=8,
    )
    sectors = _dedupe(
        [
            str(sector)
            for row in current_issues + history_issues
            for sector in _list(row.get("sectors"))
            if sector
        ],
        limit=8,
    )

    profile_context = _load_profile_context(peer_ids=peer_ids, sectors=sectors, req=req)
    skax_context = _load_skax_context(sectors=sectors)
    prior_reports = _fetch_prior_today_reports(anchor_date=anchor_date, limit=5)
    ledger_context = _fetch_analysis_ledger(peer_ids=peer_ids, window_days=req.window_days, limit=8)
    sources = _collect_sources(current_issues=current_issues, cards=cards, limit=12)
    stats = _build_change_stats(
        anchor_date=anchor_date,
        current_issues=current_issues,
        history_issues=history_issues,
        cards=cards,
        window_days=req.window_days,
    )
    comparison_facts = build_comparison_facts(
        anchor_date=anchor_date,
        window_days=req.window_days,
        current_issues=current_issues,
        history_issues=history_issues,
        cards=cards,
        change_stats=stats,
        prior_reports=prior_reports,
    )

    return {
        "report_date": anchor_date.isoformat(),
        "window_days": req.window_days,
        "current_issues": [_issue_for_prompt(row) for row in current_issues[: req.max_issues]],
        "history_issues": [_issue_for_prompt(row) for row in history_issues[: req.max_issues * 2]],
        "recent_cards": [_card_for_prompt(card) for card in cards[: req.max_cards]],
        "change_stats": stats,
        "comparison_facts": comparison_facts,
        "prior_today_insight_memory": prior_reports,
        "analysis_ledger_context": ledger_context,
        "profile_context": profile_context,
        "skax_context": skax_context,
        "sources": sources,
        "request_context": req.context or {},
    }


def _fetch_integrated_issues(
    *, anchor_date: date, window_days: int, limit: int
) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT id::text AS id,
                           cluster_id,
                           main_company,
                           event_type,
                           source_family,
                           confidence,
                           headline,
                           one_line_summary,
                           analyzed_source_ids,
                           source_ids,
                           sectors,
                           mentioned_peer_companies,
                           content_summary,
                           issue_frame,
                           sources,
                           evidence,
                           quality,
                           payload,
                           created_at,
                           updated_at,
                           ((created_at AT TIME ZONE 'Asia/Seoul')::date)::text AS created_date_kst
                      FROM integrated_issues
                     WHERE is_current = TRUE
                       AND status = 'active'
                       AND is_valid = TRUE
                       AND (created_at AT TIME ZONE 'Asia/Seoul')::date
                           >= CAST(:anchor_date AS date) - (:window_days * INTERVAL '1 day')
                     ORDER BY
                       CASE
                         WHEN (created_at AT TIME ZONE 'Asia/Seoul')::date
                              = CAST(:anchor_date AS date)
                         THEN 0 ELSE 1
                       END,
                       confidence DESC NULLS LAST,
                       created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        "anchor_date": anchor_date.isoformat(),
                        "window_days": int(window_days),
                        "limit": int(limit),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight integrated_issues lookup failed | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _fetch_cards_for_issues(
    issue_ids: list[str],
    *,
    anchor_date: date,
    window_days: int,
    limit: int,
) -> list[dict[str, Any]]:
    if not issue_ids:
        return []
    placeholders = ", ".join(f"CAST(:issue_{idx} AS uuid)" for idx in range(len(issue_ids)))
    params: dict[str, Any] = {f"issue_{idx}": issue_id for idx, issue_id in enumerate(issue_ids)}
    params["limit"] = int(limit)
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        f"""
                    SELECT id,
                           title,
                           COALESCE(peer_company_id, company) AS peer_id,
                           summary_lines,
                           event_type,
                           importance,
                           importance_score,
                           implication,
                           primary_keyword_category,
                           evidence_payload,
                           source_raw_article_ids,
                           sources,
                           integrated_issue_id::text AS integrated_issue_id,
                           created_at
                      FROM card_news
                     WHERE integrated_issue_id IN ({placeholders})
                       AND (created_at AT TIME ZONE 'Asia/Seoul')::date
                           BETWEEN CAST(:anchor_date AS date) - (:window_days * INTERVAL '1 day')
                               AND CAST(:anchor_date AS date)
                     ORDER BY importance_score DESC NULLS LAST, created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        **params,
                        "anchor_date": anchor_date.isoformat(),
                        "window_days": int(window_days),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight card lookup failed | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _fetch_recent_cards(
    *,
    anchor_date: date,
    window_days: int,
    limit: int,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    clean_exclude_ids = _dedupe(
        [str(card_id) for card_id in exclude_ids or [] if str(card_id or "").strip()],
        limit=100,
    )
    exclude_clause = ""
    params: dict[str, Any] = {
        "anchor_date": anchor_date.isoformat(),
        "window_days": int(window_days),
        "limit": int(limit),
    }
    if clean_exclude_ids:
        placeholders = ", ".join(f":exclude_{idx}" for idx in range(len(clean_exclude_ids)))
        exclude_clause = f"AND id NOT IN ({placeholders})"
        params.update({f"exclude_{idx}": card_id for idx, card_id in enumerate(clean_exclude_ids)})

    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        f"""
                    SELECT id,
                           title,
                           COALESCE(peer_company_id, company) AS peer_id,
                           summary_lines,
                           event_type,
                           importance,
                           importance_score,
                           implication,
                           primary_keyword_category,
                           evidence_payload,
                           source_raw_article_ids,
                           sources,
                           integrated_issue_id::text AS integrated_issue_id,
                           created_at
                      FROM card_news
                     WHERE (created_at AT TIME ZONE 'Asia/Seoul')::date
                           BETWEEN CAST(:anchor_date AS date) - (:window_days * INTERVAL '1 day')
                               AND CAST(:anchor_date AS date)
                       {exclude_clause}
                     ORDER BY importance_score DESC NULLS LAST, created_at DESC
                     LIMIT :limit
                    """
                    ),
                    params,
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight recent card lookup failed | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _fetch_prior_today_reports(*, anchor_date: date, limit: int) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT report_date::text AS report_date,
                           headline,
                           executive_summary,
                           output_payload,
                           source_integrated_issue_ids,
                           source_card_ids,
                           confidence,
                           created_at
                      FROM today_insight_reports
                     WHERE report_date < CAST(:anchor_date AS date)
                       AND status = 'active'
                     ORDER BY report_date DESC, created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"anchor_date": anchor_date.isoformat(), "limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("prior today insight reports unavailable | error=%s", exc)
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        item = _json_ready(dict(row))
        payload = item.get("output_payload") if isinstance(item.get("output_payload"), dict) else {}
        out.append(
            {
                "report_date": item.get("report_date"),
                "headline": item.get("headline") or payload.get("headline"),
                "executive_summary": item.get("executive_summary")
                or payload.get("executive_summary"),
                "signal_values": [
                    signal.get("value")
                    for signal in _list(payload.get("signals"))
                    if isinstance(signal, dict)
                ][:3],
                "source_integrated_issue_ids": _list(item.get("source_integrated_issue_ids")),
                "source_card_ids": _list(item.get("source_card_ids")),
                "confidence": item.get("confidence"),
            }
        )
    return out


def _load_latest_report(anchor_date: date) -> dict[str, Any] | None:
    record = _load_latest_report_record(anchor_date)
    return record["payload"] if record else None


def _load_latest_report_record(anchor_date: date) -> dict[str, Any] | None:
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                    SELECT output_payload,
                           created_at
                      FROM today_insight_reports
                     WHERE report_date = CAST(:anchor_date AS date)
                       AND status = 'active'
                     ORDER BY created_at DESC
                     LIMIT 1
                    """
                    ),
                    {"anchor_date": anchor_date.isoformat()},
                )
                .mappings()
                .first()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("today insight cache lookup skipped | error=%s", exc)
        return None
    if not row:
        return None
    payload = row.get("output_payload")
    if not isinstance(payload, dict):
        return None
    return {
        "payload": _json_ready(payload),
        "created_at": row.get("created_at"),
    }


def _preload_llm_client() -> None:
    try:
        _get_llm()
    except Exception as exc:  # noqa: BLE001
        log.debug("TodayInsight LLM client preload skipped | error=%s", exc)


def _scheduled_cache_pending_result(anchor_date: date) -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()
    return TodayInsightGenerateResponse.model_validate(
        {
            "report_date": anchor_date.isoformat(),
            "generated_at": now_iso,
            "headline": "Today's Insight 08:10 업데이트 대기",
            "executive_summary": (
                "오늘 기준 저장된 Today's Insight가 아직 없습니다. "
                "운영 정책상 신규 생성은 평일 오전 8시 10분 스케줄에서만 수행합니다."
            ),
            "executive_implication": (
                "이 응답은 전략 판단 근거가 아니라 캐시 상태 안내입니다. "
                "08:10 생성 결과가 저장된 뒤 출처 포함 인사이트를 사용해야 합니다."
            ),
            "change_summary": [
                {"label": "업데이트 정책", "value": "일 1회 08:10"},
                {"label": "현재 상태", "value": "캐시 대기"},
                {"label": "생성 조건", "value": "스케줄 전용"},
            ],
            "signals": [
                {
                    "id": "scheduled-cache-status",
                    "label": "주요 신호",
                    "value": "저장된 오늘 인사이트가 아직 없습니다",
                    "reasoning": [
                        {
                            "stage": "관찰",
                            "detail": (
                                "today_insight_reports 캐시를 조회했지만 오늘 결과가 없습니다."
                            ),
                        },
                        {
                            "stage": "판단",
                            "detail": (
                                "홈 조회에서는 신규 생성하지 않고 08:10 스케줄 결과를 기다립니다."
                            ),
                        },
                    ],
                    "evidence": {
                        "grounds": ["today_insight_reports cache miss"],
                        "changes": ["실제 변화 분석 전 상태"],
                        "related_keywords": ["cache_only", "scheduled update"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
                {
                    "id": "scheduled-cache-watch",
                    "label": "관찰 포인트",
                    "value": "08:10 스케줄 실행 및 저장 여부 확인 필요",
                    "reasoning": [
                        {
                            "stage": "관찰",
                            "detail": "로그인 warm-up과 홈 조회는 캐시 읽기만 수행합니다.",
                        },
                        {
                            "stage": "판단",
                            "detail": (
                                "스케줄 실패 시 운영 로그와 axis-ai 연결 상태를 확인해야 합니다."
                            ),
                        },
                    ],
                    "evidence": {
                        "grounds": ["cache_only request"],
                        "changes": ["사용자 진입 시점의 일중 재생성 차단"],
                        "related_keywords": ["warm-up", "08:10"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
                {
                    "id": "scheduled-cache-next",
                    "label": "다음 판단",
                    "value": "오전 생성 결과가 저장된 뒤 홈 화면에 노출",
                    "reasoning": [
                        {"stage": "관찰", "detail": "하루 한 번 생성 정책을 우선 적용했습니다."},
                        {
                            "stage": "판단",
                            "detail": "임원용 판단은 저장된 출처 포함 결과만 사용합니다.",
                        },
                    ],
                    "evidence": {
                        "grounds": ["daily scheduled generation policy"],
                        "changes": ["초고중요도 카드 기반 일중 refresh 예외 제거"],
                        "related_keywords": ["daily cache", "executive insight"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
            ],
            "response_direction": [
                {
                    "action": (
                        "08:10 스케줄러가 today_insight_reports에 결과를 저장했는지 확인합니다."
                    ),
                    "decision_owner": "플랫폼 운영",
                    "time_horizon": "오전",
                    "rationale": "로그인과 홈 조회가 신규 생성 경로로 변하지 않게 하기 위함입니다.",
                    "evidence_refs": ["scheduled-cache"],
                }
            ],
            "sources": [
                {
                    "id": "scheduled-cache",
                    "title": "Today's Insight scheduled cache status",
                    "source_name": "AXIS AI",
                    "publisher": "AXIS",
                    "url": "",
                    "published_at": now_iso,
                }
            ],
            "source_integrated_issue_ids": [],
            "source_card_ids": [],
            "peer_ids": [],
            "sectors": [],
            "confidence": 0.0,
            "provenance": {
                "mode": "cache_only",
                "update_policy": "daily_0810_kst",
                "prompt_version": _PROMPT_VERSION,
            },
            "warning": "cache_only; scheduled generation pending",
        }
    ).model_dump()


def _fetch_analysis_ledger(
    *, peer_ids: list[str], window_days: int, limit: int
) -> list[dict[str, Any]]:
    del peer_ids  # broad recent ledger is useful even when peer aliases are sparse.
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT analysis_type,
                           analysis_id,
                           peer_ids,
                           conclusion_one_liner,
                           confidence,
                           source_card_ids,
                           sk_ax_implication,
                           created_at
                      FROM analysis_ledger
                     WHERE confidence >= 0.6
                       AND superseded_by IS NULL
                       AND included_in_pack = TRUE
                       AND created_at >= now() - (:window_days * INTERVAL '1 day')
                     ORDER BY created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"window_days": int(window_days), "limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("analysis ledger context unavailable | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _load_profile_context(
    *, peer_ids: list[str], sectors: list[str], req: TodayInsightGenerateRequest
) -> dict[str, Any]:
    if not peer_ids:
        return {"skax_profile": {}, "peer_profiles": {}, "sector_context": {}}
    try:
        return (
            ProfileContextLoader()
            .load(
                companies=peer_ids[:6],
                sectors=sectors[:6],
                lookback_days=req.window_days,
            )
            .to_dict()
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("profile context load failed | error=%s", exc)
        return {"skax_profile": {}, "peer_profiles": {}, "sector_context": {}}


def _load_skax_context(*, sectors: list[str]) -> dict[str, Any]:
    try:
        return SKAXProfileLoader().load(
            sectors[:6],
            max_documents=5,
            max_newsroom_documents=5,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("SK AX official context load failed | error=%s", exc)
        return {}


def _build_change_stats(
    *,
    anchor_date: date,
    current_issues: list[dict[str, Any]],
    history_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    current_peer_counts = Counter(
        str(row.get("main_company")) for row in current_issues if row.get("main_company")
    )
    history_peer_counts = Counter(
        str(row.get("main_company")) for row in history_issues if row.get("main_company")
    )
    card_peer_counts = Counter(str(card.get("peer_id")) for card in cards if card.get("peer_id"))
    current_sector_counts = Counter(
        str(sector) for row in current_issues for sector in _list(row.get("sectors")) if sector
    )
    history_sector_counts = Counter(
        str(sector) for row in history_issues for sector in _list(row.get("sectors")) if sector
    )
    current_event_counts = Counter(
        str(row.get("event_type")) for row in current_issues if row.get("event_type")
    )
    card_event_counts = Counter(
        str(card.get("event_type")) for card in cards if card.get("event_type")
    )
    combined_peer_counts = current_peer_counts + card_peer_counts
    combined_event_counts = current_event_counts + card_event_counts
    top_axis = _top_axis(current_sector_counts, combined_peer_counts, combined_event_counts)
    return {
        "anchor_date": anchor_date.isoformat(),
        "today_issue_count": len(current_issues),
        "recent_card_count": len(cards),
        "history_issue_count": len(history_issues),
        "window_days": window_days,
        "top_peer": _counter_top(combined_peer_counts),
        "top_sector": _counter_top(current_sector_counts),
        "top_event_type": _counter_top(combined_event_counts),
        "peer_delta_vs_window": _delta_rows(combined_peer_counts, history_peer_counts, window_days),
        "sector_delta_vs_window": _delta_rows(
            current_sector_counts, history_sector_counts, window_days
        ),
        "default_change_summary": [
            {"label": "오늘 감지된 변화", "value": f"{len(current_issues) + len(cards)}건"},
            {"label": "비교 기준", "value": f"최근 {window_days}일"},
            {"label": "핵심 축", "value": top_axis or "-"},
        ],
    }


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
        "comparison_coverage": (
            (context.get("comparison_facts") or {}).get("coverage")
            if isinstance(context.get("comparison_facts"), dict)
            else {}
        ),
    }
    if not base["headline"]:
        base["headline"] = _fallback_headline(context)
    if not base["executive_summary"]:
        base["executive_summary"] = _fallback_summary(context)
    if not base["executive_implication"]:
        base["executive_implication"] = _fallback_implication(context)
    base = polish_executive_output(base, context=context)
    response = TodayInsightGenerateResponse.model_validate(base)
    return response.model_dump()


def _normalize_change_summary(value: Any, fallback: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    normalized = []
    for item in rows[:3]:
        label = _clip(str(item.get("label") or ""), 24)
        val = _clip(str(item.get("value") or ""), 36)
        if label and val:
            normalized.append({"label": label, "value": val})
    while len(normalized) < 3:
        normalized.append(fallback[len(normalized)])
    return normalized[:3]


def _normalize_signals(value: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    fallback = _fallback_signals(context)
    normalized: list[dict[str, Any]] = []
    for idx in range(3):
        item = rows[idx] if idx < len(rows) else fallback[idx]
        raw_evidence = item.get("evidence")
        evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
        reasoning: list[dict[str, Any]] = [
            step for step in _list(item.get("reasoning")) if isinstance(step, dict)
        ][:4]
        if not reasoning:
            reasoning = fallback[idx]["reasoning"]
        signal = {
            "id": _slug(str(item.get("id") or f"signal-{idx + 1}")),
            "label": _SIGNAL_LABELS[idx],
            "value": _clip(str(item.get("value") or fallback[idx]["value"]), 96),
            "reasoning": [
                {
                    "stage": _clip(
                        str(step.get("stage") or fallback[idx]["reasoning"][0]["stage"]), 14
                    ),
                    "detail": _clip(str(step.get("detail") or ""), 150),
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

    existing = [
        str(item).strip() for item in _list(evidence.get("changes")) if str(item or "").strip()
    ]
    merged = existing[:]
    for line in computed_changes:
        if line not in merged:
            merged.append(line)
        if len(merged) >= 3:
            break

    if idx == 0 and len(merged) < 3:
        for line in computed_changes:
            if line not in merged:
                merged.append(line)
            if len(merged) >= 3:
                break

    evidence["changes"] = merged[:3]
    signal["evidence"] = evidence
    return signal


def _normalize_actions(value: Any, *, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in _list(value) if isinstance(item, dict)]
    fallback = _fallback_actions(context)
    normalized = []
    for idx in range(3):
        item = rows[idx] if idx < len(rows) else fallback[idx]
        normalized.append(
            {
                "action": _clip(str(item.get("action") or fallback[idx]["action"]), 180),
                "decision_owner": _clip(
                    str(item.get("decision_owner") or fallback[idx]["decision_owner"]),
                    48,
                ),
                "time_horizon": _clip(
                    str(item.get("time_horizon") or fallback[idx]["time_horizon"]),
                    32,
                ),
                "rationale": _clip(str(item.get("rationale") or fallback[idx]["rationale"]), 160),
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
    rows = [item for item in _list(value) if isinstance(item, dict)] or fallback
    normalized: list[dict[str, Any]] = []
    for item in rows[:8]:
        source_id = str(item.get("id") or item.get("source_id") or item.get("url") or "")
        title = str(item.get("title") or item.get("headline") or "")
        if not source_id and not title:
            continue
        normalized.append(
            {
                "id": _clip(source_id or f"source-{len(normalized) + 1}", 120),
                "title": _clip(title, 180),
                "source_name": _clip(str(item.get("source_name") or item.get("source") or ""), 80),
                "publisher": _clip(str(item.get("publisher") or ""), 80),
                "url": _clip(str(item.get("url") or ""), 500),
                "published_at": str(item.get("published_at") or item.get("created_at") or "")
                or None,
            }
        )
    return normalized


def _fallback_result(
    *,
    anchor_date: date,
    context: dict[str, Any],
    warning: str | None = None,
) -> dict[str, Any]:
    return _normalize_result(
        {
            "headline": _fallback_headline(context),
            "executive_summary": _fallback_summary(context),
            "executive_implication": _fallback_implication(context),
            "change_summary": context.get("change_stats", {}).get("default_change_summary", []),
            "signals": _fallback_signals(context),
            "response_direction": _fallback_actions(context),
            "sources": context.get("sources", []),
            "confidence": 0.45 if warning else 0.62,
            "warning": warning,
        },
        anchor_date=anchor_date,
        context=context,
    )


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
            f"{company} 신호를 기준으로 오늘의 AX 제안 판단을 재정렬해야 합니다: {headline}", 120
        )
    first_card = _first_card(context)
    if first_card:
        company = _company_label(str(first_card.get("peer_id") or ""))
        title = str(first_card.get("title") or "")
        return _clip(
            f"{company} 카드뉴스 신호를 기준으로 오늘의 AX 제안 판단을 재점검해야 합니다: {title}",
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
            f"오늘 통합 이슈 {count}건에서 {axis or 'AX 실행'} 관련 변화가 "
            "우선 포착됐습니다. 과거 누적 결과와 비교해 제안서·PoC·운영 "
            "책임 범위를 다시 확인할 필요가 있습니다."
        )
    if card_count:
        return (
            f"최근 {stats.get('window_days', 60)}일 카드뉴스 {card_count}건에서 "
            f"{axis or 'AX 실행'} 관련 신호가 우선 포착됐습니다. 통합 이슈가 비어 있어도 "
            "카드뉴스 근거를 기준으로 제안서·PoC·운영 책임 범위를 점검합니다."
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
        "고객 제안서의 운영 KPI·보안 책임·검증 지표 중 어느 항목을 "
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
                        "label": "주요 신호",
                        "value": _clip(
                            title or "오늘 primary salience 신호",
                            96,
                        ),
                        "reasoning": [
                            {"stage": "관찰", "detail": hint or title or "primary_selection 기준"},
                            {
                                "stage": "비교",
                                "detail": (
                                    f"salience {lead.get('salience_score')} / "
                                    f"exposure {lead.get('exposure_score')}"
                                ),
                            },
                            {
                                "stage": "의미",
                                "detail": (
                                    "단건 고임팩트 이벤트로 분류"
                                    if label == "low_visibility_definite_event"
                                    else "오늘 우선 판단 축"
                                ),
                            },
                            {"stage": "판단", "detail": "제안·PoC·운영 책임 범위 재점검"},
                        ],
                        "evidence": {
                            "grounds": [hint] if hint else _default_grounds(context)[:2],
                            "changes": structural or _default_changes(context),
                            "related_keywords": _keywords_from_context(context),
                            "source_ids": source_ids,
                        },
                    },
                    {
                        "id": "signal-watch-volume",
                        "label": "관찰 포인트",
                        "value": _clip("보도량·sector 비중 맥락 확인", 96),
                        "reasoning": [
                            {
                                "stage": "관찰",
                                "detail": "structural 지표는 primary를 대체하지 않습니다.",
                            },
                            {
                                "stage": "비교",
                                "detail": ", ".join(structural) or "rolling baseline 대비 변화",
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
                        "id": "signal-next-judgment",
                        "label": "다음 판단",
                        "value": "제안서·PoC·운영모델에서 무엇을 바꿀지 오늘 결정",
                        "reasoning": [
                            {"stage": "판단", "detail": "primary 이벤트 기준 의사결정 항목 확정"},
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
    signal_axis = "AX 제안" if top_axis == company else top_axis
    grounds = _default_grounds(context)
    changes = _default_changes(context)
    return [
        {
            "id": "signal-primary-change",
            "label": "주요 신호",
            "value": _clip(f"{company} 신호가 {signal_axis} 판단 축을 끌어올림", 96),
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
                    "detail": "단순 보도량보다 제안 산출물에 반영할 판단 기준을 우선했습니다.",
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
            "id": "signal-watch-point",
            "label": "관찰 포인트",
            "value": _clip(f"{top_axis} 신호가 단건 뉴스인지 반복 패턴인지 확인 필요", 96),
            "reasoning": [
                {
                    "stage": "관찰",
                    "detail": "오늘 신호를 과거 today insight 메모리와 대조했습니다.",
                },
                {
                    "stage": "비교",
                    "detail": "동일 peer·sector 반복 여부를 별도 관찰 포인트로 분리했습니다.",
                },
                {
                    "stage": "의미",
                    "detail": "반복성이 확인될 때만 영업·제안 우선순위를 높이는 편이 안전합니다.",
                },
            ],
            "evidence": {
                "grounds": grounds[:2],
                "changes": changes[:2],
                "related_keywords": _keywords_from_context(context),
                "source_ids": source_ids,
            },
        },
        {
            "id": "signal-next-judgment",
            "label": "다음 판단",
            "value": "제안서·PoC·운영모델에서 무엇을 바꿀지 오늘 결정",
            "reasoning": [
                {"stage": "관찰", "detail": "SK AX 공식 관점과 피어 프로필을 함께 검토했습니다."},
                {
                    "stage": "비교",
                    "detail": (
                        "범용 AX 메시지가 아니라 고객 평가 항목 변화 여부를 기준으로 삼았습니다."
                    ),
                },
                {"stage": "판단", "detail": "다음 판단은 제안 산출물의 구조 변경 여부입니다."},
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
                f"{axis} 제안서 첫 3장에 고객 운영 KPI, 보안 책임 범위, "
                "PoC 검증 지표를 분리해 배치합니다."
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
                "근거가 약한 시장 확대 단정을 제거하고, 실제 제안에 쓸 수 "
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


def _save_report(result: dict[str, Any], *, input_snapshot: dict[str, Any]) -> None:
    try:
        with SessionLocal() as db:
            db.execute(
                text(
                    """
                    INSERT INTO today_insight_reports (
                        report_date, schema_version, prompt_version, status,
                        headline, executive_summary, executive_implication,
                        source_integrated_issue_ids, source_card_ids, source_raw_article_ids,
                        peer_ids, sectors, input_snapshot, output_payload, provenance,
                        confidence
                    ) VALUES (
                        CAST(:report_date AS date), :schema_version, :prompt_version, 'active',
                        :headline, :executive_summary, :executive_implication,
                        CAST(:source_integrated_issue_ids AS uuid[]),
                        CAST(:source_card_ids AS text[]),
                        CAST(:source_raw_article_ids AS bigint[]),
                        CAST(:peer_ids AS text[]),
                        CAST(:sectors AS text[]),
                        CAST(:input_snapshot AS jsonb),
                        CAST(:output_payload AS jsonb),
                        CAST(:provenance AS jsonb),
                        :confidence
                    )
                    """
                ),
                {
                    "report_date": result.get("report_date"),
                    "schema_version": "today_insight_v1",
                    "prompt_version": _PROMPT_VERSION,
                    "headline": result.get("headline"),
                    "executive_summary": result.get("executive_summary"),
                    "executive_implication": result.get("executive_implication"),
                    "source_integrated_issue_ids": _uuid_array_literal(
                        result.get("source_integrated_issue_ids") or []
                    ),
                    "source_card_ids": _pg_text_array(result.get("source_card_ids") or []),
                    "source_raw_article_ids": _pg_bigint_array(_source_raw_ids(input_snapshot)),
                    "peer_ids": _pg_text_array(result.get("peer_ids") or []),
                    "sectors": _pg_text_array(result.get("sectors") or []),
                    "input_snapshot": _json_dumps(input_snapshot),
                    "output_payload": _json_dumps(result),
                    "provenance": _json_dumps(result.get("provenance") or {}),
                    "confidence": _clamp_float(result.get("confidence"), default=0.0),
                },
            )
            db.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("today insight report save skipped | error=%s", exc)


def _issue_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "created_date_kst": row.get("created_date_kst"),
        "main_company": row.get("main_company"),
        "company_label": _company_label(str(row.get("main_company") or "")),
        "event_type": row.get("event_type"),
        "source_family": row.get("source_family"),
        "sectors": _list(row.get("sectors"))[:6],
        "confidence": row.get("confidence"),
        "headline": _clip(str(row.get("headline") or ""), 240),
        "one_line_summary": _clip(str(row.get("one_line_summary") or ""), 300),
        "content_summary": _clip(str(row.get("content_summary") or ""), 700),
        "issue_frame": _json_ready(row.get("issue_frame") or {}),
        "evidence": _compact_evidence(row.get("evidence")),
        "source_ids": _list(row.get("source_ids"))[:8],
        "sources": _compact_sources(row.get("sources"), limit=4),
    }


def _card_for_prompt(card: dict[str, Any]) -> dict[str, Any]:
    raw_implication = card.get("implication")
    implication: dict[str, Any] = raw_implication if isinstance(raw_implication, dict) else {}
    sector = card.get("primary_keyword_category") or implication.get("sector") or ""
    return {
        "id": card.get("id"),
        "integrated_issue_id": card.get("integrated_issue_id"),
        "peer_id": card.get("peer_id"),
        "title": _clip(str(card.get("title") or ""), 220),
        "summary_lines": _list(card.get("summary_lines"))[:3],
        "event_type": card.get("event_type"),
        "sector": sector,
        "importance": card.get("importance"),
        "importance_score": card.get("importance_score"),
        "exposure_score": implication.get("exposure_score"),
        "skax_implication": _clip(
            str(
                implication.get("potential_impact")
                or implication.get("why_important")
                or implication.get("sk_ax_implication")
                or ""
            ),
            360,
        ),
        "sources": _compact_sources(card.get("sources"), limit=3),
    }


def _collect_sources(
    *,
    current_issues: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for issue in current_issues:
        issue_id = str(issue.get("id") or "")
        if issue_id and issue_id not in seen:
            seen.add(issue_id)
            out.append(
                {
                    "id": issue_id,
                    "title": issue.get("headline") or issue.get("one_line_summary") or issue_id,
                    "source_name": "integrated_issues",
                    "publisher": _company_label(str(issue.get("main_company") or "")),
                    "url": "",
                    "published_at": str(issue.get("created_at") or ""),
                }
            )
        for source in _compact_sources(issue.get("sources"), limit=4):
            key = str(source.get("url") or source.get("id") or source.get("title") or "")
            if key and key not in seen:
                seen.add(key)
                out.append(source)
        if len(out) >= limit:
            return out[:limit]

    for card in cards:
        card_id = str(card.get("id") or "")
        if card_id and card_id not in seen:
            seen.add(card_id)
            out.append(
                {
                    "id": card_id,
                    "title": card.get("title") or card_id,
                    "source_name": "card_news",
                    "publisher": _company_label(str(card.get("peer_id") or "")),
                    "url": "",
                    "published_at": str(card.get("created_at") or ""),
                }
            )
        for source in _compact_sources(card.get("sources"), limit=3):
            key = str(source.get("url") or source.get("id") or source.get("title") or "")
            if key and key not in seen:
                seen.add(key)
                out.append(source)
        if len(out) >= limit:
            return out[:limit]
    return out[:limit]


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
            return _company_label(value) if key == "top_peer" else value
    return ""


def _proposal_axis_from_stats(stats: dict[str, Any]) -> str:
    item = stats.get("top_sector")
    if isinstance(item, dict) and item.get("key"):
        return str(item["key"])
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


def _source_raw_ids(context: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for issue in context.get("current_issues", []):
        if isinstance(issue, dict):
            ids.extend(_int_list(issue.get("source_ids")))
    for card in context.get("recent_cards", []):
        if isinstance(card, dict):
            ids.extend(_int_list(card.get("source_raw_article_ids")))
    return _dedupe_int(ids, limit=50)


def _uuid_array_literal(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if value]
    if not cleaned:
        return "{}"
    return "{" + ",".join(cleaned) + "}"


def _pg_text_array(values: list[Any]) -> str:
    cleaned = [str(value).replace('"', '\\"') for value in values if value]
    if not cleaned:
        return "{}"
    return "{" + ",".join(f'"{value}"' for value in cleaned) + "}"


def _pg_bigint_array(values: list[int]) -> str:
    if not values:
        return "{}"
    return "{" + ",".join(str(int(value)) for value in values) + "}"


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


def _compact_json(value: Any, *, max_chars: int) -> str:
    text_value = _json_dumps(value)
    if len(text_value) <= max_chars:
        return text_value
    return text_value[:max_chars] + "\n...TRUNCATED..."


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
    out = [_clip(str(item), max_len) for item in _list(value) if str(item or "").strip()]
    out = _dedupe(out, limit=limit)
    if out:
        return out
    return fallback[:limit]


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
