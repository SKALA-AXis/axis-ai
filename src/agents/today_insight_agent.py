"""TodayInsightAgent — home dashboard executive daily signal synthesis.

The agent compares today's integrated issues against accumulated JSON context
(`today_insight_reports.output_payload`), integrated issue history, company
profiles, and SK AX official context. The public output is intentionally compact
and UI-ready: key point, watch point, implication, response direction, evidence,
and sources.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Mapping
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.config.companies import COMPANIES
from src.config.company_tiers import SELF_COMPANY_IDS
from src.contracts.today_insight_schemas import (
    TodayInsightGenerateRequest,
    TodayInsightGenerateResponse,
)
from src.db.postgres import SessionLocal
from src.llm import LLMSpec, build_chat_llm
from src.observability.langfuse_client import tracing_config
from src.services.peer_id_aliases import PEER_ID_ALIASES, normalize_to_canonical_id
from src.services.profile_context_loader import ProfileContextLoader
from src.services.skax_profile_context_loader import SKAXProfileLoader
from src.services.today_insight_comparison_engine import (
    build_comparison_facts,
    build_ui_change_summary,
    format_evidence_change_lines,
    polish_executive_output,
    trim_comparison_facts_for_prompt,
)

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
_LLM_MODEL = os.getenv("TODAY_INSIGHT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-5.5"
_PROMPT_VERSION = "today-insight-v1.4-qualitative-signals"
_LLM_CONTEXT_MAX_CHARS = 48_000
_LLM_CONTEXT_DROP_ORDER = (
    "analysis_ledger_context",
    "prior_today_insight_memory",
    "history_issues",
    "skax_context",
    "profile_context",
)
_SIGNAL_LABELS = ("주요 신호", "관찰 포인트", "다음 판단")
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

_llm: ChatOpenAI | None = None


def _llm_max_completion_tokens() -> int:
    default = "12000" if str(_LLM_MODEL).startswith("gpt-5") else "3200"
    return int(os.getenv("TODAY_INSIGHT_MAX_COMPLETION_TOKENS", default))


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # gpt-5 reasoning_effort 분기·json_object 래핑은 공용 팩토리가 처리.
        _llm = build_chat_llm(
            LLMSpec(
                model=_LLM_MODEL,
                temperature=0.18,
                max_tokens=_llm_max_completion_tokens(),
                json_object=True,
                reasoning_effort=os.getenv("TODAY_INSIGHT_REASONING_EFFORT", "low"),
            )
        )
    return _llm


_TODAY_INSIGHT_PROMPT = """\
당신은 SK AX CEO/임원 홈 대시보드 Today's Insight의 **신호·대응 방향** 작성자입니다.

서버 후처리로 이미 채워지는 필드 (품질에 영향 없음, 빈 문자열 가능):
- headline, executive_summary, change_summary

당신이 집중할 핵심 산출물:
1. signals 3개 — label 순서: "주요 신호", "관찰 포인트", "다음 판단"
2. response_direction — 실행 산출물/판단 기준이 보이는 액션 1~3개
3. executive_implication — SK AX 임원 관점 시사점 (일반론 금지)
4. sources — 입력 id/url/title 만

판단 순서:
- comparison_facts.primary_selection(확실한 이벤트) → structural/keyword_trends 맥락
  → SK AX 고객 대응·운영 KPI → 오늘 확인 항목.
- "오늘"은 이슈/카드 생성일이 아니라 연결된 원문 published_at 의 KST 날짜가
  report_date 와 같은 입력만 current 로 봅니다.
- Peer 모니터링이므로 SK AX 자체 뉴스/공식자료/자사 기사 원문은 current 입력에서
  제외하고, SK AX 자료는 해석 관점으로만 사용합니다.
- "주요 포인트"와 "관찰 포인트"를 별도 항목으로 나누지 말고 signal[0]의
  "주요 신호" 안에 핵심 사건과 관찰 맥락을 함께 담습니다.
- label=low_visibility_definite_event 이면 signal[0]에 반드시 반영.
- structural·keyword_trends는 primary를 대체하지 않는 보조 맥락.
- 뉴스/카드/이슈 건수 또는 노출량만으로 판단하지 않습니다. 건수는 맥락으로만 쓰고,
  실제 사건·내용·변화 방향·SK AX 고객 대응 관점과 함께 해석합니다.
- "중요", "긴급", "우선순위"를 임의로 단정하지 않습니다. 입력에 있는 중요도/확실성
  근거가 없으면 "확인 필요", "점검 대상"처럼 표현합니다.

reasoning step label: "관찰", "비교", "의미", "판단" 만 사용.
각 reasoning.detail 은 1문장 이내로 짧게 씁니다.

절대 규칙:
1. comparison_facts 에 없는 수치를 만들지 않습니다.
2. "경쟁 환경", "전략 강화", "시장 확대", "제안서 작성" 같은 넓은 결론만 단독으로 쓰지 않습니다.
3. signal[0]은 primary_selection.items[0]을 우선 반영합니다.
4. evidence.changes에는 comparison_facts 에 확인 가능한 항목만 씁니다.
5. 사용자에게 보이는 문장에는 DB 테이블명, 컬럼명, 내부 id, source id, raw id,
   IC-/CN-/raw- 같은 식별자를 쓰지 않습니다. sources 배열의 id 필드에만 식별자를 둡니다.
6. 사용자에게 보이는 문장에는 salience_score, exposure_score, visibility_gap,
   importance_score 같은 내부 점수명이나 0.x 원점수를 쓰지 않습니다.
   "내용 영향은 큰 편", "보도 확산은 아직 낮은 편", "확산 전 신호"처럼 경향으로 표현합니다.

입력 JSON:
{context_json}

출력 JSON schema:
{{
  "headline": "",
  "executive_summary": "",
  "executive_implication": "SK AX 임원 관점의 시사점 2문장 이내",
  "change_summary": [],
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

        llm_context, llm_context_meta = _fit_llm_prompt_context(context)
        context["llm_context_meta"] = llm_context_meta
        prompt = _TODAY_INSIGHT_PROMPT.replace(
            "{context_json}",
            _json_dumps(llm_context),
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
    issues = [row for row in issues if not _is_self_company_issue(row)]
    current_issues = [
        row for row in issues if _is_anchor_current_issue(row, anchor_date=anchor_date)
    ][: req.max_issues]
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
    cards = [card for card in cards if not _is_self_company_card(card)]
    if len(cards) < req.max_cards:
        supplemental_cards = _fetch_anchor_date_cards(
            anchor_date=anchor_date,
            limit=req.max_cards - len(cards),
            exclude_ids=[
                str(card.get("id")) for card in cards if str(card.get("id") or "").strip()
            ],
        )
        supplemental_cards = [
            card for card in supplemental_cards if not _is_self_company_card(card)
        ]
        cards = [*cards, *supplemental_cards][: req.max_cards]
    if not cards:
        cards = _fetch_recent_cards(
            anchor_date=anchor_date,
            window_days=req.window_days,
            limit=req.max_cards,
        )
        cards = [card for card in cards if not _is_self_company_card(card)][: req.max_cards]
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
    sources = _collect_sources(
        current_issues=current_issues,
        cards=cards,
        anchor_date=anchor_date,
        limit=12,
    )
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
        "current_issues": [_issue_for_prompt(row) for row in current_issues],
        "history_issues": [_issue_for_prompt(row) for row in history_issues],
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


def _is_anchor_current_issue(issue: Mapping[str, Any], *, anchor_date: date) -> bool:
    """Treat backfilled issue rows as current only when their source date is current."""
    anchor_iso = anchor_date.isoformat()
    if issue.get("has_anchor_source") is True:
        return True
    latest_source_date = str(issue.get("latest_source_date_kst") or "").strip()
    if latest_source_date:
        return latest_source_date[:10] == anchor_iso
    return str(issue.get("created_date_kst") or "").strip()[:10] == anchor_iso


def _fetch_integrated_issues(
    *, anchor_date: date, window_days: int, limit: int
) -> list[dict[str, Any]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT ii.id::text AS id,
                           ii.cluster_id,
                           ii.main_company,
                           ii.event_type,
                           ii.source_family,
                           ii.confidence,
                           ii.headline,
                           ii.one_line_summary,
                           ii.analyzed_source_ids,
                           ii.source_ids,
                           ii.sectors,
                           ii.mentioned_peer_companies,
                           ii.content_summary,
                           ii.issue_frame,
                           ii.sources,
                           ii.evidence,
                           ii.quality,
                           ii.payload,
                           ii.created_at,
                           ii.updated_at,
                           ((ii.created_at AT TIME ZONE 'Asia/Seoul')::date)::text
                               AS created_date_kst,
                           (
                               SELECT MAX(source_date)::text
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                           ) AS latest_source_date_kst,
                           EXISTS (
                               SELECT 1
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                                WHERE source_date = CAST(:anchor_date AS date)
                           ) AS has_anchor_source
                      FROM integrated_issues ii
                     WHERE ii.is_current = TRUE
                       AND ii.status = 'active'
                       AND ii.is_valid = TRUE
                       AND (ii.created_at AT TIME ZONE 'Asia/Seoul')::date
                           >= CAST(:anchor_date AS date) - (:window_days * INTERVAL '1 day')
                     ORDER BY
                       CASE
                         WHEN EXISTS (
                               SELECT 1
                                 FROM (
                                       SELECT (
                                           COALESCE(
                                               iisa.published_at,
                                               ra.published_at,
                                               ra.created_at
                                           ) AT TIME ZONE 'Asia/Seoul'
                                       )::date AS source_date
                                         FROM integrated_issue_source_articles iisa
                                         LEFT JOIN raw_articles ra ON ra.id = iisa.raw_article_id
                                        WHERE iisa.integrated_issue_id = ii.id
                                       UNION ALL
                                       SELECT (COALESCE(ra.published_at, ra.created_at)
                                           AT TIME ZONE 'Asia/Seoul')::date AS source_date
                                         FROM raw_articles ra
                                        WHERE ra.id = ii.representative_raw_article_id
                                           OR ra.id = ANY(ii.source_ids)
                                           OR ra.id = ANY(ii.analyzed_source_ids)
                                      ) issue_source_dates
                                WHERE source_date = CAST(:anchor_date AS date)
                           )
                         THEN 0 ELSE 1
                       END,
                       ii.confidence DESC NULLS LAST,
                       ii.created_at DESC
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
    window_days: int | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    del window_days
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
                           created_at,
                           ((created_at AT TIME ZONE 'Asia/Seoul')::date)::text AS created_date_kst
                     FROM card_news
                     WHERE integrated_issue_id IN ({placeholders})
                       AND (created_at AT TIME ZONE 'Asia/Seoul')::date = CAST(:anchor_date AS date)
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date = CAST(:anchor_date AS date)
                           )
                       )
                     ORDER BY importance_score DESC NULLS LAST, created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {
                        **params,
                        "anchor_date": anchor_date.isoformat(),
                    },
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("today insight card lookup failed | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _fetch_anchor_date_cards(
    *,
    anchor_date: date,
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
                           created_at,
                           ((created_at AT TIME ZONE 'Asia/Seoul')::date)::text AS created_date_kst
                      FROM card_news
                     WHERE (created_at AT TIME ZONE 'Asia/Seoul')::date = CAST(:anchor_date AS date)
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date = CAST(:anchor_date AS date)
                           )
                       )
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
        log.warning("today insight anchor-date card lookup failed | error=%s", exc)
        return []
    return [_json_ready(dict(row)) for row in rows]


def _fetch_recent_cards(
    *,
    anchor_date: date,
    window_days: int,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
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
                           created_at,
                           ((created_at AT TIME ZONE 'Asia/Seoul')::date)::text
                               AS created_date_kst
                      FROM card_news
                     WHERE (created_at AT TIME ZONE 'Asia/Seoul')::date
                           BETWEEN CAST(:anchor_date AS date)
                               - (:window_days * INTERVAL '1 day')
                               AND CAST(:anchor_date AS date)
                       AND COALESCE(peer_company_id, company, '') <> 'sk_ax'
                       AND (
                           COALESCE(cardinality(source_raw_article_ids), 0) = 0
                           OR EXISTS (
                               SELECT 1
                                 FROM raw_articles ra
                                WHERE ra.id = ANY(source_raw_article_ids)
                                  AND (
                                      COALESCE(ra.published_at, ra.created_at)
                                      AT TIME ZONE 'Asia/Seoul'
                                  )::date BETWEEN CAST(:anchor_date AS date)
                                      - (:window_days * INTERVAL '1 day')
                                      AND CAST(:anchor_date AS date)
                           )
                       )
                     ORDER BY importance_score DESC NULLS LAST, created_at DESC
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
    """Return the newest active report on or before ``anchor_date``.

    Home can show the latest saved report, but marks it as a fallback when it is
    older than the selected anchor date.
    """
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                    SELECT report_date::text AS report_date,
                           output_payload,
                           created_at
                      FROM today_insight_reports
                     WHERE report_date <= CAST(:anchor_date AS date)
                       AND status = 'active'
                     ORDER BY report_date DESC, created_at DESC
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
    report_date_value = str(row.get("report_date") or "")
    payload = _json_ready(payload)
    payload.setdefault("report_date", report_date_value)
    provenance = payload.setdefault("provenance", {})
    if isinstance(provenance, dict):
        provenance["cache_lookup"] = "latest_saved_on_or_before_anchor"
        provenance["served_anchor_date"] = anchor_date.isoformat()
        provenance["cached_report_date"] = report_date_value
        if report_date_value and report_date_value != anchor_date.isoformat():
            provenance["latest_fallback"] = True
    return {
        "payload": payload,
        "created_at": row.get("created_at"),
        "report_date": report_date_value,
    }


def _preload_llm_client() -> None:
    try:
        _get_llm()
    except Exception as exc:  # noqa: BLE001
        log.debug("TodayInsight LLM client preload skipped | error=%s", exc)


def _scheduled_cache_pending_result(anchor_date: date) -> dict[str, Any]:
    now_iso = datetime.now(UTC).isoformat()
    prior_reports = _fetch_prior_today_reports(anchor_date=anchor_date, limit=1)
    latest_prior = prior_reports[0] if prior_reports else {}
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
                    "label": "관찰 포인트",
                    "value": "저장된 오늘 인사이트가 아직 없습니다",
                    "reasoning": [
                        {
                            "stage": "관찰",
                            "detail": "저장된 오늘 분석 결과가 아직 확인되지 않았습니다.",
                        },
                        {
                            "stage": "판단",
                            "detail": (
                                "홈 조회에서는 신규 생성하지 않고 08:10 스케줄 결과를 기다립니다."
                            ),
                        },
                    ],
                    "evidence": {
                        "grounds": ["저장된 오늘 분석 결과 없음"],
                        "changes": ["실제 변화 분석 전 상태"],
                        "related_keywords": ["스케줄 업데이트", "캐시 대기"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
                {
                    "id": "scheduled-cache-next",
                    "label": "시사점",
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
                        "changes": ["카드 상태 기반 일중 재생성 예외 제거"],
                        "related_keywords": ["daily cache", "executive insight"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
                {
                    "id": "scheduled-cache-response",
                    "label": "대응방향",
                    "value": "스케줄 완료 뒤 출처 포함 결과를 확인",
                    "reasoning": [
                        {
                            "stage": "관찰",
                            "detail": "현재 응답은 생성 결과가 아닌 상태 안내입니다.",
                        },
                        {
                            "stage": "판단",
                            "detail": "저장 완료 이후 실제 근거 기반 인사이트를 사용해야 합니다.",
                        },
                    ],
                    "evidence": {
                        "grounds": ["스케줄 생성 대기 상태"],
                        "changes": ["출처 포함 결과 노출 전"],
                        "related_keywords": ["출처 확인", "저장 결과"],
                        "source_ids": ["scheduled-cache"],
                    },
                },
            ],
            "response_direction": [
                {
                    "action": "08:10 스케줄러가 오늘 분석 결과를 저장했는지 확인합니다.",
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
                "result_kind": "scheduled_pending",
                "is_status_placeholder": True,
                "is_fixture": False,
                "update_policy": "daily_0810_kst",
                "prompt_version": _PROMPT_VERSION,
                "latest_available_report_date": latest_prior.get("report_date"),
                "latest_available_headline": latest_prior.get("headline"),
            },
            "warning": "Today's Insight 생성 결과가 아직 없어 스케줄 대기 상태를 표시합니다.",
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
    matched = [
        source
        for source in sources
        if str(source.get("id") or "") in source_ids or str(source.get("url") or "") in source_ids
    ]
    matched_url_sources = [
        source
        for source in matched
        if str(source.get("url") or "").startswith(("http://", "https://"))
    ]
    return matched_url_sources or matched or url_sources[:2] or sources[:2]


def _match_trace_by_ids(
    trace: list[dict[str, Any]],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    if not source_ids:
        return trace[:6]
    matched = []
    for row in trace:
        values = {
            str(row.get("source_integrated_issue_id") or ""),
            str(row.get("source_card_id") or ""),
            *[str(raw_id) for raw_id in _list(row.get("source_raw_article_ids"))],
        }
        if values.intersection(source_ids):
            matched.append(row)
    return matched or trace[:3]


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


def _is_before_window(raw_date: str, window_start: date) -> bool:
    if not raw_date:
        return False
    try:
        return date.fromisoformat(raw_date[:10]) < window_start
    except ValueError:
        return False


def _fallback_result(
    *,
    anchor_date: date,
    context: dict[str, Any],
    warning: str | None = None,
) -> dict[str, Any]:
    no_current_signals = warning == "today insight source data unavailable"
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
            "provenance": {
                "mode": "deterministic_fallback",
                "result_kind": "no_current_signals" if no_current_signals else "generated_fallback",
                "is_status_placeholder": no_current_signals,
                "is_fixture": False,
                "warning": warning or "",
            },
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
        "latest_source_date_kst": row.get("latest_source_date_kst"),
        "has_anchor_source": row.get("has_anchor_source"),
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
        "created_date_kst": card.get("created_date_kst"),
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
    anchor_date: date,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    raw_sources = _fetch_raw_article_sources(
        _source_raw_ids({"recent_cards": cards}),
        anchor_date=anchor_date,
    )
    for source in raw_sources:
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
        if len(out) >= limit:
            return out[:limit]

        for source in _compact_sources(card.get("sources"), limit=3):
            key = str(source.get("url") or source.get("id") or source.get("title") or "")
            if key and key not in seen:
                seen.add(key)
                out.append(source)
        if len(out) >= limit:
            return out[:limit]

    for issue in current_issues:
        for source in _compact_sources(issue.get("sources"), limit=4):
            key = str(source.get("url") or source.get("id") or source.get("title") or "")
            if key and key not in seen:
                seen.add(key)
                out.append(source)
        if len(out) >= limit:
            return out[:limit]

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
        if len(out) >= limit:
            return out[:limit]
    return out[:limit]


def _fetch_raw_article_sources(raw_ids: list[int], *, anchor_date: date) -> list[dict[str, Any]]:
    if not raw_ids:
        return []
    clean_ids = _dedupe([str(raw_id) for raw_id in raw_ids if raw_id], limit=100)
    if not clean_ids:
        return []
    placeholders = ", ".join(f":raw_{idx}" for idx in range(len(clean_ids)))
    params = {f"raw_{idx}": int(raw_id) for idx, raw_id in enumerate(clean_ids)}
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        f"""
                    SELECT id,
                           title,
                           source_name,
                           company,
                           content,
                           url,
                           published_at
                      FROM raw_articles
                     WHERE id IN ({placeholders})
                       AND (COALESCE(published_at, created_at) AT TIME ZONE 'Asia/Seoul')::date
                           = CAST(:anchor_date AS date)
                     ORDER BY array_position(ARRAY[{placeholders}]::bigint[], id)
                    """
                    ),
                    {**params, "anchor_date": anchor_date.isoformat()},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("today insight raw article source lookup skipped | error=%s", exc)
        return []

    sources: list[dict[str, Any]] = []
    for row in rows:
        item = _json_ready(dict(row))
        if _is_self_company_source(item):
            continue
        related_companies = _related_company_labels_from_source(item)
        sources.append(
            {
                "id": f"raw-{item.get('id')}",
                "title": item.get("title") or "",
                "source_name": item.get("source_name") or "",
                "publisher": item.get("source_name") or "",
                "related_companies": related_companies,
                "url": item.get("url") or "",
                "published_at": str(item.get("published_at") or "") or None,
            }
        )
    return sources


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


def _is_self_company_issue(issue: dict[str, Any]) -> bool:
    if _is_self_company_id(issue.get("main_company")):
        return True
    if any(
        _is_self_company_id(company) for company in _list(issue.get("mentioned_peer_companies"))
    ):
        return True
    source_texts = [
        str(source.get("title") or "")
        for source in _list(issue.get("sources"))
        if isinstance(source, dict)
    ]
    return _contains_self_company_reference(
        issue.get("headline"),
        issue.get("one_line_summary"),
        issue.get("content_summary"),
        *source_texts,
    )


def _is_self_company_card(card: dict[str, Any]) -> bool:
    if _is_self_company_id(card.get("peer_id")):
        return True
    source_texts = [
        str(source.get("title") or "")
        for source in _list(card.get("sources"))
        if isinstance(source, dict)
    ]
    return _contains_self_company_reference(
        card.get("title"),
        " ".join(str(line) for line in _list(card.get("summary_lines"))),
        *source_texts,
    )


def _is_self_company_source(source: dict[str, Any]) -> bool:
    if any(_is_self_company_id(company) for company in _list(source.get("company"))):
        return True
    return _contains_self_company_reference(source.get("title"), source.get("content"))


def _is_self_company_id(value: Any) -> bool:
    canonical_id = normalize_to_canonical_id(str(value or "")) or str(value or "")
    return canonical_id in SELF_COMPANY_IDS


def _contains_self_company_reference(*values: Any) -> bool:
    normalized_text = " ".join(str(value or "") for value in values)
    normalized_text = normalized_text.lower().replace(" ", "")
    if not normalized_text:
        return False
    for company_id in SELF_COMPANY_IDS:
        aliases = PEER_ID_ALIASES.get(company_id, [company_id])
        for alias in aliases:
            normalized_alias = alias.lower().replace(" ", "")
            if normalized_alias and normalized_alias in normalized_text:
                return True
    return False


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


def _slim_issue_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "created_date_kst": row.get("created_date_kst"),
        "main_company": row.get("main_company"),
        "company_label": row.get("company_label"),
        "event_type": row.get("event_type"),
        "source_family": row.get("source_family"),
        "sectors": _list(row.get("sectors"))[:4],
        "confidence": row.get("confidence"),
        "headline": _clip(str(row.get("headline") or ""), 200),
        "one_line_summary": _clip(str(row.get("one_line_summary") or ""), 220),
        "content_summary": _clip(str(row.get("content_summary") or ""), 360),
        "source_ids": _list(row.get("source_ids"))[:4],
    }


def _slim_card_for_llm(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "integrated_issue_id": card.get("integrated_issue_id"),
        "peer_id": card.get("peer_id"),
        "title": _clip(str(card.get("title") or ""), 200),
        "summary_lines": [_clip(str(line), 120) for line in _list(card.get("summary_lines"))[:3]],
        "event_type": card.get("event_type"),
        "sector": card.get("sector"),
        "importance_score": card.get("importance_score"),
        "skax_implication": _clip(str(card.get("skax_implication") or ""), 220),
    }


def _slim_history_issue_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "main_company": row.get("main_company"),
        "company_label": row.get("company_label"),
        "event_type": row.get("event_type"),
        "headline": _clip(str(row.get("headline") or ""), 160),
        "one_line_summary": _clip(str(row.get("one_line_summary") or ""), 180),
    }


def _slim_ledger_item_for_llm(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "peer_id": row.get("peer_id"),
        "event_type": row.get("event_type"),
        "headline": _clip(str(row.get("headline") or row.get("title") or ""), 160),
        "summary": _clip(str(row.get("summary") or row.get("one_line_summary") or ""), 200),
    }


def _slim_profile_context_for_llm(profile_context: dict[str, Any]) -> dict[str, Any]:
    peers = profile_context.get("peer_profiles")
    slim_peers: dict[str, Any] = {}
    if isinstance(peers, dict):
        for peer_id, profile in list(peers.items())[:6]:
            if not isinstance(profile, dict):
                continue
            recent_signals = []
            for signal in _list(
                profile.get("recent_signals") or profile.get("recent_business_signals")
            )[:2]:
                if isinstance(signal, dict):
                    recent_signals.append(
                        _clip(str(signal.get("headline") or signal.get("signal") or ""), 120)
                    )
                elif signal:
                    recent_signals.append(_clip(str(signal), 120))
            slim_peers[str(peer_id)] = {
                "company_name_ko": profile.get("company_name_ko") or profile.get("name_ko"),
                "business_areas": _list(profile.get("business_areas"))[:3],
                "strategic_direction": _clip(
                    str(
                        profile.get("strategic_direction")
                        or profile.get("direction_summary")
                        or profile.get("executive_summary")
                        or ""
                    ),
                    260,
                ),
                "recent_signals": recent_signals,
            }

    sector_context = profile_context.get("sector_context")
    slim_sectors: dict[str, Any] = {}
    if isinstance(sector_context, dict):
        for sector, payload in list(sector_context.items())[:4]:
            if isinstance(payload, dict):
                slim_sectors[str(sector)] = {
                    "summary": _clip(
                        str(payload.get("summary") or payload.get("headline") or ""), 180
                    )
                }
    return {"peer_profiles": slim_peers, "sector_context": slim_sectors}


def _slim_skax_context_for_llm(skax_context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(skax_context, dict):
        return {}
    slim: dict[str, Any] = {}
    for sector, payload in list(skax_context.items())[:4]:
        if not isinstance(payload, dict):
            continue
        documents = []
        for doc in _list(payload.get("documents") or payload.get("newsroom_documents"))[:2]:
            if isinstance(doc, dict):
                documents.append(
                    {
                        "title": _clip(str(doc.get("title") or ""), 120),
                        "published_at": doc.get("published_at"),
                    }
                )
        slim[str(sector)] = {
            "summary": _clip(str(payload.get("summary") or payload.get("headline") or ""), 200),
            "documents": documents,
        }
    return slim


def _build_llm_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    comparison = context.get("comparison_facts")
    raw_profile = context.get("profile_context")
    profile_context: dict[str, Any] = raw_profile if isinstance(raw_profile, dict) else {}
    raw_skax = context.get("skax_context")
    skax_context: dict[str, Any] = raw_skax if isinstance(raw_skax, dict) else {}
    return {
        "report_date": context.get("report_date"),
        "window_days": context.get("window_days"),
        "generation_focus": {
            "server_filled_fields": ["headline", "executive_summary", "change_summary"],
            "llm_priority_fields": [
                "signals",
                "response_direction",
                "executive_implication",
                "sources",
            ],
        },
        "comparison_facts": trim_comparison_facts_for_prompt(
            comparison if isinstance(comparison, dict) else None
        ),
        "change_stats": context.get("change_stats"),
        "current_issues": [
            _slim_issue_for_llm(row)
            for row in _list(context.get("current_issues"))
            if isinstance(row, dict)
        ],
        "recent_cards": [
            _slim_card_for_llm(card)
            for card in _list(context.get("recent_cards"))
            if isinstance(card, dict)
        ],
        "history_issues": [
            _slim_history_issue_for_llm(row)
            for row in _list(context.get("history_issues"))[:6]
            if isinstance(row, dict)
        ],
        "sources": _list(context.get("sources"))[:10],
        "prior_today_insight_memory": _list(context.get("prior_today_insight_memory"))[:3],
        "analysis_ledger_context": [
            _slim_ledger_item_for_llm(row)
            for row in _list(context.get("analysis_ledger_context"))[:4]
            if isinstance(row, dict)
        ],
        "profile_context": _slim_profile_context_for_llm(profile_context),
        "skax_context": _slim_skax_context_for_llm(skax_context),
    }


def _fit_llm_prompt_context(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    working = _build_llm_prompt_context(context)
    meta: dict[str, Any] = {
        "max_chars": _LLM_CONTEXT_MAX_CHARS,
        "serialized_chars": len(_json_dumps(working)),
        "truncated": False,
        "dropped_sections": [],
    }
    if meta["serialized_chars"] <= _LLM_CONTEXT_MAX_CHARS:
        return working, meta

    for section in _LLM_CONTEXT_DROP_ORDER:
        if section not in working:
            continue
        if section in {"profile_context", "skax_context"}:
            working[section] = {}
        else:
            working[section] = []
        meta["dropped_sections"].append(section)
        meta["serialized_chars"] = len(_json_dumps(working))
        if meta["serialized_chars"] <= _LLM_CONTEXT_MAX_CHARS:
            return working, meta

    meta["truncated"] = True
    shrink_targets = ("recent_cards", "current_issues", "sources")
    while meta["serialized_chars"] > _LLM_CONTEXT_MAX_CHARS:
        reduced = False
        for target in shrink_targets:
            rows = working.get(target)
            if isinstance(rows, list) and len(rows) > 3:
                working[target] = rows[:-1]
                reduced = True
                break
        if not reduced:
            break
        meta["serialized_chars"] = len(_json_dumps(working))
    return working, meta


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
