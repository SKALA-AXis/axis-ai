# 작성일: 2026-06-05
# 작성자: 박진
# 변경이력:
#   2026-06-05 박진 — 투데이 인사이트 에이전트 및 60일 카드뉴스 입력 신규 추가
#   2026-06-09 최종민 — dual-lane 비교
#   2026-06-14 안가은 — 키워드 트렌드 파이프라인 갱신, 신호를 출처일에 고정
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
from collections import Counter
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Mapping
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.agents.today_insight._config import _LLM_MODEL, _PROMPT_VERSION
from src.agents.today_insight.output_schema import (  # noqa: F401
    _SIGNAL_LABELS,
    _apply_insight_state,
    _build_coverage_stats,
    _build_memory_document,
    _build_week_synthesis,
    _derive_insight_state,
    _match_sources_by_ids,
    _match_trace_by_ids,
    _merge_comparison_evidence,
    _norm_id_set,
    _normalize_actions,
    _normalize_change_summary,
    _normalize_insight_sections,
    _normalize_result,
    _normalize_signals,
    _normalize_source_trace,
    _normalize_sources,
    _source_trace_from_context,
)
from src.agents.today_insight.text_processing import (  # noqa: F401
    _PUBLIC_TEXT_REPLACEMENTS,
    _WEAK_OR_MOCK_MARKERS,
    _clamp_float,
    _clip,
    _compact_evidence,
    _compact_sources,
    _company_label,
    _counter_top,
    _dedupe,
    _dedupe_int,
    _default_changes,
    _default_grounds,
    _default_source_ids,
    _delta_rows,
    _fallback_actions,
    _fallback_headline,
    _fallback_implication,
    _fallback_signals,
    _fallback_summary,
    _first_card,
    _first_issue,
    _int_list,
    _is_before_window,
    _is_domestic_card,
    _is_domestic_company_id,
    _is_domestic_issue,
    _is_weak_or_mock_text,
    _json_dumps,
    _json_ready,
    _keywords_from_context,
    _list,
    _proposal_axis_from_stats,
    _qualify_internal_score_text,
    _related_company_labels_from_source,
    _sanitize_public_text,
    _section_actions,
    _slug,
    _string_list,
    _top_axis,
    _top_axis_from_stats,
)
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
    trim_comparison_facts_for_prompt,
)

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
_LLM_CONTEXT_MAX_CHARS = 48_000
_LLM_CONTEXT_DROP_ORDER = (
    "analysis_ledger_context",
    "prior_today_insight_memory",
    "history_issues",
    "skax_context",
    "profile_context",
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
        # anchor_date(오늘) 기준 current 신호(현재 이슈 또는 당일 카드)가 없으면,
        # 과거 카드로 active 리포트를 만들지 않고 '신규 신호 없음' 플레이스홀더를 반환한다.
        # (recent_cards 폴백이 과거 카드를 채워도 그것을 오늘 헤드라인으로 내보내지 않음.)
        if not context.get("has_current_signal", True):
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
    issues = [row for row in issues if _is_domestic_issue(row) and not _is_self_company_issue(row)]
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
    cards = [card for card in cards if _is_domestic_card(card) and not _is_self_company_card(card)]
    if len(cards) < req.max_cards:
        supplemental_cards = _fetch_anchor_date_cards(
            anchor_date=anchor_date,
            limit=req.max_cards - len(cards),
            exclude_ids=[
                str(card.get("id")) for card in cards if str(card.get("id") or "").strip()
            ],
        )
        supplemental_cards = [
            card
            for card in supplemental_cards
            if _is_domestic_card(card) and not _is_self_company_card(card)
        ]
        cards = [*cards, *supplemental_cards][: req.max_cards]
    # anchor_date(오늘) 기준 current 카드(현재 이슈 카드 또는 당일 카드) 존재 여부 —
    # 아래 _fetch_recent_cards 폴백(과거 카드)으로 채워지기 전에 확정해 둔다.
    has_anchor_cards = bool(cards)
    if not cards:
        cards = _fetch_recent_cards(
            anchor_date=anchor_date,
            window_days=req.window_days,
            limit=req.max_cards,
        )
        cards = [
            card for card in cards if _is_domestic_card(card) and not _is_self_company_card(card)
        ][: req.max_cards]
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
        "has_current_signal": bool(current_issues) or has_anchor_cards,
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
                       AND (
                           NOT EXISTS (
                               SELECT 1
                                 FROM card_news cn_any
                                WHERE cn_any.integrated_issue_id = ii.id
                           )
                           OR EXISTS (
                               SELECT 1
                                 FROM card_news cn_active
                                WHERE cn_active.integrated_issue_id = ii.id
                                  AND LOWER(cn_active.status) = 'active'
                           )
                       )
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
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                     FROM card_news
                     LEFT JOIN LATERAL (
                         SELECT MIN((
                                    COALESCE(ra.published_at, ra.created_at)
                                    AT TIME ZONE 'Asia/Seoul'
                                )::date) AS earliest_source_date
                           FROM raw_articles ra
                          WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{{}}'::bigint[]))
                     ) src ON TRUE
                     WHERE integrated_issue_id IN ({placeholders})
                       AND LOWER(status) = 'active'
                       AND COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           ) = CAST(:anchor_date AS date)
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
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
                       created_at DESC
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
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                      FROM card_news
                      LEFT JOIN LATERAL (
                          SELECT MIN((
                                     COALESCE(ra.published_at, ra.created_at)
                                     AT TIME ZONE 'Asia/Seoul'
                                 )::date) AS earliest_source_date
                            FROM raw_articles ra
                           WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{{}}'::bigint[]))
                      ) src ON TRUE
                     WHERE COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           ) = CAST(:anchor_date AS date)
                       AND LOWER(status) = 'active'
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
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
                       created_at DESC
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
                           COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )::text AS created_date_kst
                      FROM card_news
                      LEFT JOIN LATERAL (
                          SELECT MIN((
                                     COALESCE(ra.published_at, ra.created_at)
                                     AT TIME ZONE 'Asia/Seoul'
                                 )::date) AS earliest_source_date
                            FROM raw_articles ra
                           WHERE ra.id = ANY(COALESCE(source_raw_article_ids, '{}'::bigint[]))
                      ) src ON TRUE
                     WHERE COALESCE(
                               src.earliest_source_date,
                               (created_at AT TIME ZONE 'Asia/Seoul')::date
                           )
                           BETWEEN CAST(:anchor_date AS date)
                               - (:window_days * INTERVAL '1 day')
                               AND CAST(:anchor_date AS date)
                       AND LOWER(status) = 'active'
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
                     ORDER BY
                       COALESCE(
                           src.earliest_source_date,
                           (created_at AT TIME ZONE 'Asia/Seoul')::date
                       ) DESC,
                       importance_score DESC NULLS LAST,
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


# === 홈 3상태(today_signal/recent_signal/quiet) + 모니터링 커버리지 ===
# PR #212 의 has_current_signal/has_anchor_cards 게이트는 그대로 두고, 결과 정규화
# 단계에서 state/signal_date/week_synthesis/coverage_stats 만 얹는다.
# (recent_signal: 최근 2~3 영업일 윈도우는 후속 단계에서 추가.)


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
