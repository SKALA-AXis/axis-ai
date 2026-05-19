"""KnowledgeCuration read-model writeback middleware.

분석 agent (InsightCascade / MixerAnalysis / PeerComparison / GlobalTrends /
BriefingGeneration) 의 결론을 V30 최소 스키마의 화면별 read model에 저장한다.

design: ``axis-ai/design/25-knowledge-curation/analysis-ledger.md``.

핵심 entry point:

    ``AnalysisLedger.insert(input)`` — 단일 분석 결과 INSERT.

    ``with_ledger_writeback(agent_class)`` — 분석 agent 의 async method 를 wrapping
    하는 데코레이터. 정상 종료 시 결과 dict 의 표준 필드를 자동 추출하여 INSERT.
    실패 시 분석 결과는 그대로 반환 (fail-soft) + warning log.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, Iterable, Mapping, TypedDict

from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

_DEFAULT_MIN_CONF = float(os.getenv("LEDGER_MIN_CONFIDENCE_FOR_PACK", "0.7"))


def _git_sha() -> str:
    sha = os.getenv("AXIS_GIT_SHA")
    if sha:
        return sha[:12]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


_GIT_SHA_CACHE = _git_sha()


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────


class LedgerInsertInput(TypedDict, total=False):
    analysis_type: str  # "insight" | "mixer" | "peer" | "global" | "briefing"
    analysis_id: str
    peer_ids: list[str]
    conclusion_one_liner: str
    strategy_label: str | None
    confidence: float
    source_card_ids: list[str]
    sk_ax_implication: str | None
    langfuse_trace_id: str | None
    prompt_version: str
    git_sha: str


class LedgerInsertResult(TypedDict):
    ledger_id: int
    inserted: bool
    superseded_ledger_ids: list[int]
    included_in_pack: bool


# ──────────────────────────────────────────────────────────────────────────
# Core insert
# ──────────────────────────────────────────────────────────────────────────


class AnalysisLedger:
    """V30 read-model write-through helper."""

    @classmethod
    def insert(cls, payload: LedgerInsertInput) -> LedgerInsertResult:
        threshold = _DEFAULT_MIN_CONF
        confidence = float(payload.get("confidence", 0.0))
        included = confidence >= threshold

        peer_ids = list(payload.get("peer_ids") or [])
        conclusion = (payload.get("conclusion_one_liner") or "").strip()

        if not peer_ids:
            log.warning(
                "AnalysisLedger.insert | peer_ids 비어있음 — INSERT 진행 (cross-peer mixer 등)"
            )
        if not conclusion:
            log.warning("AnalysisLedger.insert | conclusion_one_liner 비어있음 — fail-soft skip")
            return {
                "ledger_id": -1,
                "inserted": False,
                "superseded_ledger_ids": [],
                "included_in_pack": False,
            }

        return cls._insert_read_model(payload, included)

    @classmethod
    def _insert_read_model(
        cls,
        payload: LedgerInsertInput,
        included: bool,
    ) -> LedgerInsertResult:
        analysis_type = (payload.get("analysis_type") or "unknown").lower()
        analysis_id = payload.get("analysis_id") or f"{analysis_type}-{datetime.now(timezone.utc).isoformat()}"
        peer_ids = list(payload.get("peer_ids") or [])
        source_card_ids = list(payload.get("source_card_ids") or [])
        conclusion = payload.get("conclusion_one_liner") or ""
        confidence = float(payload.get("confidence", 0.0))
        body = {
            **payload,
            "included_in_pack": included,
            "created_by": "analysis_ledger_compat",
        }

        try:
            with SessionLocal() as db:
                if analysis_type in {"mixer", "mixeranalysis"}:
                    db.execute(
                        text("""
                            INSERT INTO mixer_results (
                                source_analysis_id, title, input_peer_ids, input_card_ids,
                                generated_implication, sk_ax_implication, final_one_liner,
                                confidence, payload
                            ) VALUES (
                                :analysis_id, :title, CAST(:peer_ids AS text[]),
                                CAST(:source_card_ids AS text[]),
                                CAST(:generated_implication AS jsonb),
                                :sk_ax_implication, :final_one_liner,
                                :confidence, CAST(:payload AS jsonb)
                            )
                            ON CONFLICT (source_analysis_id)
                            WHERE source_analysis_id IS NOT NULL DO UPDATE SET
                                generated_implication = EXCLUDED.generated_implication,
                                sk_ax_implication = EXCLUDED.sk_ax_implication,
                                final_one_liner = EXCLUDED.final_one_liner,
                                confidence = EXCLUDED.confidence,
                                payload = EXCLUDED.payload,
                                updated_at = NOW()
                        """),
                        {
                            "analysis_id": analysis_id,
                            "title": conclusion,
                            "peer_ids": _pg_text_array(peer_ids),
                            "source_card_ids": _pg_text_array(source_card_ids),
                            "generated_implication": json.dumps(body, ensure_ascii=False),
                            "sk_ax_implication": payload.get("sk_ax_implication"),
                            "final_one_liner": conclusion,
                            "confidence": confidence,
                            "payload": json.dumps(body, ensure_ascii=False),
                        },
                    )
                elif analysis_type in {"insight", "insightcascade"}:
                    db.execute(
                        text("""
                            INSERT INTO insight_reports (
                                source_analysis_id, title, insight_type, status,
                                focus_peer_ids, focus_card_ids, summary, final_one_liner,
                                sk_ax_implication, source_card_ids, confidence, payload
                            ) VALUES (
                                :analysis_id, :title, 'cascade', 'completed',
                                CAST(:peer_ids AS text[]), CAST(:source_card_ids AS text[]),
                                :summary, :final_one_liner, :sk_ax_implication,
                                CAST(:source_card_ids AS text[]), :confidence,
                                CAST(:payload AS jsonb)
                            )
                            ON CONFLICT (source_analysis_id)
                            WHERE source_analysis_id IS NOT NULL DO UPDATE SET
                                summary = EXCLUDED.summary,
                                final_one_liner = EXCLUDED.final_one_liner,
                                sk_ax_implication = EXCLUDED.sk_ax_implication,
                                confidence = EXCLUDED.confidence,
                                payload = EXCLUDED.payload,
                                updated_at = NOW()
                        """),
                        {
                            "analysis_id": analysis_id,
                            "title": conclusion,
                            "peer_ids": _pg_text_array(peer_ids),
                            "source_card_ids": _pg_text_array(source_card_ids),
                            "summary": conclusion,
                            "final_one_liner": conclusion,
                            "sk_ax_implication": payload.get("sk_ax_implication"),
                            "confidence": confidence,
                            "payload": json.dumps(body, ensure_ascii=False),
                        },
                    )
                elif analysis_type in {"global", "globaltrends"}:
                    db.execute(
                        text("""
                            INSERT INTO global_industry_trends (
                                source_analysis_id, trend_date, industry, region, keyword,
                                keyword_category, title, summary, mention_count, confidence,
                                related_peer_ids, related_card_ids, sk_ax_implication, payload
                            ) VALUES (
                                :analysis_id, CURRENT_DATE, 'legacy', 'global',
                                LEFT(:keyword, 120), 'analysis', :title, :summary, 0,
                                :confidence, CAST(:peer_ids AS text[]),
                                CAST(:source_card_ids AS text[]), :sk_ax_implication,
                                CAST(:payload AS jsonb)
                            )
                            ON CONFLICT (source_analysis_id)
                            WHERE source_analysis_id IS NOT NULL DO UPDATE SET
                                summary = EXCLUDED.summary,
                                confidence = EXCLUDED.confidence,
                                payload = EXCLUDED.payload,
                                updated_at = NOW()
                        """),
                        {
                            "analysis_id": analysis_id,
                            "keyword": payload.get("strategy_label") or analysis_id,
                            "title": conclusion,
                            "summary": conclusion,
                            "confidence": confidence,
                            "peer_ids": _pg_text_array(peer_ids),
                            "source_card_ids": _pg_text_array(source_card_ids),
                            "sk_ax_implication": payload.get("sk_ax_implication"),
                            "payload": json.dumps(body, ensure_ascii=False),
                        },
                    )
                else:
                    db.execute(
                        text("""
                            INSERT INTO legacy_records (
                                source_table, source_pk, owner_table, owner_id, payload
                            ) VALUES (
                                'analysis_ledger', :analysis_id, 'analysis', :analysis_type,
                                CAST(:payload AS jsonb)
                            )
                            ON CONFLICT (source_table, source_pk)
                            WHERE source_pk IS NOT NULL DO UPDATE SET
                                payload = EXCLUDED.payload,
                                archived_at = NOW()
                        """),
                        {
                            "analysis_id": analysis_id,
                            "analysis_type": analysis_type,
                            "payload": json.dumps(body, ensure_ascii=False),
                        },
                    )
                db.commit()
        except Exception as e:
            log.warning("Analysis read-model writeback failed (fail-soft) | %s", e)
            return {
                "ledger_id": -1,
                "inserted": False,
                "superseded_ledger_ids": [],
                "included_in_pack": False,
            }

        return {
            "ledger_id": -1,
            "inserted": True,
            "superseded_ledger_ids": [],
            "included_in_pack": included,
        }

    # ── Fetch (ContextPackBuilder 가 호출) ──────────────────────────────────

    @classmethod
    def fetch_top_n(
        cls,
        peer_id: str,
        top_n: int = 5,
        min_confidence: float | None = None,
        retention_days: int = 90,
    ) -> list[dict[str, Any]]:
        """ContextPackBuilder carry-over 용 V30 read model 조회."""
        thr = min_confidence if min_confidence is not None else _DEFAULT_MIN_CONF
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            """
                        SELECT source_analysis_id AS analysis_id,
                               'insight' AS analysis_type,
                               final_one_liner AS conclusion_one_liner,
                               NULL AS strategy_label,
                               confidence,
                               to_jsonb(source_card_ids) AS source_card_ids,
                               sk_ax_implication,
                               created_at
                        FROM insight_reports
                        WHERE :peer_id = ANY(focus_peer_ids)
                          AND confidence >= :thr
                          AND created_at > now() - make_interval(days => :days)
                        UNION ALL
                        SELECT source_analysis_id AS analysis_id,
                               'mixer' AS analysis_type,
                               final_one_liner AS conclusion_one_liner,
                               NULL AS strategy_label,
                               confidence,
                               to_jsonb(input_card_ids) AS source_card_ids,
                               sk_ax_implication,
                               created_at
                        FROM mixer_results
                        WHERE :peer_id = ANY(input_peer_ids)
                          AND confidence >= :thr
                          AND created_at > now() - make_interval(days => :days)
                        ORDER BY created_at DESC
                        LIMIT :top_n
                        """
                        ),
                        {
                            "peer_id": peer_id,
                            "thr": thr,
                            "days": retention_days,
                            "top_n": top_n,
                        },
                    )
                    .mappings()
                    .all()
                )
        except Exception as e:
            log.warning("AnalysisLedger.fetch_top_n failed | peer=%s | %s", peer_id, e)
            return []

        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────

_AGENT_TYPE_MAP = {
    "InsightCascadeAgent": "insight",
    "MixerAnalysisAgent": "mixer",
    "PeerComparisonAgent": "peer",
    "GlobalTrendsAgent": "global",
    "BriefingGenerationAgent": "briefing",
}


def with_ledger_writeback(agent_class: str) -> Callable:
    """분석 agent 의 async method 를 wrapping 하여 결과를 ledger 에 자동 INSERT.

    Decorator 는 결과 dict 에서 표준 필드를 추출 — agent 별로 schema 가 약간 다르므로
    fallback 키 후보를 다수 시도. INSERT 실패는 fail-soft (분석 결과는 정상 return).
    """

    analysis_type = _AGENT_TYPE_MAP.get(agent_class, agent_class.lower())

    def deco(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            if not isinstance(result, Mapping):
                return result
            try:
                payload = _extract_ledger_payload(agent_class, analysis_type, result, kwargs)
                AnalysisLedger.insert(payload)
            except Exception as e:
                log.warning(
                    "with_ledger_writeback 실패 (fail-soft) | agent=%s | %s",
                    agent_class,
                    e,
                )
            return result

        return wrapper

    return deco


def _extract_ledger_payload(
    agent_class: str,
    analysis_type: str,
    result: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> LedgerInsertInput:
    analysis_id = (
        result.get("id")
        or result.get("mix_id")
        or result.get("analysis_id")
        or result.get("briefing_id")
        or f"{analysis_type}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    peer_ids = result.get("peer_ids") or result.get("peers") or []
    if not peer_ids:
        # Mixer/Insight 는 input card_ids 의 peer 를 후방 추출
        cards = kwargs.get("cards") or kwargs.get("card_ids") or []
        peer_ids = _peer_ids_from_cards(cards)

    conclusion = (result.get("final_one_liner") or result.get("insight") or "")[:1000]
    confidence = float(result.get("confidence") or 0.0)
    source_card_ids = list(
        result.get("sources_used")
        or result.get("source_card_ids")
        or result.get("card_ids")
        or kwargs.get("card_ids")
        or []
    )

    return {
        "analysis_type": analysis_type,
        "analysis_id": str(analysis_id),
        "peer_ids": list(peer_ids),
        "conclusion_one_liner": conclusion,
        "strategy_label": result.get("strategy_label"),
        "confidence": confidence,
        "source_card_ids": [str(c) for c in source_card_ids],
        "sk_ax_implication": result.get("sk_ax_implication"),
        "langfuse_trace_id": result.get("langfuse_trace_id"),
        "prompt_version": str(
            kwargs.get("_prompt_version") or result.get("prompt_version") or "unversioned"
        ),
        "git_sha": _GIT_SHA_CACHE,
    }


def _peer_ids_from_cards(cards: Iterable[Any]) -> list[str]:
    """input cards (id list 또는 dict list) 에서 peer 후방 추출 — best-effort."""
    out: list[str] = []
    for card in cards:
        if isinstance(card, Mapping):
            peer = card.get("peer_id") or card.get("company")
            if peer:
                if isinstance(peer, list):
                    out.extend(str(p) for p in peer)
                else:
                    out.append(str(peer))
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _pg_text_array(values: Iterable[Any]) -> str:
    escaped: list[str] = []
    for value in values:
        text_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{text_value}"')
    return "{" + ",".join(escaped) + "}"
