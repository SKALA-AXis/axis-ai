# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — AnalysisLedger backbone 신설, Layer B 분석 파이프라인 연동, ledger read 경로·mypy/ruff 정리
"""KnowledgeCuration K1 — AnalysisLedger middleware.

분석 agent (InsightCascade / MixerAnalysis / ITTrend /
BriefingGeneration) 의 결론을 ``analysis_ledger`` 테이블에 INSERT 하여 다음
분석 호출 시 ``ContextPackBuilder`` 가 carry-over 하도록 한다.

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
_DEDUP_WINDOW_HOURS = int(os.getenv("LEDGER_DEDUP_WINDOW_HOURS", "24"))
_DEDUP_TEXT_SIM = float(os.getenv("LEDGER_DEDUP_TEXT_SIM", "0.92"))
_DEDUP_CONF_DELTA = float(os.getenv("LEDGER_DEDUP_CONF_DELTA", "0.05"))


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


def _normalize_entry(
    *,
    row_id: Any,
    analysis_id: str,
    analysis_type: str,
    conclusion: str,
    confidence: float,
    source_card_ids: list[str],
    sk_ax_implication: str | None,
    created_at: str,
    source: str,
    strategy_label: str | None = None,
    included_in_pack: bool | None = None,
) -> dict[str, Any] | None:
    text_value = (conclusion or "").strip()
    if not text_value:
        return None
    entry: dict[str, Any] = {
        "id": row_id,
        "analysis_id": analysis_id,
        "analysis_type": analysis_type,
        "conclusion_one_liner": text_value,
        "confidence": confidence,
        "source_card_ids": list(source_card_ids or []),
        "sk_ax_implication": sk_ax_implication,
        "created_at": created_at,
        "source": source,
    }
    if strategy_label is not None:
        entry["strategy_label"] = strategy_label
    if included_in_pack is not None:
        entry["included_in_pack"] = included_in_pack
    return entry


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


_CONTRADICTORY_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Aggressive Expansion", "Defensive Hold"),
        ("Defensive Hold", "Aggressive Expansion"),
        ("Tech Pivot", "Cost Leadership"),
        ("Cost Leadership", "Tech Pivot"),
    }
)


# ──────────────────────────────────────────────────────────────────────────
# Core insert
# ──────────────────────────────────────────────────────────────────────────


class AnalysisLedger:
    """``analysis_ledger`` 테이블 write-through helper."""

    @classmethod
    def insert(cls, payload: LedgerInsertInput) -> LedgerInsertResult:
        threshold = _DEFAULT_MIN_CONF
        confidence = float(payload.get("confidence", 0.0))
        included = confidence >= threshold

        peer_ids = list(payload.get("peer_ids") or [])
        conclusion = (payload.get("conclusion_one_liner") or "").strip()
        analysis_type = payload.get("analysis_type") or "unknown"

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

        # 1. Dedup — 24h 내 동일 peer + type + similar conclusion + similar confidence
        if cls._is_recent_duplicate(peer_ids, analysis_type, conclusion, confidence):
            log.info(
                "AnalysisLedger.insert | dedup skip | type=%s peer_ids=%s",
                analysis_type,
                peer_ids,
            )
            return {
                "ledger_id": -1,
                "inserted": False,
                "superseded_ledger_ids": [],
                "included_in_pack": False,
            }

        # 2. Supersede detection — strategy 가 *반대 방향* 으로 뒤집힘
        superseded_ids = cls._detect_supersede(peer_ids, analysis_type, payload)

        # 3. INSERT
        try:
            with SessionLocal() as db:
                row = db.execute(
                    text(
                        """
                        INSERT INTO analysis_ledger (
                            analysis_type, analysis_id, peer_ids, conclusion_one_liner,
                            strategy_label, confidence, source_card_ids, sk_ax_implication,
                            langfuse_trace_id, prompt_version, git_sha, included_in_pack
                        ) VALUES (
                            :analysis_type, :analysis_id, CAST(:peer_ids AS JSONB),
                            :conclusion, :strategy_label, :confidence,
                            CAST(:source_card_ids AS JSONB), :sk_ax_implication,
                            :langfuse_trace_id, :prompt_version, :git_sha, :included_in_pack
                        )
                        RETURNING id
                        """
                    ),
                    {
                        "analysis_type": analysis_type,
                        "analysis_id": payload.get("analysis_id") or "",
                        "peer_ids": json.dumps(peer_ids),
                        "conclusion": conclusion[:1000],
                        "strategy_label": payload.get("strategy_label"),
                        "confidence": confidence,
                        "source_card_ids": json.dumps(list(payload.get("source_card_ids") or [])),
                        "sk_ax_implication": payload.get("sk_ax_implication"),
                        "langfuse_trace_id": payload.get("langfuse_trace_id"),
                        "prompt_version": payload.get("prompt_version") or "unversioned",
                        "git_sha": payload.get("git_sha") or _GIT_SHA_CACHE,
                        "included_in_pack": included,
                    },
                ).fetchone()
                ledger_id = int(row[0]) if row else -1

                if superseded_ids:
                    db.execute(
                        text(
                            "UPDATE analysis_ledger SET superseded_by = :new_id "
                            "WHERE id = ANY(CAST(:ids AS BIGINT[]))"
                        ),
                        {
                            "new_id": ledger_id,
                            "ids": "{" + ",".join(str(i) for i in superseded_ids) + "}",
                        },
                    )

                db.commit()
        except Exception as e:  # pragma: no cover — fail-soft
            log.warning("AnalysisLedger.insert | DB INSERT 실패 (fail-soft) | error=%s", e)
            return {
                "ledger_id": -1,
                "inserted": False,
                "superseded_ledger_ids": [],
                "included_in_pack": False,
            }

        log.info(
            "AnalysisLedger.insert | id=%d type=%s peers=%s conf=%.2f included=%s superseded=%s",
            ledger_id,
            analysis_type,
            peer_ids,
            confidence,
            included,
            superseded_ids,
        )
        return {
            "ledger_id": ledger_id,
            "inserted": True,
            "superseded_ledger_ids": superseded_ids,
            "included_in_pack": included,
        }

    # ── Dedup ────────────────────────────────────────────────────────────

    @classmethod
    def _is_recent_duplicate(
        cls,
        peer_ids: list[str],
        analysis_type: str,
        conclusion: str,
        confidence: float,
    ) -> bool:
        if not peer_ids:
            return False
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT conclusion_one_liner, confidence
                        FROM analysis_ledger
                        WHERE analysis_type = :type
                          AND peer_ids ?| CAST(:peers AS TEXT[])
                          AND created_at > now() - make_interval(hours => :hours)
                        ORDER BY created_at DESC
                        LIMIT 20
                        """
                    ),
                    {
                        "type": analysis_type,
                        "peers": "{" + ",".join(peer_ids) + "}",
                        "hours": _DEDUP_WINDOW_HOURS,
                    },
                ).fetchall()
        except Exception as e:
            log.warning("AnalysisLedger dedup query failed (skip dedup) | %s", e)
            return False

        for row in rows:
            existing = (row[0] or "").strip()
            existing_conf = float(row[1] or 0.0)
            if abs(existing_conf - confidence) > _DEDUP_CONF_DELTA:
                continue
            if _text_similarity(existing, conclusion) >= _DEDUP_TEXT_SIM:
                return True
        return False

    # ── Supersede ────────────────────────────────────────────────────────

    @classmethod
    def _detect_supersede(
        cls,
        peer_ids: list[str],
        analysis_type: str,
        payload: LedgerInsertInput,
    ) -> list[int]:
        if not peer_ids:
            return []
        new_strategy = payload.get("strategy_label")
        new_polarity = _polarity(payload.get("sk_ax_implication") or "")
        new_conclusion = payload.get("conclusion_one_liner") or ""

        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT id, strategy_label, sk_ax_implication, conclusion_one_liner
                        FROM analysis_ledger
                        WHERE analysis_type = :type
                          AND peer_ids ?| CAST(:peers AS TEXT[])
                          AND superseded_by IS NULL
                        ORDER BY created_at DESC
                        LIMIT 5
                        """
                    ),
                    {
                        "type": analysis_type,
                        "peers": "{" + ",".join(peer_ids) + "}",
                    },
                ).fetchall()
        except Exception as e:
            log.warning("AnalysisLedger supersede query failed (skip) | %s", e)
            return []

        superseded: list[int] = []
        for prev in rows:
            prev_id = int(prev[0])
            prev_strategy = prev[1]
            prev_impl = prev[2] or ""
            prev_conclusion = prev[3] or ""

            contradictory_strategy = bool(
                new_strategy
                and prev_strategy
                and (prev_strategy, new_strategy) in _CONTRADICTORY_PAIRS
            )
            polarity_swap = _polarity(prev_impl) != new_polarity and (
                _polarity(prev_impl) in ("positive", "negative")
                and new_polarity in ("positive", "negative")
            )
            negation_swap = _contains_negation_swap(prev_conclusion, new_conclusion)

            if contradictory_strategy or polarity_swap or negation_swap:
                superseded.append(prev_id)
        return superseded

    # ── Fetch (ContextPackAssembler read path) ─────────────────────────────

    @classmethod
    def fetch_top_n(
        cls,
        peer_id: str,
        top_n: int = 5,
        min_confidence: float | None = None,
        retention_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Peer-scoped recent analysis conclusions for context carry-over."""
        thr = min_confidence if min_confidence is not None else _DEFAULT_MIN_CONF
        try:
            entries = cls._collect_read_model_entries(
                peer_id=peer_id,
                min_confidence=thr,
                retention_days=retention_days,
                limit=top_n,
            )
        except Exception as e:
            log.warning("AnalysisLedger.fetch_top_n failed | peer=%s | %s", peer_id, e)
            return []
        return entries[:top_n]

    @classmethod
    def fetch_recent(
        cls,
        *,
        window_days: int = 60,
        limit: int = 8,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Cross-peer recent conclusions for dashboard agents."""
        try:
            entries = cls._collect_read_model_entries(
                peer_id=None,
                min_confidence=min_confidence,
                retention_days=window_days,
                limit=limit,
            )
        except Exception as e:
            log.warning("AnalysisLedger.fetch_recent failed | %s", e)
            return []
        return entries[:limit]

    @classmethod
    def _collect_read_model_entries(
        cls,
        *,
        peer_id: str | None,
        min_confidence: float,
        retention_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        fetch_limit = max(limit * 3, limit)
        with SessionLocal() as db:
            entries.extend(
                cls._fetch_insight_entries(
                    db,
                    peer_id=peer_id,
                    min_confidence=min_confidence,
                    retention_days=retention_days,
                    limit=fetch_limit,
                )
            )
            entries.extend(
                cls._fetch_mixer_entries(
                    db,
                    peer_id=peer_id,
                    min_confidence=min_confidence,
                    retention_days=retention_days,
                    limit=fetch_limit,
                )
            )
            if peer_id:
                entries.extend(
                    cls._fetch_legacy_entries(
                        db,
                        peer_id=peer_id,
                        min_confidence=min_confidence,
                    )
                )
        entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return entries[:limit]

    @classmethod
    def _fetch_insight_entries(
        cls,
        db: Any,
        *,
        peer_id: str | None,
        min_confidence: float,
        retention_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        peer_filter = ""
        params: dict[str, Any] = {
            "thr": min_confidence,
            "days": retention_days,
            "limit": limit,
        }
        if peer_id:
            peer_filter = "AND :peer_id = ANY(focus_peer_ids)"
            params["peer_id"] = peer_id
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, final_one_liner, confidence, source_card_ids,
                       sk_ax_implication, payload, created_at
                FROM insight_reports
                WHERE confidence >= :thr
                  AND created_at > now() - make_interval(days => :days)
                  {peer_filter}
                ORDER BY created_at DESC
                LIMIT :limit
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            normalized = _normalize_entry(
                row_id=row.get("id"),
                analysis_id=str(row.get("id") or ""),
                analysis_type="insight",
                conclusion=str(row.get("final_one_liner") or ""),
                confidence=float(row.get("confidence") or 0.0),
                source_card_ids=list(row.get("source_card_ids") or []),
                sk_ax_implication=row.get("sk_ax_implication"),
                created_at=str(row.get("created_at") or ""),
                source="insight_reports",
                strategy_label=payload.get("strategy_label"),
            )
            if normalized:
                out.append(normalized)
        return out

    @classmethod
    def _fetch_mixer_entries(
        cls,
        db: Any,
        *,
        peer_id: str | None,
        min_confidence: float,
        retention_days: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        peer_filter = ""
        params: dict[str, Any] = {
            "thr": min_confidence,
            "days": retention_days,
            "limit": limit,
        }
        if peer_id:
            peer_filter = "AND :peer_id = ANY(input_peer_ids)"
            params["peer_id"] = peer_id
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, final_one_liner, confidence, input_card_ids,
                       sk_ax_implication, payload, created_at
                FROM mixer_results
                WHERE confidence >= :thr
                  AND created_at > now() - make_interval(days => :days)
                  {peer_filter}
                ORDER BY created_at DESC
                LIMIT :limit
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            normalized = _normalize_entry(
                row_id=row.get("id"),
                analysis_id=str(row.get("id") or ""),
                analysis_type="mixer",
                conclusion=str(row.get("final_one_liner") or ""),
                confidence=float(row.get("confidence") or 0.0),
                source_card_ids=list(row.get("input_card_ids") or []),
                sk_ax_implication=row.get("sk_ax_implication"),
                created_at=str(row.get("created_at") or ""),
                source="mixer_results",
                strategy_label=payload.get("strategy_label"),
            )
            if normalized:
                out.append(normalized)
        return out

    @classmethod
    def _fetch_legacy_entries(
        cls,
        db: Any,
        *,
        peer_id: str,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        row = db.execute(
            text(
                """
                SELECT legacy_payload->'analysis_ledger' AS ledger
                FROM peer_companies
                WHERE id = :peer_id
                """
            ),
            {"peer_id": peer_id},
        ).fetchone()
        if not row or row[0] is None:
            return []
        raw = row[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence") or 0.0)
            if confidence < min_confidence:
                continue
            normalized = _normalize_entry(
                row_id=item.get("id"),
                analysis_id=str(item.get("analysis_id") or ""),
                analysis_type=str(item.get("analysis_type") or "legacy"),
                conclusion=str(item.get("conclusion_one_liner") or ""),
                confidence=confidence,
                source_card_ids=list(item.get("source_card_ids") or []),
                sk_ax_implication=item.get("sk_ax_implication"),
                created_at=str(item.get("created_at") or ""),
                source="legacy_payload",
                strategy_label=item.get("strategy_label"),
                included_in_pack=item.get("included_in_pack"),
            )
            if normalized:
                out.append(normalized)
        return out


# ──────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────

_AGENT_TYPE_MAP = {
    "InsightCascadeAgent": "insight",
    "MixerAnalysisAgent": "mixer",
    "ITTrendAgent": "it_trend",
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


# ──────────────────────────────────────────────────────────────────────────
# Helpers — text similarity / polarity / negation
# ──────────────────────────────────────────────────────────────────────────


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_set = set(a.split())
    b_set = set(b.split())
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


_POSITIVE_KEYWORDS = ("긍정", "기회", "확대", "성장", "강화", "유리")
_NEGATIVE_KEYWORDS = ("부정", "위협", "축소", "감소", "약화", "불리", "철수")


def _polarity(text_value: str) -> str:
    t = text_value or ""
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in t)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in t)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


_NEGATION_PAIRS = (
    ("확대", "축소"),
    ("축소", "확대"),
    ("진출", "철수"),
    ("철수", "진출"),
    ("성장", "감소"),
    ("감소", "성장"),
)


def _contains_negation_swap(a: str, b: str) -> bool:
    if not a or not b:
        return False
    for pos, neg in _NEGATION_PAIRS:
        if pos in a and neg in b:
            return True
    return False
