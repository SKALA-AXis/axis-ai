"""CapabilityEvolutionAgent — W4-3 monthly capability narrative (Layer 2-B).

design/01-analysis-pipeline-implementation-plan.md §3.4.6, W4-3.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.analysis.prompts.capability_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.db.postgres import SessionLocal
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)

_LLM_MODEL = "gpt-4o"
_LLM_TEMPERATURE = 0.2
_LLM_MAX_COMPLETION_TOKENS = 3000
_MIN_SIGNALS = 5
_TOP_N_PER_GROUP = 5
_MAX_WINDOWS = 24

_PRE_AGG_SQL = """
WITH ranked_signals AS (
    SELECT
        id,
        peer_id,
        business_area,
        period_year,
        COALESCE(period_quarter::text, 'annual') AS pq,
        signal_type,
        sentiment,
        summary,
        evidence_text,
        confidence,
        raw_article_id,
        ROW_NUMBER() OVER (
            PARTITION BY peer_id, business_area, period_year, period_quarter, signal_type
            ORDER BY confidence DESC NULLS LAST, id DESC
        ) AS rn
    FROM raw_article_business_signals
    WHERE peer_id = ANY(:aliases)
      AND period_year >= :start_year
)
SELECT id, business_area, period_year, pq, signal_type, sentiment,
       summary, evidence_text, confidence, raw_article_id
FROM ranked_signals
WHERE rn <= :top_n
ORDER BY period_year DESC, business_area, signal_type
"""


class CapabilityEvolutionAgent:
    """월 1회. peer 별 business_signals 4분기 분 → LLM 합성 narrative."""

    prompt_version = PROMPT_VERSION

    def __init__(self, *, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    def run(
        self,
        *,
        peer_id: str,
        lookback_quarters: int = 4,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        signals = self._load_aggregated_signals(peer_id, lookback_quarters=lookback_quarters)
        if len(signals) < _MIN_SIGNALS:
            return {"skipped": True, "reason": "insufficient_signals", "signal_count": len(signals)}

        period_label = _period_label(signals, lookback_quarters=lookback_quarters)
        if use_llm:
            windows = self._invoke_llm(peer_id, signals, period_label, lookback_quarters)
            if not windows:
                windows = _deterministic_windows(signals, period_label=period_label)
        else:
            windows = _deterministic_windows(signals, period_label=period_label)

        return {
            "version": "capability-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "peer_id": peer_id,
            "period_label": period_label,
            "signal_count": len(signals),
            "windows": windows,
        }

    def persist(self, peer_id: str, result: dict[str, Any]) -> None:
        if result.get("skipped"):
            log.info("capability evolution skip | peer=%s reason=%s", peer_id, result.get("reason"))
            return
        windows = result.get("windows") or []
        if not windows:
            log.warning("capability evolution empty windows | peer=%s", peer_id)
            return

        with SessionLocal() as db:
            row = db.execute(
                text(
                    """
                    SELECT peer_plus_payload
                      FROM peer_companies
                     WHERE id = :peer_id
                    """
                ),
                {"peer_id": peer_id},
            ).fetchone()
            existing_payload: dict[str, Any] = {}
            if row is not None:
                raw = row._mapping.get("peer_plus_payload")
                if isinstance(raw, dict):
                    existing_payload = dict(raw)

            cap = existing_payload.get("capability_evolution") or {}
            if not isinstance(cap, dict):
                cap = {}
            prior_windows = cap.get("windows") or []
            if not isinstance(prior_windows, list):
                prior_windows = []

            merged = [*prior_windows, *windows]
            cap.update(
                {
                    "version": result.get("version") or "capability-v1",
                    "generated_at": result.get("generated_at"),
                    "period_label": result.get("period_label"),
                    "windows": merged[-_MAX_WINDOWS:],
                }
            )
            existing_payload["capability_evolution"] = cap

            db.execute(
                text(
                    """
                    UPDATE peer_companies
                       SET peer_plus_payload = CAST(:payload AS jsonb)
                     WHERE id = :peer_id
                    """
                ),
                {
                    "peer_id": peer_id,
                    "payload": json.dumps(existing_payload, ensure_ascii=False, default=str),
                },
            )
            db.commit()
        log.info(
            "capability evolution persisted | peer=%s windows=%s total=%s",
            peer_id,
            len(windows),
            len(cap.get("windows") or []),
        )

    def _load_aggregated_signals(
        self, peer_id: str, *, lookback_quarters: int
    ) -> list[dict[str, Any]]:
        aliases = expand_peer_aliases(peer_id)
        start_year = datetime.now(UTC).year - max(1, lookback_quarters // 4 + 1)
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(_PRE_AGG_SQL),
                    {
                        "aliases": aliases,
                        "start_year": start_year,
                        "top_n": _TOP_N_PER_GROUP,
                    },
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.warning("capability signals load failed | peer=%s error=%s", peer_id, exc)
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            mapping = row._mapping
            out.append(
                {
                    "id": str(mapping.get("id")),
                    "business_area": str(mapping.get("business_area") or ""),
                    "period_year": mapping.get("period_year"),
                    "period_quarter": mapping.get("pq"),
                    "signal_type": mapping.get("signal_type"),
                    "sentiment": mapping.get("sentiment"),
                    "summary": (mapping.get("summary") or "")[:400],
                    "evidence_text": (mapping.get("evidence_text") or "")[:500],
                    "confidence": mapping.get("confidence"),
                    "raw_article_id": mapping.get("raw_article_id"),
                }
            )
        return out

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=_LLM_MODEL,
                temperature=_LLM_TEMPERATURE,
                max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
            )
        return self._llm

    def _invoke_llm(
        self,
        peer_id: str,
        signals: list[dict[str, Any]],
        period_label: str,
        lookback_quarters: int,
    ) -> list[dict[str, Any]]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            peer_id=peer_id,
            lookback_quarters=lookback_quarters,
            period_label=period_label,
            signals_json=json.dumps(signals[:300], ensure_ascii=False, default=str),
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        try:
            response = self._get_llm().invoke(messages)
            text = str(getattr(response, "content", "") or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("capability LLM invoke failed | peer=%s error=%s", peer_id, exc)
            return []

        parsed = _parse_json_dict(text)
        windows = parsed.get("windows") if isinstance(parsed, dict) else None
        if not isinstance(windows, list):
            return []
        return _normalize_windows(windows, signals=signals, period_label=period_label)


def _period_label(signals: list[dict[str, Any]], *, lookback_quarters: int) -> str:
    years = sorted(
        {int(s["period_year"]) for s in signals if s.get("period_year") is not None},
        reverse=True,
    )
    if not years:
        return f"last_{lookback_quarters}q"
    quarters = [
        f"{s.get('period_year')}{s.get('period_quarter')}"
        for s in signals[:20]
        if s.get("period_quarter") and s.get("period_quarter") != "annual"
    ]
    if quarters:
        return f"{quarters[-1]}-{quarters[0]}"
    return f"{years[-1]}-{years[0]}"


def _deterministic_windows(
    signals: list[dict[str, Any]], *, period_label: str
) -> list[dict[str, Any]]:
    by_area: dict[str, list[dict[str, Any]]] = {}
    for signal in signals:
        area = str(signal.get("business_area") or "general")
        by_area.setdefault(area, []).append(signal)

    windows: list[dict[str, Any]] = []
    for area, items in sorted(by_area.items()):
        top = sorted(
            items,
            key=lambda s: float(s.get("confidence") or 0),
            reverse=True,
        )[:3]
        summaries = [str(s.get("summary") or "").strip() for s in top if s.get("summary")]
        narrative = (
            " ".join(summaries)[:280] if summaries else f"{area} 영역 신호가 누적되고 있습니다."
        )
        evidence_ids = [str(s.get("id")) for s in top if s.get("id")]
        confidences = [
            float(s.get("confidence") or 0) for s in top if s.get("confidence") is not None
        ]
        windows.append(
            {
                "period": period_label,
                "business_area": area,
                "narrative": narrative,
                "evidence_signal_ids": evidence_ids,
                "delta_intensity": round(min(1.0, len(top) / 5), 2),
                "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.5,
                "generated_at": datetime.now(UTC).isoformat(),
            }
        )
    return windows[:20]


def _normalize_windows(
    windows: list[Any],
    *,
    signals: list[dict[str, Any]],
    period_label: str,
) -> list[dict[str, Any]]:
    valid_ids = {str(s.get("id")) for s in signals if s.get("id")}
    signal_by_area: dict[str, list[str]] = {}
    for signal in signals:
        area = str(signal.get("business_area") or "")
        if area and signal.get("id"):
            signal_by_area.setdefault(area, []).append(str(signal["id"]))

    out: list[dict[str, Any]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        area = str(item.get("business_area") or "").strip()
        narrative = str(item.get("narrative") or "").strip()
        if not area or not narrative:
            continue
        evidence_ids = [
            str(x) for x in (item.get("evidence_signal_ids") or []) if str(x) in valid_ids
        ]
        if not evidence_ids and area in signal_by_area:
            evidence_ids = signal_by_area[area][:3]
        if not evidence_ids:
            continue
        out.append(
            {
                "period": str(item.get("period") or period_label),
                "business_area": area,
                "narrative": narrative[:400],
                "evidence_signal_ids": evidence_ids,
                "delta_intensity": _clamp_float(item.get("delta_intensity"), default=0.5),
                "confidence": _clamp_float(item.get("confidence"), default=0.6),
                "generated_at": datetime.now(UTC).isoformat(),
            }
        )
    return out


def _parse_json_dict(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, round(num, 2)))


__all__ = ["CapabilityEvolutionAgent"]
