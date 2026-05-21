"""CapabilityEvolutionAgent — W4-3.

월 1회 (CronJob `axis-cron-capability-evolution`) 실행. 4분기 누적
`raw_article_business_signals` 를 SQL 단에서 (peer × business_area × period ×
signal_type) top-5 confidence 로 압축 후 LLM 으로 business_area 별 narrative
합성. 결과를 `peer_companies.peer_plus_payload['capability_evolution']` JSONB 에
저장.

설계: design/01-supervisor-implementation-plan.md §3.4.6.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

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
_LLM_MAX_COMPLETION_TOKENS = 2000

_MIN_SIGNALS = 5
_TOP_PER_GROUP = 5

_PRE_AGG_SQL = text(
    """
    WITH ranked_signals AS (
        SELECT
            peer_id,
            business_area,
            period_year,
            COALESCE(period_quarter::text, 'annual') AS period_quarter_safe,
            signal_type,
            sentiment,
            summary,
            evidence_text,
            confidence,
            raw_article_id,
            ROW_NUMBER() OVER (
                PARTITION BY peer_id, business_area, period_year, period_quarter, signal_type
                ORDER BY confidence DESC NULLS LAST
            ) AS rn
          FROM raw_article_business_signals
         WHERE peer_id = ANY(:aliases)
           AND period_year >= :start_year
    )
    SELECT peer_id, business_area, period_year, period_quarter_safe AS pq,
           signal_type, sentiment, summary, evidence_text,
           confidence, raw_article_id
      FROM ranked_signals
     WHERE rn <= :top
     ORDER BY period_year DESC, business_area, signal_type
    """
)


class CapabilityEvolutionAgent:
    """월 1회 peer 별 capability narrative 합성."""

    prompt_version = PROMPT_VERSION
    model = _LLM_MODEL

    def __init__(self, *, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    def run(self, *, peer_id: str, lookback_quarters: int = 4) -> dict[str, Any]:
        signals = self._load_signals(peer_id=peer_id, lookback_quarters=lookback_quarters)
        if len(signals) < _MIN_SIGNALS:
            log.info(
                "capability evolution skipped | peer=%s signals=%d < %d",
                peer_id,
                len(signals),
                _MIN_SIGNALS,
            )
            return {"skipped": True, "reason": "insufficient_signals", "peer_id": peer_id}
        narrative = self._invoke_llm(peer_id=peer_id, signals=signals)
        narrative.setdefault("version", "capability-v1")
        narrative.setdefault("generated_at", datetime.now(UTC).isoformat())
        narrative.setdefault("peer_id", peer_id)
        return narrative

    def persist(self, *, peer_id: str, result: dict[str, Any]) -> None:
        if result.get("skipped"):
            return
        payload = {
            "version": result.get("version", "capability-v1"),
            "generated_at": result.get("generated_at"),
            "windows": result.get("windows", []),
        }
        with SessionLocal() as db:
            db.execute(
                text(
                    """
                    UPDATE peer_companies
                       SET peer_plus_payload = COALESCE(peer_plus_payload, '{}'::jsonb)
                           || jsonb_build_object('capability_evolution', CAST(:payload AS jsonb))
                     WHERE id = :peer_id
                    """
                ),
                {
                    "peer_id": peer_id,
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            )
            db.commit()
        log.info(
            "capability evolution persisted | peer=%s windows=%d",
            peer_id,
            len(payload["windows"]),
        )

    # ------------------------------------------------------------------
    def _load_signals(self, *, peer_id: str, lookback_quarters: int) -> list[dict[str, Any]]:
        aliases = expand_peer_aliases(peer_id)
        start_year = max(2020, datetime.now(UTC).year - (lookback_quarters // 4 + 1))
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    _PRE_AGG_SQL,
                    {
                        "aliases": aliases,
                        "start_year": int(start_year),
                        "top": int(_TOP_PER_GROUP),
                    },
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.warning("signals load failed | peer=%s error=%s", peer_id, exc)
            return []
        return [dict(row._mapping) for row in rows]

    def _invoke_llm(
        self,
        *,
        peer_id: str,
        signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            peer_id=peer_id,
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2, default=str),
        )
        llm = self._get_llm()
        try:
            from src.observability import tracing_config

            config = tracing_config(
                agent="CapabilityEvolutionAgent",
                phase="generate",
                prompt_version=self.prompt_version,
                peer_id=peer_id,
            )
        except Exception:  # noqa: BLE001
            config = None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = llm.invoke(messages, config=config) if config else llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_json_loose(content)

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=_LLM_MODEL,
                temperature=_LLM_TEMPERATURE,
                max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        return self._llm


def _parse_json_loose(text_value: str) -> dict[str, Any]:
    text_value = (text_value or "").strip()
    if not text_value:
        return {}
    if text_value.startswith("```"):
        parts = text_value.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            text_value = body.strip()
    if not text_value.startswith("{"):
        first = text_value.find("{")
        last = text_value.rfind("}")
        if first >= 0 and last > first:
            text_value = text_value[first : last + 1]
    try:
        parsed = json.loads(text_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["CapabilityEvolutionAgent"]
