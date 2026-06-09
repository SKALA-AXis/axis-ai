"""WeeklyDigestAgent — K2 weekly peer narrative from card_news (Layer 2-C supplement).

Stores in peer_companies.peer_plus_payload['weekly_digest'] (append-only digests).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.analysis.prompts.weekly_digest_v1 import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.db.postgres import SessionLocal
from src.services.peer_id_aliases import expand_peer_aliases

log = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")
_LLM_MODEL = "gpt-4o"
_LLM_TEMPERATURE = 0.2
_LLM_MAX_COMPLETION_TOKENS = 2500
_MIN_CARDS = 1
_MAX_DIGESTS = 12
_CARD_LIMIT = 30


class WeeklyDigestAgent:
    """주 1회. peer 별 최근 7일 card_news → LLM weekly narrative."""

    prompt_version = PROMPT_VERSION

    def __init__(self, *, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    def run(
        self,
        *,
        peer_id: str,
        lookback_days: int = 7,
        use_llm: bool = True,
        anchor_date: date | None = None,
    ) -> dict[str, Any]:
        anchor = anchor_date or datetime.now(KST).date()
        since = anchor - timedelta(days=lookback_days)
        cards = self._load_recent_cards(peer_id, since_date=since, until_date=anchor)
        if len(cards) < _MIN_CARDS:
            return {
                "skipped": True,
                "reason": "insufficient_cards",
                "card_count": len(cards),
            }

        week_iso = anchor.strftime("%G-W%V")
        period_label = f"{since.isoformat()}~{anchor.isoformat()}"
        prev_digest = self._load_previous_digest(peer_id)

        if use_llm:
            digest_body = self._invoke_llm(
                peer_id=peer_id,
                cards=cards,
                week_iso=week_iso,
                period_label=period_label,
                prev_digest=prev_digest,
            )
            if not digest_body:
                digest_body = _deterministic_digest(
                    cards=cards,
                    week_iso=week_iso,
                    period_label=period_label,
                    prev_digest=prev_digest,
                )
        else:
            digest_body = _deterministic_digest(
                cards=cards,
                week_iso=week_iso,
                period_label=period_label,
                prev_digest=prev_digest,
            )

        digest_entry = {
            "week_iso": week_iso,
            "period_label": period_label,
            "since_kst": since.isoformat(),
            "until_kst": anchor.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            **digest_body,
        }
        return {
            "version": "weekly-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "peer_id": peer_id,
            "card_count": len(cards),
            "digest": digest_entry,
        }

    def persist(self, peer_id: str, result: dict[str, Any]) -> None:
        if result.get("skipped"):
            log.info("weekly digest skip | peer=%s reason=%s", peer_id, result.get("reason"))
            return
        digest = result.get("digest")
        if not isinstance(digest, dict):
            return

        with SessionLocal() as db:
            row = db.execute(
                text("SELECT peer_plus_payload FROM peer_companies WHERE id = :peer_id"),
                {"peer_id": peer_id},
            ).fetchone()
            payload: dict[str, Any] = {}
            if row is not None:
                raw = row._mapping.get("peer_plus_payload")
                if isinstance(raw, dict):
                    payload = dict(raw)

            weekly = payload.get("weekly_digest") or {}
            if not isinstance(weekly, dict):
                weekly = {}
            digests = weekly.get("digests") or []
            if not isinstance(digests, list):
                digests = []
            digests = [d for d in digests if isinstance(d, dict) and d.get("week_iso") != digest.get("week_iso")]
            digests.append(digest)
            weekly.update(
                {
                    "version": result.get("version") or "weekly-v1",
                    "generated_at": result.get("generated_at"),
                    "digests": digests[-_MAX_DIGESTS:],
                }
            )
            payload["weekly_digest"] = weekly
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
                    "payload": json.dumps(payload, ensure_ascii=False, default=str),
                },
            )
            db.commit()
        log.info("weekly digest persisted | peer=%s week=%s", peer_id, digest.get("week_iso"))

    def _load_recent_cards(
        self, peer_id: str, *, since_date: date, until_date: date
    ) -> list[dict[str, Any]]:
        aliases = expand_peer_aliases(peer_id)
        try:
            with SessionLocal() as db:
                rows = db.execute(
                    text(
                        """
                        SELECT id, title, summary_lines, event_type,
                               primary_keyword_category AS sector,
                               importance, importance_score, created_at
                          FROM card_news
                         WHERE peer_company_id = ANY(:aliases)
                           AND (created_at AT TIME ZONE 'Asia/Seoul')::date
                               BETWEEN :since_date AND :until_date
                         ORDER BY importance_score DESC NULLS LAST, created_at DESC
                         LIMIT :limit
                        """
                    ),
                    {
                        "aliases": aliases,
                        "since_date": since_date.isoformat(),
                        "until_date": until_date.isoformat(),
                        "limit": _CARD_LIMIT,
                    },
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            log.warning("weekly digest cards load failed | peer=%s error=%s", peer_id, exc)
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            mapping = row._mapping
            summary_lines = mapping.get("summary_lines") or []
            if not isinstance(summary_lines, list):
                summary_lines = []
            out.append(
                {
                    "id": str(mapping.get("id") or ""),
                    "title": str(mapping.get("title") or "")[:300],
                    "summary_lines": [str(line)[:200] for line in summary_lines[:4]],
                    "event_type": mapping.get("event_type"),
                    "sector": mapping.get("sector"),
                    "importance": mapping.get("importance"),
                    "importance_score": mapping.get("importance_score"),
                    "created_at": str(mapping.get("created_at") or ""),
                }
            )
        return out

    def _load_previous_digest(self, peer_id: str) -> dict[str, Any] | None:
        try:
            with SessionLocal() as db:
                row = db.execute(
                    text(
                        """
                        SELECT peer_plus_payload->'weekly_digest'->'digests' AS digests
                          FROM peer_companies
                         WHERE id = :peer_id
                        """
                    ),
                    {"peer_id": peer_id},
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            log.debug("previous weekly digest load failed | peer=%s error=%s", peer_id, exc)
            return None
        if row is None:
            return None
        digests = row._mapping.get("digests")
        if not isinstance(digests, list) or not digests:
            return None
        latest = digests[-1]
        return latest if isinstance(latest, dict) else None

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
        *,
        peer_id: str,
        cards: list[dict[str, Any]],
        week_iso: str,
        period_label: str,
        prev_digest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            peer_id=peer_id,
            week_iso=week_iso,
            period_label=period_label,
            prev_digest_json=json.dumps(prev_digest or {}, ensure_ascii=False, default=str),
            cards_json=json.dumps(cards, ensure_ascii=False, default=str),
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        try:
            response = self._get_llm().invoke(messages)
            text = str(getattr(response, "content", "") or "")
        except Exception as exc:  # noqa: BLE001
            log.warning("weekly digest LLM failed | peer=%s error=%s", peer_id, exc)
            return {}
        parsed = _parse_json_dict(text)
        return _normalize_digest_body(parsed, cards=cards)


def _deterministic_digest(
    *,
    cards: list[dict[str, Any]],
    week_iso: str,
    period_label: str,
    prev_digest: dict[str, Any] | None,
) -> dict[str, Any]:
    titles = [str(c.get("title") or "").strip() for c in cards[:5] if c.get("title")]
    narrative = (
        f"{period_label} 동안 {len(cards)}건의 카드뉴스가 감지되었습니다. "
        + " ".join(titles[:3])
    )[:800]
    card_ids = [str(c["id"]) for c in cards[:10] if c.get("id")]
    delta: list[str] = []
    if prev_digest and prev_digest.get("narrative"):
        delta.append("이번 주 카드 건수·주요 이벤트 유형이 이전 주 digest 와 비교해 갱신되었습니다.")
    return {
        "narrative": narrative,
        "delta_vs_prev": delta,
        "strategy_label": "Observing",
        "source_card_ids": card_ids,
        "confidence": 0.55 if len(cards) < 3 else 0.65,
    }


def _normalize_digest_body(parsed: dict[str, Any], *, cards: list[dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {str(c["id"]) for c in cards if c.get("id")}
    source_ids = [str(x) for x in (parsed.get("source_card_ids") or []) if str(x) in valid_ids]
    if not source_ids:
        source_ids = list(valid_ids)[:5]
    narrative = str(parsed.get("narrative") or "").strip()
    if not narrative:
        return {}
    delta = [str(x).strip() for x in (parsed.get("delta_vs_prev") or []) if str(x).strip()][:5]
    confidence = parsed.get("confidence")
    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.6
    return {
        "narrative": narrative[:1500],
        "delta_vs_prev": delta,
        "strategy_label": str(parsed.get("strategy_label") or "Observing")[:30],
        "source_card_ids": source_ids,
        "confidence": round(conf, 2),
    }


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


__all__ = ["WeeklyDigestAgent"]
