"""Persistence helpers for generated briefing reports."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal

_UPSERT_REPORT_SQL = text("""
    INSERT INTO briefing_reports (
        id, title, briefing_type, date_from, date_to, requested_by_user_id,
        status, progress, payload, error_message, confidence, provenance,
        completed_at
    ) VALUES (
        :id, :title, :briefing_type, :date_from, :date_to, :requested_by_user_id,
        :status, :progress, CAST(:payload AS jsonb), :error_message, :confidence,
        CAST(:provenance AS jsonb), NOW()
    )
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        briefing_type = EXCLUDED.briefing_type,
        date_from = EXCLUDED.date_from,
        date_to = EXCLUDED.date_to,
        requested_by_user_id = EXCLUDED.requested_by_user_id,
        status = EXCLUDED.status,
        progress = EXCLUDED.progress,
        payload = EXCLUDED.payload,
        error_message = EXCLUDED.error_message,
        confidence = EXCLUDED.confidence,
        provenance = EXCLUDED.provenance,
        completed_at = NOW()
""")

_INSERT_CARD_SQL = text("""
    INSERT INTO briefing_report_cards (briefing_report_id, card_news_id)
    VALUES (:briefing_report_id, :card_news_id)
    ON CONFLICT DO NOTHING
""")

_INSERT_ARTICLE_SQL = text("""
    INSERT INTO briefing_report_articles (briefing_report_id, raw_article_id)
    VALUES (:briefing_report_id, :raw_article_id)
    ON CONFLICT DO NOTHING
""")

_SELECT_REPORT_SQL = text("""
    SELECT payload, completed_at
    FROM briefing_reports
    WHERE id = :id
      AND status IN ('completed', 'delivered')
      AND payload IS NOT NULL
""")


def load_briefing_report(report_id: str) -> dict[str, Any] | None:
    """저장된 브리핑을 재사용(read-through 캐시)하기 위해 조회한다.

    Returns:
        ``{"payload": dict, "completed_at": datetime | None}`` 또는 미존재 시 None.
        payload 가 dict 로 해석되지 않으면 None (손상 row 는 캐시 미스로 처리).
    """

    with SessionLocal() as db:
        row = db.execute(_SELECT_REPORT_SQL, {"id": report_id}).mappings().first()
    if row is None:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, dict):
        return None
    return {"payload": payload, "completed_at": row["completed_at"]}


def save_briefing_report(
    report: dict[str, Any],
    *,
    selected_cards: list[dict[str, Any]],
) -> None:
    """Persist a generated briefing and its card/article mappings.

    Expected schema:
    - ``briefing_reports`` stores the full frontend payload in ``payload`` JSONB.
    - ``briefing_report_cards`` stores report-card links.
    - ``briefing_report_articles`` stores report-raw article links.
    """

    with SessionLocal() as db:
        db.execute(
            _UPSERT_REPORT_SQL,
            {
                "id": report.get("id"),
                "title": report.get("title"),
                "briefing_type": report.get("briefing_type"),
                "date_from": report.get("date_from"),
                "date_to": report.get("date_to"),
                "requested_by_user_id": report.get("requested_by_user_id"),
                "status": report.get("status"),
                "progress": report.get("progress"),
                "payload": json.dumps(report, ensure_ascii=False, default=str),
                "error_message": report.get("error_message"),
                "confidence": report.get("confidence"),
                "provenance": json.dumps(report.get("provenance") or {}, ensure_ascii=False),
            },
        )
        for card in selected_cards:
            card_id = str(card.get("id") or "").strip()
            if not card_id:
                continue
            db.execute(
                _INSERT_CARD_SQL,
                {"briefing_report_id": report.get("id"), "card_news_id": card_id},
            )
            for article_id in _source_article_ids(card):
                db.execute(
                    _INSERT_ARTICLE_SQL,
                    {
                        "briefing_report_id": report.get("id"),
                        "raw_article_id": article_id,
                    },
                )
        db.commit()


def _source_article_ids(card: dict[str, Any]) -> list[int]:
    values = card.get("source_raw_article_ids") or []
    if not isinstance(values, list):
        values = [values]
    result: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed not in result:
            result.append(parsed)
    return result
