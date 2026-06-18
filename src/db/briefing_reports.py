# 작성일: 2026-05-27
# 작성자: 심유정
# 변경이력:
#   2026-05-27 심유정 — 브리핑 생성 agent 추가 및 포맷 정리
#   2026-06-12 최종민 — 브리핑 read-through 재사용으로 페이지 로딩 시 LLM 반복 호출 제거
#   2026-06-12 박진 — 목업 삭제 및 챗봇 고도화
"""Persistence helpers for generated briefing reports."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from src.db.postgres import SessionLocal

_UPSERT_REPORT_SQL = text("""
    INSERT INTO briefing_reports (
        id, title, briefing_type, date_from, date_to, requested_by_user_id,
        report_date, period_label, status, progress, key_summary, sk_implication,
        related_card_ids, related_raw_article_ids, payload, error_message, confidence, provenance,
        completed_at
    ) VALUES (
        :id, :title, :briefing_type, :date_from, :date_to, :requested_by_user_id,
        :report_date, :period_label, :status, :progress, :key_summary, :sk_implication,
        CAST(:related_card_ids AS text[]), CAST(:related_raw_article_ids AS bigint[]),
        CAST(:payload AS jsonb), :error_message, :confidence,
        CAST(:provenance AS jsonb), NOW()
    )
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        briefing_type = EXCLUDED.briefing_type,
        date_from = EXCLUDED.date_from,
        date_to = EXCLUDED.date_to,
        requested_by_user_id = EXCLUDED.requested_by_user_id,
        report_date = EXCLUDED.report_date,
        period_label = EXCLUDED.period_label,
        status = EXCLUDED.status,
        progress = EXCLUDED.progress,
        key_summary = EXCLUDED.key_summary,
        sk_implication = EXCLUDED.sk_implication,
        related_card_ids = EXCLUDED.related_card_ids,
        related_raw_article_ids = EXCLUDED.related_raw_article_ids,
        payload = EXCLUDED.payload,
        error_message = EXCLUDED.error_message,
        confidence = EXCLUDED.confidence,
        provenance = EXCLUDED.provenance,
        completed_at = NOW()
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
    - Related card/article ids are stored in array columns on ``briefing_reports``.
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
                "report_date": report.get("report_date") or report.get("date_to"),
                "period_label": report.get("period_label"),
                "status": report.get("status"),
                "progress": report.get("progress"),
                "key_summary": report.get("key_summary") or report.get("executive_summary"),
                "sk_implication": _sk_implication(report),
                "related_card_ids": _pg_text_array(_card_ids(report, selected_cards)),
                "related_raw_article_ids": _pg_bigint_array(_raw_article_ids(selected_cards)),
                "payload": json.dumps(report, ensure_ascii=False, default=str),
                "error_message": report.get("error_message"),
                "confidence": report.get("confidence"),
                "provenance": json.dumps(report.get("provenance") or {}, ensure_ascii=False),
            },
        )
        db.commit()


def _card_ids(report: dict[str, Any], selected_cards: list[dict[str, Any]]) -> list[str]:
    raw_values = report.get("related_card_ids") or report.get("source_card_ids") or []
    values = list(raw_values) if isinstance(raw_values, list) else [raw_values]
    values.extend(card.get("id") for card in selected_cards)
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result


def _raw_article_ids(selected_cards: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for card in selected_cards:
        for article_id in _source_article_ids(card):
            if article_id not in result:
                result.append(article_id)
    return result


def _sk_implication(report: dict[str, Any]) -> str | None:
    basis = report.get("briefing_basis")
    if isinstance(basis, dict):
        strategy = basis.get("strategy_implication")
        if isinstance(strategy, dict):
            value = strategy.get("finding") or strategy.get("rationale")
            if value:
                return str(value)
    value = report.get("sk_implication") or report.get("executive_implication")
    return str(value) if value else None


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


def _pg_text_array(values: list[Any]) -> str:
    cleaned = [str(value).replace('"', '\\"') for value in values if value]
    if not cleaned:
        return "{}"
    return "{" + ",".join(f'"{value}"' for value in cleaned) + "}"


def _pg_bigint_array(values: list[int]) -> str:
    if not values:
        return "{}"
    return "{" + ",".join(str(int(value)) for value in values) + "}"
