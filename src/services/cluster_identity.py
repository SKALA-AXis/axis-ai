"""Card 의 stable identity 추출 — P3-CRIT-1/2 대응.

`card_news.cluster_id` 는 ephemeral seq (distinct 21, max 90,001) 이므로 join key
로 사용하면 안 된다. 대신:
- `source_raw_article_ids` (15% 카드는 빈 array) → 우선 join key
- `(peer_company_id, event_type, DATE(created_at))` 복합키 → fallback

`stable_card_join_key()` 가 두 정보를 함께 묶어 반환하여 downstream 에서 어떤 키를
사용할지 일관되게 결정할 수 있도록 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class CardStableIdentity:
    """카드 1건의 stable join 키. card_id 는 항상 안전 sentinel."""

    card_id: str
    source_raw_article_ids: tuple[int, ...] = ()
    peer_event_date_key: tuple[str | None, str, str] | None = None
    has_raw_articles: bool = False
    has_peer_event_date: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def stable_card_join_key(card: dict[str, Any]) -> CardStableIdentity:
    """card_news row dict 로부터 stable identity 를 추출.

    Parameters
    ----------
    card : dict
        `card_news` row 또는 카드 payload. 다음 필드를 우선 읽음:
        id, source_raw_article_ids, peer_company_id, company, event_type, created_at.
    """
    card_id = str(card.get("id") or "")
    raw_ids = _normalize_int_tuple(card.get("source_raw_article_ids"))
    peer = card.get("peer_company_id") or card.get("company") or card.get("peer_id") or None
    if peer is not None:
        peer = str(peer)
    event_type = str(card.get("event_type") or "").strip()
    date_key = _date_from(card.get("created_at"))
    peer_event_date_key: tuple[str | None, str, str] | None = None
    if event_type and date_key:
        peer_event_date_key = (peer, event_type, date_key)
    return CardStableIdentity(
        card_id=card_id,
        source_raw_article_ids=raw_ids,
        peer_event_date_key=peer_event_date_key,
        has_raw_articles=len(raw_ids) > 0,
        has_peer_event_date=peer_event_date_key is not None,
    )


def _normalize_int_tuple(value: Any) -> tuple[int, ...]:
    if not value:
        return ()
    if isinstance(value, list | tuple | set):
        out: list[int] = []
        seen: set[int] = set()
        for item in value:
            try:
                ivalue = int(item)
            except (TypeError, ValueError):
                continue
            if ivalue <= 0 or ivalue in seen:
                continue
            seen.add(ivalue)
            out.append(ivalue)
        return tuple(out)
    try:
        single = int(value)
    except (TypeError, ValueError):
        return ()
    return (single,) if single > 0 else ()


def _date_from(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return ""
    # ISO-8601 like "2026-05-21T03:21:14+09:00" → take first 10 chars.
    return text[:10]


__all__ = ["CardStableIdentity", "stable_card_join_key"]
