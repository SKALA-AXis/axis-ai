"""HTTP HEAD/GET 기반 출처 링크 검증 서비스.

design: ``axis-ai/design/30-analysis/link-verification.md``.

prototype 범위 (Walking Skeleton Phase 2): HTTP HEAD (200 ok / 404 dead / 3xx redirect /
기타 error) + GET 으로 content hash diff (옵션). LLM 미사용 — deterministic.

``link_verification_logs`` 테이블 INSERT 는 Day 90+ 후속 작업 (V14 migration 신설 필요).
본 prototype 은 in-memory 결과만 반환.

핵심 entry point:

    ``LinkVerificationService().verify(card_id)``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_HEAD_TIMEOUT_SECONDS = 5.0
_GET_TIMEOUT_SECONDS = 8.0
_USER_AGENT = "AxisLinkVerifier/1.0"


class LinkVerificationService:
    """deterministic HTTP head/get 기반 link liveness + content diff 검증."""

    async def verify(self, card_id: str) -> dict:
        """카드의 source URL 들을 검증.

        Args:
            card_id: 검증 대상 카드 id.

        Returns:
            LinkVerificationOutput dict — design §5 schema.
        """
        sources = _fetch_card_sources(card_id)
        verified_at = datetime.now(UTC)

        if not sources:
            return {
                "card_id": card_id,
                "sources": [],
                "overall_status": "all_dead",
                "verified_at": verified_at,
                "warning": "카드 source URL 없음",
            }

        statuses = await asyncio.gather(
            *(_check_source(src) for src in sources),
            return_exceptions=False,
        )

        overall = _compute_overall(statuses)
        return {
            "card_id": card_id,
            "sources": statuses,
            "overall_status": overall,
            "verified_at": verified_at,
            "warning": None if overall == "all_live" else None,
        }


# ──────────────────────────────────────────────────────────────────────────
# Per-source check (HEAD → optional GET hash)
# ──────────────────────────────────────────────────────────────────────────


async def _check_source(src: dict) -> dict:
    url = src.get("url") or ""
    original_hash = src.get("original_content_hash") or src.get("content_hash")
    checked_at = datetime.now(UTC)

    if not url:
        return {
            "url": "",
            "status": "error",
            "http_code": None,
            "final_url": None,
            "content_changed": False,
            "last_modified": None,
            "checked_at": checked_at,
        }

    try:
        async with httpx.AsyncClient(
            timeout=_HEAD_TIMEOUT_SECONDS,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            head = await client.head(url)
            final_url = url
            http_code = head.status_code

            # 3xx → redirect chain (max 1 hop, 추가 hop 은 final_url 만 기록)
            if 300 <= http_code < 400:
                final_url = head.headers.get("location", url)
                if final_url and final_url != url:
                    try:
                        head = await client.head(final_url)
                        http_code = head.status_code
                    except (httpx.RequestError, httpx.TimeoutException):
                        pass

            if http_code == 200:
                base_status: str = "live"
            elif http_code in (404, 410):
                base_status = "dead"
            elif 300 <= http_code < 400:
                base_status = "redirected"
            else:
                base_status = "error"

            content_changed = False
            if base_status == "live" and original_hash:
                content_changed = await _check_content_hash(client, final_url, original_hash)

            status = (
                "live (content_changed)"
                if (base_status == "live" and content_changed)
                else base_status
            )

            return {
                "url": url,
                "status": status,
                "http_code": http_code,
                "final_url": final_url if final_url != url else None,
                "content_changed": content_changed,
                "last_modified": head.headers.get("last-modified"),
                "checked_at": checked_at,
            }
    except (httpx.TimeoutException, httpx.RequestError) as e:
        log.warning("LinkVerify | %s 검증 실패 (%s)", url, e)
        return {
            "url": url,
            "status": "error",
            "http_code": None,
            "final_url": None,
            "content_changed": False,
            "last_modified": None,
            "checked_at": checked_at,
        }


async def _check_content_hash(client: httpx.AsyncClient, url: str, original_hash: str) -> bool:
    try:
        resp = await asyncio.wait_for(client.get(url), timeout=_GET_TIMEOUT_SECONDS)
        body = resp.text
        current_hash = hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return current_hash != original_hash
    except (httpx.RequestError, httpx.TimeoutException, asyncio.TimeoutError):
        return False


def _compute_overall(statuses: list[dict]) -> str:
    if not statuses:
        return "all_dead"
    live_count = sum(1 for s in statuses if str(s["status"]).startswith("live"))
    dead_count = sum(1 for s in statuses if s["status"] in ("dead", "error"))
    content_changed = any(s.get("content_changed") for s in statuses)
    if content_changed:
        return "content_changed"
    if live_count == len(statuses):
        return "all_live"
    if dead_count == len(statuses):
        return "all_dead"
    return "some_dead"


# ──────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────


def _fetch_card_sources(card_id: str) -> list[dict]:
    """카드의 sources jsonb 를 조회하여 list[dict] 로 반환."""
    sql = "SELECT sources FROM card_news WHERE id = :id"
    try:
        with SessionLocal() as db:
            row = db.execute(text(sql), {"id": card_id}).mappings().first()
    except Exception as e:
        log.exception("LinkVerify card_news 조회 실패 | %s", e)
        return []

    if not row:
        return []

    raw = row["sources"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [src for src in raw if isinstance(src, dict) and src.get("url")]
