"""Supabase·Qdrant Cloud 자동 일시정지 방지용 keepalive ping.

무료 티어는 일정 시간 무사용 시 인스턴스를 일시정지한다.
APScheduler에서 24시간 간격으로 ping하여 활성 상태 유지.
"""

import logging

from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.db.qdrant_client import get_qdrant_client

log = logging.getLogger(__name__)


async def keepalive() -> None:
    """Supabase·Qdrant Cloud에 가벼운 쿼리를 보내 활성 상태를 유지."""
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        log.info("keepalive: Supabase ping ok")
    except Exception as e:
        log.error("keepalive: Supabase ping 실패 | error=%s", e)

    try:
        client = get_qdrant_client()
        client.get_collections()
        log.info("keepalive: Qdrant Cloud ping ok")
    except Exception as e:
        log.error("keepalive: Qdrant Cloud ping 실패 | error=%s", e)
