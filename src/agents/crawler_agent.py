"""크롤러 에이전트 — 수집 파이프라인용 DB 조회 인터페이스.

실제 수집은 APScheduler → BatchProcessor 경로로 이미 처리되어 raw_articles에 저장됩니다.
이 에이전트는 LangGraph ingestion_graph의 crawl_node에서 처리 대기 중인
RAW 기사 ID를 DB에서 가져오는 역할만 합니다.
"""

import logging

from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_LOAD_SQL = text("""
    SELECT id FROM raw_articles
    WHERE processing_status = 'RAW'
      AND (:no_filter OR peer_id = ANY(:peer_ids))
    ORDER BY published_at DESC
    LIMIT :limit
""")


class CrawlerAgent:
    """수집 파이프라인의 crawl_node 전담 — RAW 기사 ID 조회."""

    def load_raw_ids(self, peer_ids: list[str], limit: int = 500) -> list[int]:
        """처리 대기 중인 RAW 기사 ID를 조회한다.

        Args:
            peer_ids: 대상 Peer사 ID 목록. 빈 리스트 → 전체 조회.
            limit: 최대 조회 건수.

        Returns:
            raw_articles.id 목록.
        """
        with SessionLocal() as db:
            rows = db.execute(
                _LOAD_SQL,
                {
                    "peer_ids": peer_ids if peer_ids else [""],
                    "no_filter": len(peer_ids) == 0,
                    "limit": limit,
                },
            ).fetchall()
        ids = [row.id for row in rows]
        log.info("RAW 기사 로드 | peer_ids=%s count=%d", peer_ids, len(ids))
        return ids
