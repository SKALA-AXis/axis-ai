"""크롤러 에이전트 — 수집 파이프라인용 DB 조회 인터페이스.

실제 수집은 APScheduler → BatchProcessor 경로로 이미 처리되어 raw_articles에 저장된다.
이 에이전트는 LangGraph ingestion_graph의 crawl_node에서 처리 대기 중인
RAW 기사 ID를 DB에서 가져오는 역할만 한다.
"""

import logging

from sqlalchemy import text

from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

_LOAD_SQL = text("""
    SELECT ra.id
    FROM raw_articles ra
    WHERE ra.processing_status = 'RAW'
      AND (:no_filter OR ra.company ?| :company)
      AND (:collected_since IS NULL OR ra.collected_at >= CAST(:collected_since AS timestamptz))
      AND (
          :crawl_run_id IS NULL
          OR EXISTS (
              SELECT 1
              FROM crawl_run_articles cra
              WHERE cra.raw_article_id = ra.id
                AND cra.crawl_run_id = CAST(:crawl_run_id AS uuid)
          )
      )
    ORDER BY ra.published_at DESC NULLS LAST
    LIMIT :limit
""")


class CrawlerAgent:
    """수집 파이프라인의 crawl_node 전담 — RAW 기사 ID 조회."""

    def load_raw_ids(
        self,
        company: list[str] | None = None,
        limit: int = 500,
        collected_since: str | None = None,
        crawl_run_id: str | None = None,
    ) -> list[int]:
        """처리 대기 중인 RAW 기사 ID를 조회한다.

        Args:
            company: 대상 기업 ID 목록. 빈 리스트 또는 None이면 전체 조회.
            limit: 최대 조회 건수.
            collected_since: 지정 시 해당 시각 이후 수집된 RAW만 조회.
            crawl_run_id: 지정 시 해당 backfill 실행에서 수집된 RAW만 조회.

        Returns:
            raw_articles.id 목록.
        """
        company_filter = company or []

        with SessionLocal() as db:
            rows = db.execute(
                _LOAD_SQL,
                {
                    "company": company_filter if company_filter else [""],
                    "no_filter": len(company_filter) == 0,
                    "collected_since": collected_since,
                    "crawl_run_id": crawl_run_id,
                    "limit": limit,
                },
            ).fetchall()

        ids = [row.id for row in rows]
        log.info(
            "RAW 기사 로드 | company=%s collected_since=%s crawl_run_id=%s count=%d",
            company_filter,
            collected_since,
            crawl_run_id,
            len(ids),
        )
        return ids
