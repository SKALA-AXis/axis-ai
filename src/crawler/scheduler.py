"""APScheduler 기반 크롤 스케줄러 — Track A (1시간) + Track B (새벽 2시)."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.crawler.batch_processor import BatchProcessor

log = logging.getLogger(__name__)

# 모니터링 대상 Peer사별 검색 키워드
PEER_KEYWORDS: dict[str, list[str]] = {
    "samsung_sds": ["삼성SDS", "Samsung SDS", "삼성에스디에스"],
    "lg_cns": ["LG CNS", "엘지씨엔에스", "LGCNS"],
    "hyundai_autoever": ["현대오토에버", "Hyundai AutoEver", "현대오토에버시스템"],
    "posco_dx": ["포스코DX", "포스코디엑스", "POSCO DX"],
}


def build_scheduler() -> AsyncIOScheduler:
    """APScheduler 인스턴스 생성 및 Track A/B 잡 등록."""
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    processor = BatchProcessor()

    # Track A — 실시간 수집 (1시간 간격)
    scheduler.add_job(
        processor.run_track_a,
        trigger=IntervalTrigger(hours=1),
        kwargs={"keywords": PEER_KEYWORDS},
        id="track_a_crawl",
        name="Track A — 실시간 크롤 (Naver/RSS/BigKinds)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    # Track B — 배치 수집 (매일 새벽 2시)
    scheduler.add_job(
        processor.run_track_b,
        trigger=CronTrigger(hour=2, minute=0),
        kwargs={"keywords": PEER_KEYWORDS},
        id="track_b_crawl",
        name="Track B — 배치 크롤 (DART/KIPRIS/뉴스룸/채용)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    log.info("크롤 스케줄러 구성 완료 | jobs=%d", len(scheduler.get_jobs()))
    return scheduler
