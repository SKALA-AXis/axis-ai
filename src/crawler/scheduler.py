"""APScheduler 기반 크롤 스케줄러."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config.companies import COMPANY_ALIASES
from src.crawler.batch_processor import BatchProcessor
from src.crawler.monitors.keepalive import keepalive

log = logging.getLogger(__name__)

PEER_ALIASES: dict[str, list[str]] = dict(COMPANY_ALIASES)


def build_scheduler() -> AsyncIOScheduler:
    """APScheduler 인스턴스 생성 및 크롤링 잡 등록."""
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    processor = BatchProcessor()

    scheduler.add_job(
        processor.run_track_a,
        trigger=IntervalTrigger(hours=1),
        kwargs={"keywords": PEER_ALIASES},
        id="track_a_crawl",
        name="Track A — 실시간 크롤 (Naver/RSS/BigKinds)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    scheduler.add_job(
        processor.run_track_b,
        trigger=CronTrigger(hour=2, minute=0),
        kwargs={"keywords": PEER_ALIASES},
        id="track_b_crawl",
        name="Track B — 배치 크롤 (공시/공식채널/채용/특허/리서치)",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        keepalive,
        trigger=IntervalTrigger(hours=24),
        id="cloud_keepalive",
        name="Supabase·Qdrant Cloud keepalive ping",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    log.info("크롤 스케줄러 구성 완료 | jobs=%d", len(scheduler.get_jobs()))
    return scheduler
