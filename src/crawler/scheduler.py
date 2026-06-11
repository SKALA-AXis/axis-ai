"""APScheduler 기반 소스별 크롤 스케줄러."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config.companies import COMPANY_ALIASES, CORP_CODES
from src.crawler.base import CrawlRunContext, RawArticle
from src.crawler.monitors.keepalive import keepalive
from src.crawler.parsers.dedup import DedupStore
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.result_writer import DEFAULT_RESULTS_DIR
from src.db.article_store import save_articles
from src.db.crawl_state_store import (
    create_crawl_run,
    mark_crawl_run_failed,
    mark_crawl_run_success,
)

log = logging.getLogger(__name__)

PEER_ALIASES: dict[str, list[str]] = dict(COMPANY_ALIASES)
KEYWORD_RUNNER_PATH = Path(__file__).resolve().parents[2] / "keyword.py"

_CrawlerFactory = Callable[[str, list[str]], Any]


def build_scheduler() -> AsyncIOScheduler:
    """APScheduler 인스턴스 생성 및 소스별 크롤링 job 등록."""
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")

    _add_cron_job(
        scheduler,
        _run_naver_news,
        job_id="news_naver",
        name="뉴스 — Naver",
        minute=0,
        misfire_grace_time=600,
    )
    _add_cron_job(
        scheduler,
        _run_global_newsroom,
        job_id="global_newsroom",
        name="공식 뉴스룸 — Global",
        minute=25,
        misfire_grace_time=600,
    )
    _add_cron_job(
        scheduler,
        _run_company_news,
        job_id="homepage_news",
        name="홈페이지별 뉴스",
        hour=11,
        minute=0,
        misfire_grace_time=3600,
    )
    _add_cron_job(
        scheduler,
        _run_dart_adhoc,
        job_id="dart_adhoc",
        name="DART 수시 체크",
        day_of_week="mon,wed,fri",
        hour=2,
        minute=0,
        misfire_grace_time=7200,
    )
    _add_cron_job(
        scheduler,
        _run_dart_quarter,
        job_id="dart_quarter",
        name="DART 분기 집중 체크",
        month="1,4,7,10",
        day="1-10",
        hour=2,
        minute=30,
        misfire_grace_time=7200,
    )
    _add_cron_job(
        scheduler,
        _run_ir,
        job_id="ir",
        name="IR",
        day_of_week="mon",
        hour=3,
        minute=0,
        misfire_grace_time=7200,
    )
    _add_cron_job(
        scheduler,
        _run_jobs,
        job_id="jobs_work24",
        name="채용공고 — 고용24",
        hour="8,17",
        minute=30,
        misfire_grace_time=1800,
    )
    _add_cron_job(
        scheduler,
        _run_naver_research,
        job_id="naver_research",
        name="증권사 리포트 — 네이버 증권",
        day_of_week="mon-fri",
        hour=10,
        minute=30,
        misfire_grace_time=1800,
    )
    _add_cron_job(
        scheduler,
        _run_naver_datalab,
        job_id="naver_datalab",
        name="검색량 — Naver DataLab",
        hour=0,
        minute=30,
        misfire_grace_time=1800,
    )
    _add_cron_job(
        scheduler,
        _run_spri,
        job_id="industry_spri",
        name="산업동향 — SPRi",
        day="1-7",
        hour=4,
        minute=0,
        misfire_grace_time=7200,
    )
    _add_cron_job(
        scheduler,
        _run_bcg,
        job_id="industry_bcg",
        name="산업동향 — BCG",
        day_of_week="mon",
        hour=4,
        minute=30,
        misfire_grace_time=3600,
    )
    _add_cron_job(
        scheduler,
        _run_stock,
        job_id="stock_naver_finance",
        name="주식 — Naver Finance",
        day_of_week="mon-fri",
        hour="9-15",
        minute=40,
        misfire_grace_time=600,
    )

    scheduler.add_job(
        keepalive,
        trigger=IntervalTrigger(hours=24),
        id="cloud_keepalive",
        name="Keepalive",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )

    log.info("크롤 스케줄러 구성 완료 | jobs=%d", len(scheduler.get_jobs()))
    return scheduler


def _add_cron_job(
    scheduler: AsyncIOScheduler,
    func: Callable[[], Awaitable[None]],
    *,
    job_id: str,
    name: str,
    misfire_grace_time: int,
    **cron_kwargs: Any,
) -> None:
    scheduler.add_job(
        func,
        trigger=CronTrigger(**cron_kwargs),
        id=job_id,
        name=name,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=misfire_grace_time,
    )


async def _run_naver_news() -> None:
    from src.crawler.sources.naver import NaverNewsCrawler

    await _run_company_source(
        "news_naver",
        lambda peer_id, aliases: NaverNewsCrawler(
            peer_id=peer_id,
            aliases=aliases,
            cutoff_datetime=_news_cutoff(),
        ),
    )


async def _run_global_newsroom() -> None:
    from src.crawler.sources.global_newsroom import GlobalNewsroomCrawler

    await _run_shared_source("global_newsroom", GlobalNewsroomCrawler(max_pages=5))


async def _run_company_news() -> None:
    from src.crawler.sources.company_news import CompanyNewsCrawler

    await _run_shared_source("homepage_news", CompanyNewsCrawler())


async def _run_dart_adhoc() -> None:
    await _run_dart("dart_adhoc", lookback_days=3)


async def _run_dart_quarter() -> None:
    await _run_dart("dart_quarter", lookback_days=14)


async def _run_dart(source_label: str, lookback_days: int) -> None:
    await _run_company_source(
        source_label,
        lambda peer_id, aliases: _build_dart_crawler(peer_id, aliases, lookback_days),
    )


def _build_dart_crawler(peer_id: str, aliases: list[str], lookback_days: int) -> Any:
    from src.crawler.sources.dart import DartCrawler

    crawler = DartCrawler(
        peer_id=peer_id,
        corp_code=CORP_CODES.get(peer_id, ""),
        corp_names=aliases,
    )
    crawler.lookback_days = lookback_days
    return crawler


async def _run_ir() -> None:
    from src.crawler.sources.ir import IRCrawler

    await _run_company_source("ir", lambda peer_id, _aliases: IRCrawler(peer_id=peer_id))


async def _run_jobs() -> None:
    from src.crawler.sources.jobs import JobCrawler

    await _run_company_source("jobs_work24", lambda peer_id, _aliases: JobCrawler(peer_id=peer_id))


async def _run_naver_research() -> None:
    from src.crawler.sources.naver_research import NaverResearchCrawler

    await _run_company_source(
        "naver_research",
        lambda peer_id, _aliases: NaverResearchCrawler(peer_id=peer_id),
    )


async def _run_naver_datalab() -> None:
    today = datetime.now().astimezone().date()
    run_id = create_crawl_run(
        "naver_datalab",
        today,
        today,
        run_type="realtime",
    )
    try:
        inserted = await asyncio.to_thread(_run_keyword_sector_runner_sync, str(run_id))
        mark_crawl_run_success(run_id, inserted_count=inserted, skipped_count=0)
    except Exception as e:
        mark_crawl_run_failed(run_id, f"{type(e).__name__}: {e}")
        raise

    log.info(
        "Naver DataLab keyword.py 스케줄 실행 완료 | crawl_run_id=%s inserted=%s",
        run_id,
        inserted,
    )


def _run_keyword_sector_runner_sync(crawl_run_id: str) -> int:
    spec = importlib.util.spec_from_file_location(
        "axis_naver_datalab_keyword_sector_runner",
        KEYWORD_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"keyword.py 로드 실패: {KEYWORD_RUNNER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_scheduled = getattr(module, "run_scheduled", None)
    if not callable(run_scheduled):
        raise RuntimeError("keyword.py에 run_scheduled()가 없습니다.")

    return int(run_scheduled(crawl_run_id=crawl_run_id) or 0)


async def _run_spri() -> None:
    from src.crawler.sources.spri import SpriCrawler

    month = _previous_month_label()
    output_path = DEFAULT_RESULTS_DIR / "spri_scheduler.json"
    await _run_shared_source("industry_spri", SpriCrawler(month=month, output_path=output_path))


async def _run_bcg() -> None:
    from src.crawler.sources.bcg import BcgCrawler

    output_path = DEFAULT_RESULTS_DIR / "bcg_scheduler.json"
    await _run_shared_source(
        "industry_bcg",
        BcgCrawler(days=7, max_articles=100, output_path=output_path),
    )


async def _run_stock() -> None:
    from src.crawler.sources.stock import StockCrawler

    await _run_company_source(
        "stock_naver_finance",
        lambda peer_id, _aliases: StockCrawler(peer_id=peer_id),
    )


async def _run_company_source(source_label: str, factory: _CrawlerFactory) -> None:
    log.info("소스 크롤 시작 | source=%s companies=%d", source_label, len(PEER_ALIASES))

    async def crawl_one(peer_id: str, aliases: list[str]) -> list[RawArticle]:
        crawler = factory(peer_id, aliases)
        try:
            return await _normalize_articles(await crawler.crawl())
        except Exception as e:
            log.error(
                "소스 크롤 실패 | source=%s company=%s crawler=%s error=%s",
                source_label,
                peer_id,
                type(crawler).__name__,
                e,
            )
            return []

    results = await asyncio.gather(
        *(crawl_one(peer_id, aliases) for peer_id, aliases in PEER_ALIASES.items())
    )
    articles = [article for chunk in results for article in chunk]
    await _persist_articles(source_label, articles)


async def _run_shared_source(source_label: str, crawler: Any) -> None:
    log.info("소스 크롤 시작 | source=%s crawler=%s", source_label, type(crawler).__name__)
    try:
        articles = await _normalize_articles(await crawler.crawl())
    except Exception as e:
        log.error(
            "소스 크롤 실패 | source=%s crawler=%s error=%s",
            source_label,
            type(crawler).__name__,
            e,
        )
        return

    await _persist_articles(source_label, articles)


async def _persist_articles(source_label: str, articles: list[RawArticle]) -> None:
    today = datetime.now().astimezone().date()
    run_id = create_crawl_run(
        source_label,
        today,
        today,
        run_type="realtime",
    )

    if not articles:
        log.info("소스 크롤 결과 없음 | source=%s", source_label)
        mark_crawl_run_success(run_id, inserted_count=0, skipped_count=0)
        return

    try:
        accessible, rejected = await LinkChecker().filter_accessible(articles)
        new_articles = DedupStore().filter_new(accessible)
        inserted = save_articles(
            new_articles,
            run_context=CrawlRunContext(
                collection_mode="realtime",
                crawl_run_id=str(run_id),
                source_name=source_label,
            ),
        )
        skipped = len(new_articles) - inserted
        mark_crawl_run_success(run_id, inserted_count=inserted, skipped_count=skipped)
        _run_realtime_pipeline_for_run(str(run_id), trigger_type=f"scheduler:{source_label}")
    except Exception as e:
        mark_crawl_run_failed(run_id, f"{type(e).__name__}: {e}")
        raise

    log.info(
        "소스 크롤 저장 완료 | source=%s raw=%d accessible=%d rejected=%d new=%d inserted=%d",
        source_label,
        len(articles),
        len(accessible),
        len(rejected),
        len(new_articles),
        inserted,
    )


def _run_realtime_pipeline_for_run(crawl_run_id: str, *, trigger_type: str) -> None:
    """Run preprocessing for a realtime crawl run."""
    from src.preprocessing.classification import ClusterClassifier
    from src.preprocessing.preprocessing import PreprocessingService
    from src.preprocessing.relevance import RelevanceEvaluator

    source_label = trigger_type.removeprefix("scheduler:")
    track_a_sources = {"news_naver", "global_newsroom"}

    result = PreprocessingService(
        relevance_evaluator=RelevanceEvaluator(enable_llm=source_label in track_a_sources),
        classifier=ClusterClassifier(enable_llm=False),
    ).run(
        company=[],
        trigger_type=trigger_type,
        collected_since=None,
        crawl_run_id=crawl_run_id,
    )
    log.info(
        (
            "실시간 후처리 완료 | crawl_run_id=%s raw=%d parsed_docs=%d "
            "analysis_metrics=%d analysis_signals=%d classified=%d"
        ),
        crawl_run_id,
        len(result.get("raw_article_ids", [])),
        len(result.get("parsed_document_ids", [])),
        result.get("analysis_metric_count", 0),
        result.get("analysis_signal_count", 0),
        len(result.get("classified_clusters", [])),
    )


async def _normalize_articles(items: list[Any]) -> list[RawArticle]:
    return [_to_raw_article(item) for item in items]


def _to_raw_article(item: Any) -> RawArticle:
    if isinstance(item, RawArticle):
        return item

    if hasattr(item, "to_common_dict"):
        data = item.to_common_dict()
    elif isinstance(item, dict):
        data = item
    else:
        raise TypeError(f"RawArticle로 변환할 수 없는 크롤 결과입니다: {type(item).__name__}")

    extra = dict(data.get("extra") or data.get("metadata") or {})
    published_at = _parse_datetime(data.get("published_at"))
    collected_at = _parse_datetime(data.get("collected_at")) or datetime.now().astimezone()

    if data.get("data") is not None:
        extra.setdefault("data", data["data"])
    if data.get("ticker") is not None:
        extra.setdefault("ticker", data["ticker"])
    if data.get("schema_name") is not None:
        extra.setdefault("schema_name", data["schema_name"])
    if data.get("company_tier") is not None:
        extra.setdefault("company_tier", data["company_tier"])

    return RawArticle(
        url=str(data.get("url") or data.get("source") or ""),
        title=str(data.get("title") or ""),
        content=data.get("content"),
        source_name=str(data.get("source_name") or "unknown"),
        published_at=published_at,
        collected_at=collected_at,
        source_type=data.get("source_type") or extra.get("source_type") or "news",
        content_type=data.get("content_type") or extra.get("content_type") or "html",
        publisher=data.get("publisher"),
        company=list(data.get("company") or []),
        language=data.get("language") or "ko",
        crawl_status=data.get("crawl_status") or "success",
        error_message=data.get("error_message"),
        url_hash=data.get("url_hash") or "",
        extra=extra,
        peer_id=data.get("peer_id"),
    )


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _previous_month_label() -> str:
    today = datetime.now().date()
    year = today.year
    month = today.month - 1

    if month == 0:
        year -= 1
        month = 12

    return f"{year:04d}-{month:02d}"


def _news_cutoff() -> datetime:
    return datetime.now().astimezone() - timedelta(hours=1)
