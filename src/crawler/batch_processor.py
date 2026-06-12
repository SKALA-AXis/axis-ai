"""Track A/B/C/D 크롤 오케스트레이터 — 원천 수집 + URL 중복 제거 + 저장."""

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol
from uuid import UUID

from src.config.companies import COMPANY_ALIASES, CORP_CODES
from src.config.global_companies import GLOBAL_COMPANY_ALIASES, GLOBAL_COMPANY_IDS
from src.crawler.base import CrawlRunContext, CrawlWindow, DailyLimitGuard, RawArticle
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

TRACK_A_SOURCES = (
    "naver_news",
    "global_newsroom",
)
TRACK_B_SOURCES = (
    "stock",
    "naver_research",
)
TRACK_C_SOURCES = (
    "jobs",
    "company_news",
)
TRACK_D_SOURCES = (
    "dart",
    "ir",
    "naver_datalab",
    "spri",
    "bcg",
)

REALTIME_SOURCE_OVERLAP_DAYS = {
    "naver_news": 0,
    "global_newsroom": 1,
    "stock": 0,
    "naver_research": 3,
    "jobs": 1,
    "company_news": 7,
    "dart": 30,
    "ir": 365,
    "sk_ax_site": 7,
    "naver_datalab": 7,
    "spri": 30,
    "bcg": 30,
}


class _Crawlable(Protocol):
    async def crawl(self) -> list[Any]: ...


class BatchProcessor:
    def __init__(self) -> None:
        self.limit_guard = DailyLimitGuard()
        self.dedup = DedupStore()
        self.link_checker = LinkChecker()
        self.last_inserted_count = 0
        self.last_crawl_run_ids: list[str] = []
        self.last_crawl_run_records: list[dict[str, str]] = []

    async def run_track_a(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        recent_hours: int = 1,
        crawl_window: CrawlWindow | None = None,
        run_context: CrawlRunContext | None = None,
    ) -> list[RawArticle]:
        """Track A — 뉴스성 고빈도 수집."""
        return await self._run_track_sources(
            TRACK_A_SOURCES,
            keywords=keywords,
            persist=persist,
            crawl_window=crawl_window,
            recent_hours=recent_hours,
            run_context=run_context or CrawlRunContext(collection_mode="realtime", track="A"),
        )

    async def run_track_b(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
        run_context: CrawlRunContext | None = None,
    ) -> list[RawArticle]:
        """Track B — 마켓/증권 리포트 수집."""
        return await self._run_track_sources(
            TRACK_B_SOURCES,
            keywords=keywords,
            persist=persist,
            crawl_window=crawl_window,
            run_context=run_context or CrawlRunContext(collection_mode="realtime", track="B"),
        )

    async def run_track_c(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
        run_context: CrawlRunContext | None = None,
    ) -> list[RawArticle]:
        """Track C — 업무시간성 채용/기업 뉴스 수집."""
        return await self._run_track_sources(
            TRACK_C_SOURCES,
            keywords=keywords,
            persist=persist,
            crawl_window=crawl_window,
            run_context=run_context or CrawlRunContext(collection_mode="realtime", track="C"),
        )

    async def run_track_d(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
        run_context: CrawlRunContext | None = None,
    ) -> list[RawArticle]:
        """Track D — 문서/리포트/자사 사이트 저빈도 수집."""
        return await self._run_track_sources(
            TRACK_D_SOURCES,
            keywords=keywords,
            persist=persist,
            crawl_window=crawl_window,
            run_context=run_context or CrawlRunContext(collection_mode="realtime", track="D"),
        )

    async def _run_track_sources(
        self,
        source_names: tuple[str, ...],
        *,
        keywords: dict[str, list[str]],
        persist: bool,
        crawl_window: CrawlWindow | None,
        run_context: CrawlRunContext,
        recent_hours: int = 1,
    ) -> list[RawArticle]:
        articles: list[RawArticle] = []
        total_inserted = 0

        for source_name in source_names:
            source_context = replace(run_context, source_name=source_name)
            source_articles = await self.run_sources(
                (source_name,),
                keywords=keywords,
                persist=persist,
                crawl_window=crawl_window,
                run_context=source_context,
                recent_hours=recent_hours,
            )
            total_inserted += self.last_inserted_count
            articles.extend(source_articles)

        self.last_inserted_count = total_inserted
        return articles

    async def run_sources(
        self,
        source_names: list[str] | tuple[str, ...],
        *,
        keywords: dict[str, list[str]] | None = None,
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
        run_context: CrawlRunContext | None = None,
        recent_hours: int = 1,
    ) -> list[RawArticle]:
        """Source 이름 목록을 실행한다. realtime/backfill 공통 진입점."""
        from src.crawler.sources.company_news import CompanyNewsCrawler
        from src.crawler.sources.dart import DartCrawler
        from src.crawler.sources.global_newsroom import GlobalNewsroomCrawler
        from src.crawler.sources.ir import IRCrawler
        from src.crawler.sources.jobs import JobCrawler
        from src.crawler.sources.keyword import KeywordCrawler
        from src.crawler.sources.naver import NaverNewsCrawler
        from src.crawler.sources.naver_research import NaverResearchCrawler

        requested = set(source_names)
        effective_window = _effective_source_window(source_names, crawl_window, run_context)
        keywords = keywords or {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}
        articles: list[RawArticle] = []
        domestic_keywords = {
            peer_id: kws for peer_id, kws in keywords.items() if peer_id not in GLOBAL_COMPANY_IDS
        }
        global_company_ids = [peer_id for peer_id in keywords if peer_id in GLOBAL_COMPANY_IDS]

        if _should_run_keyword_sector_runner(requested, persist, run_context):
            return await self._run_keyword_sector_runner(
                source_names=source_names,
                run_context=run_context,
                effective_window=effective_window,
            )

        if "naver_news" in requested:
            naver_errors: list[str] = []
            cutoff = effective_window.start if effective_window else _hours_cutoff(recent_hours)
            for peer_id, kws in domestic_keywords.items():
                naver_crawler = NaverNewsCrawler(
                    peer_id=peer_id,
                    aliases=kws,
                    cutoff_datetime=cutoff,
                    end_datetime=effective_window.end if effective_window else None,
                )
                try:
                    articles.extend(await naver_crawler.crawl())
                except Exception as e:
                    naver_errors.append(f"{peer_id}: {type(e).__name__}: {e}")
                    log.error(
                        "source 크롤 오류 | source=naver_news company=%s error=%s",
                        peer_id,
                        e,
                    )
            if (
                run_context
                and run_context.collection_mode == "backfill"
                and effective_window
                and naver_errors
            ):
                raise RuntimeError(
                    "naver_news backfill failed; cursor not advanced. "
                    + "; ".join(naver_errors[:5])
                )

        for peer_id, kws in domestic_keywords.items():
            crawlers: list[tuple[str, _Crawlable]] = []
            if "dart" in requested:
                crawlers.append(
                    (
                        "dart",
                        DartCrawler(
                            peer_id=peer_id,
                            corp_code=CORP_CODES.get(peer_id, ""),
                            corp_names=kws,
                            start_date=_window_date(effective_window, "start"),
                            end_date=_window_date(effective_window, "end"),
                        ),
                    )
                )
            if "ir" in requested:
                crawlers.append(
                    (
                        "ir",
                        IRCrawler(
                            peer_id=peer_id,
                            lookback_days=_window_lookback_days(effective_window),
                            start_date=_window_date(effective_window, "start"),
                            end_date=_window_date(effective_window, "end"),
                        ),
                    )
                )
            if "naver_research" in requested:
                crawlers.append(
                    (
                        "naver_research",
                        NaverResearchCrawler(
                            peer_id=peer_id,
                            lookback_days=_window_lookback_days(effective_window),
                            start_date=_window_date(effective_window, "start"),
                            end_date=_window_date(effective_window, "end"),
                        ),
                    )
                )
            if "jobs" in requested:
                crawlers.append(
                    (
                        "jobs",
                        JobCrawler(
                            peer_id=peer_id,
                            start_date=_window_date(effective_window, "start"),
                            end_date=_window_date(effective_window, "end"),
                        ),
                    )
                )
            if "stock" in requested:
                from src.crawler.sources.stock import StockCrawler

                crawlers.append(
                    (
                        "stock",
                        StockCrawler(
                            peer_id=peer_id,
                            start_date=_window_date(effective_window, "start"),
                            end_date=_window_date(effective_window, "end"),
                            include_realtime=(
                                run_context is not None
                                and run_context.collection_mode == "realtime"
                            ),
                        ),
                    )
                )

            for source_name, crawler in crawlers:
                try:
                    articles.extend(_normalize_articles(await crawler.crawl()))
                except Exception as e:
                    log.error(
                        "source 크롤 오류 | source=%s company=%s crawler=%s error=%s",
                        source_name,
                        peer_id,
                        type(crawler).__name__,
                        e,
                    )

        # 아래 크롤러들은 내부에서 전체 company/industry를 처리하므로 1회만 호출
        shared_crawlers: list[tuple[str, _Crawlable]] = []
        if "company_news" in requested:
            shared_crawlers.append(
                (
                    "company_news",
                    CompanyNewsCrawler(
                        crawl_window=effective_window,
                        latest_limit=100 if effective_window else 5,
                    ),
                )
            )
        if "naver_datalab" in requested:
            shared_crawlers.append(
                (
                    "naver_datalab",
                    KeywordCrawler(
                        start_date=_window_iso(effective_window, "start"),
                        end_date=_window_iso(effective_window, "end"),
                    ),
                )
            )
        if "global_newsroom" in requested and global_company_ids:
            is_realtime = bool(run_context and run_context.collection_mode == "realtime")
            shared_crawlers.extend(
                (
                    f"global_newsroom[{company_id}]",
                    GlobalNewsroomCrawler(
                        company=company_id,
                        start_date=None if is_realtime else _window_date(effective_window, "start"),
                        end_date=None if is_realtime else _window_date(effective_window, "end"),
                    ),
                )
                for company_id in global_company_ids
            )
        elif "global_newsroom" in requested and not keywords:
            is_realtime = bool(run_context and run_context.collection_mode == "realtime")
            shared_crawlers.append(
                (
                    "global_newsroom",
                    GlobalNewsroomCrawler(
                        start_date=None if is_realtime else _window_date(effective_window, "start"),
                        end_date=None if is_realtime else _window_date(effective_window, "end"),
                    ),
                )
            )
        if "spri" in requested:
            from src.crawler.sources.spri import SpriCrawler

            shared_crawlers.extend(
                (
                    f"spri[{month}]",
                    SpriCrawler(
                        month=month,
                        output_path=DEFAULT_RESULTS_DIR / f"spri_backfill_{month}.json",
                    ),
                )
                for month in _window_months(effective_window)
            )
        if "bcg" in requested:
            from src.crawler.sources.bcg import BcgCrawler

            shared_crawlers.append(
                (
                    "bcg",
                    BcgCrawler(
                        days=_window_lookback_days(effective_window) or 7,
                        max_articles=100,
                        output_path=DEFAULT_RESULTS_DIR / "bcg_backfill.json",
                        start_date=_window_date(effective_window, "start"),
                        end_date=_window_date(effective_window, "end"),
                    ),
                )
            )
        if "sk_ax_site" in requested:
            from src.crawler.sources.skax_crawler import SkaxSiteCrawler

            shared_crawlers.append(
                (
                    "sk_ax_site",
                    SkaxSiteCrawler(
                        max_pages=200,
                        output_path=DEFAULT_RESULTS_DIR / "skax_site_backfill.json",
                    ),
                )
            )

        for name, shared in shared_crawlers:
            try:
                articles.extend(_normalize_articles(await shared.crawl()))
            except Exception as e:
                log.error("source 크롤 오류 | source=%s error=%s", name, e)

        run_id: UUID | None = None
        effective_run_context = run_context
        if persist and run_context and not run_context.crawl_run_id:
            run_source_name = run_context.source_name or ",".join(source_names)
            window_start = _window_date(effective_window, "start") or datetime.now().date()
            window_end = _window_date(effective_window, "end") or window_start
            run_id = create_crawl_run(
                run_source_name,
                window_start,
                window_end,
                run_type=run_context.collection_mode,
            )
            self.last_crawl_run_ids.append(str(run_id))
            effective_run_context = replace(
                run_context,
                crawl_run_id=str(run_id),
                source_name=run_source_name,
                window_start=effective_window.start if effective_window else None,
                window_end=effective_window.end if effective_window else None,
            )
            self.last_crawl_run_records.append(
                {
                    "crawl_run_id": str(run_id),
                    "source_name": run_source_name,
                    "track": effective_run_context.track or "",
                }
            )

        try:
            articles = _filter_window(articles, effective_window)
            accessible, rejected = await self.link_checker.filter_accessible(articles)
            new_articles = self.dedup.filter_new(accessible)
            inserted = (
                save_articles(new_articles, run_context=effective_run_context) if persist else 0
            )
            self.last_inserted_count = inserted
            if run_id:
                mark_crawl_run_success(
                    run_id,
                    inserted_count=inserted,
                    skipped_count=len(new_articles) - inserted,
                )
        except Exception as e:
            if run_id:
                mark_crawl_run_failed(run_id, f"{type(e).__name__}: {e}")
            raise
        log.info(
            "source 크롤 완료 | sources=%s raw=%d accessible=%d "
            "rejected_links=%d new=%d db_inserted=%d",
            ",".join(source_names),
            len(articles),
            len(accessible),
            len(rejected),
            len(new_articles),
            inserted,
        )
        return new_articles

    async def _run_keyword_sector_runner(
        self,
        *,
        source_names: list[str] | tuple[str, ...],
        run_context: CrawlRunContext | None,
        effective_window: CrawlWindow | None,
    ) -> list[RawArticle]:
        from src.crawler.sources.keyword_sector_runner import run_scheduled

        run_id = None
        run_source_name = (
            run_context.source_name
            if run_context and run_context.source_name
            else ",".join(source_names)
        )
        window_start = _window_date(effective_window, "start") or datetime.now().date()
        window_end = _window_date(effective_window, "end") or window_start
        run_type = run_context.collection_mode if run_context else "realtime"
        track = run_context.track if run_context else ""

        try:
            if run_context and run_context.crawl_run_id:
                run_id = UUID(run_context.crawl_run_id)
            else:
                run_id = create_crawl_run(
                    run_source_name,
                    window_start,
                    window_end,
                    run_type=run_type,
                )
                self.last_crawl_run_ids.append(str(run_id))
                self.last_crawl_run_records.append(
                    {
                        "crawl_run_id": str(run_id),
                        "source_name": run_source_name,
                        "track": track or "",
                    }
                )

            inserted = await asyncio.to_thread(run_scheduled, crawl_run_id=str(run_id))
            self.last_inserted_count = inserted
            mark_crawl_run_success(run_id, inserted_count=inserted, skipped_count=0)
        except Exception as e:
            self.last_inserted_count = 0
            if run_id:
                mark_crawl_run_failed(run_id, f"{type(e).__name__}: {e}")
            raise

        log.info(
            "source 크롤 완료 | sources=%s mode=keyword_sector_runner db_inserted=%d",
            ",".join(source_names),
            inserted,
        )
        return []


def _hours_cutoff(hours: int) -> datetime | None:
    if hours <= 0:
        return None
    return datetime.now().astimezone() - timedelta(hours=hours)


def _should_run_keyword_sector_runner(
    requested: set[str],
    persist: bool,
    run_context: CrawlRunContext | None,
) -> bool:
    if requested != {"naver_datalab"}:
        return False
    if not persist:
        return False
    return run_context is None or run_context.collection_mode == "realtime"


def _filter_window(
    articles: list[RawArticle],
    crawl_window: CrawlWindow | None,
) -> list[RawArticle]:
    if not crawl_window:
        return articles
    return [
        article
        for article in articles
        if (
            article.published_at is None
            and article.source_type == "company_site"
            and article.extra.get("source_family") == "sk_ax_site"
        )
        or crawl_window.contains(article.published_at)
    ]


def _effective_source_window(
    source_names: list[str] | tuple[str, ...],
    crawl_window: CrawlWindow | None,
    run_context: CrawlRunContext | None,
) -> CrawlWindow | None:
    if not crawl_window:
        return None
    if not run_context or run_context.collection_mode != "realtime":
        return crawl_window

    overlap_days = max(
        (REALTIME_SOURCE_OVERLAP_DAYS.get(source_name, 0) for source_name in source_names),
        default=0,
    )
    if overlap_days <= 0:
        return crawl_window

    end = crawl_window.end or datetime.now().astimezone()
    overlap_start = end - timedelta(days=overlap_days)
    start = overlap_start if overlap_start < crawl_window.start else crawl_window.start
    return CrawlWindow(start=start, end=end)


def _window_date(crawl_window: CrawlWindow | None, bound: str) -> date | None:
    if not crawl_window:
        return None
    value = crawl_window.start if bound == "start" else crawl_window.end
    return value.date() if value else None


def _window_iso(crawl_window: CrawlWindow | None, bound: str) -> str | None:
    value = _window_date(crawl_window, bound)
    return value.isoformat() if value else None


def _window_lookback_days(crawl_window: CrawlWindow | None) -> int | None:
    if not crawl_window:
        return None
    start = crawl_window.start.date()
    end = (crawl_window.end or datetime.now().astimezone()).date()
    return max((end - start).days, 1)


def _window_month(crawl_window: CrawlWindow | None) -> str:
    target = (crawl_window.end or crawl_window.start) if crawl_window else datetime.now()
    return target.strftime("%Y-%m")


def _window_months(crawl_window: CrawlWindow | None) -> list[str]:
    if not crawl_window:
        return [_window_month(None)]

    start = crawl_window.start.date().replace(day=1)
    end_bound = crawl_window.end or crawl_window.start
    end = end_bound.date().replace(day=1)

    months: list[str] = []
    cursor = start
    while cursor <= end:
        months.append(cursor.strftime("%Y-%m"))
        cursor = _next_month(cursor)
    return months


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _normalize_articles(items: Sequence[object]) -> list[RawArticle]:
    return [_to_raw_article(item) for item in items]


def _to_raw_article(item: object) -> RawArticle:
    if isinstance(item, RawArticle):
        return item

    if hasattr(item, "to_common_dict"):
        data = item.to_common_dict()
    elif isinstance(item, dict):
        data = item
    else:
        raise TypeError(f"RawArticle로 변환할 수 없는 크롤 결과입니다: {type(item).__name__}")

    extra = dict(data.get("extra") or data.get("metadata") or {})
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
        published_at=_parse_datetime(data.get("published_at")),
        collected_at=_parse_datetime(data.get("collected_at")) or datetime.now().astimezone(),
        source_type=data.get("source_type") or extra.get("source_type") or "news",
        content_type=data.get("content_type") or extra.get("content_type") or "html",
        publisher=data.get("publisher"),
        company=_company_list(data.get("company")),
        language=data.get("language") or "ko",
        crawl_status=data.get("crawl_status") or "success",
        error_message=data.get("error_message"),
        url_hash=data.get("url_hash") or "",
        extra=extra,
        peer_id=data.get("peer_id"),
    )


def _parse_datetime(value: object) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min).astimezone()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _company_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
        return [value] if value else []
    return []
