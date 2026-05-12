"""Track A/B 크롤 오케스트레이터 — 원천 수집 + URL 중복 제거 + 저장."""

import json
import logging
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

from src.config.companies import COMPANY_ALIASES, CORP_CODES
from src.config.global_companies import GLOBAL_COMPANY_ALIASES, GLOBAL_COMPANY_IDS
from src.crawler.base import CrawlRunContext, CrawlWindow, DailyLimitGuard, RawArticle
from src.crawler.parsers.dedup import DedupStore
from src.crawler.parsers.link_check import LinkChecker
from src.crawler.result_writer import DEFAULT_RESULTS_DIR
from src.db.article_store import save_articles

log = logging.getLogger(__name__)

TRACK_A_SOURCES = ("naver_news",)
TRACK_B_SOURCES = (
    "dart",
    "ir",
    "naver_research",
    "jobs",
    "company_news",
    "global_newsroom",
    "naver_datalab",
    "stock",
    "spri",
    "bcg",
)


class _Crawlable(Protocol):
    async def crawl(self) -> list[Any]: ...


class BatchProcessor:
    def __init__(self) -> None:
        self.limit_guard = DailyLimitGuard()
        self.dedup = DedupStore()
        self.link_checker = LinkChecker()
        self.last_inserted_count = 0

    async def run_track_a(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        recent_hours: int = 1,
    ) -> list[RawArticle]:
        """Track A — Naver News (1시간 간격)."""
        return await self.run_sources(
            TRACK_A_SOURCES,
            keywords=keywords,
            persist=persist,
            recent_hours=recent_hours,
        )

    async def run_track_b(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
    ) -> list[RawArticle]:
        """Track B — DART, IR, 리서치, 공식 뉴스룸, 채용공고, 검색 트렌드."""
        return await self.run_sources(
            TRACK_B_SOURCES,
            keywords=keywords,
            persist=persist,
            crawl_window=crawl_window,
        )

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
        keywords = keywords or {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}
        articles: list[RawArticle] = []
        domestic_keywords = {
            peer_id: kws for peer_id, kws in keywords.items() if peer_id not in GLOBAL_COMPANY_IDS
        }
        global_company_ids = [peer_id for peer_id in keywords if peer_id in GLOBAL_COMPANY_IDS]

        if "naver_news" in requested:
            cutoff = crawl_window.start if crawl_window else _hours_cutoff(recent_hours)
            for peer_id, kws in domestic_keywords.items():
                naver_crawler = NaverNewsCrawler(
                    peer_id=peer_id,
                    aliases=kws,
                    cutoff_datetime=cutoff,
                )
                try:
                    articles.extend(await naver_crawler.crawl())
                except Exception as e:
                    log.error(
                        "source 크롤 오류 | source=naver_news company=%s error=%s",
                        peer_id,
                        e,
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
                            start_date=_window_date(crawl_window, "start"),
                            end_date=_window_date(crawl_window, "end"),
                        ),
                    )
                )
            if "ir" in requested:
                crawlers.append(
                    (
                        "ir",
                        IRCrawler(
                            peer_id=peer_id,
                            lookback_days=_window_lookback_days(crawl_window),
                        ),
                    )
                )
            if "naver_research" in requested:
                crawlers.append(
                    (
                        "naver_research",
                        NaverResearchCrawler(
                            peer_id=peer_id,
                            lookback_days=_window_lookback_days(crawl_window),
                        ),
                    )
                )
            if "jobs" in requested:
                crawlers.append(("jobs", JobCrawler(peer_id=peer_id)))
            if "stock" in requested:
                from src.crawler.sources.stock import StockCrawler

                crawlers.append(
                    (
                        "stock",
                        StockCrawler(
                            peer_id=peer_id,
                            start_date=_window_date(crawl_window, "start"),
                            end_date=_window_date(crawl_window, "end"),
                            include_realtime=not crawl_window,
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
                        crawl_window=crawl_window,
                        latest_limit=20 if crawl_window else 5,
                    ),
                )
            )
        if "naver_datalab" in requested:
            shared_crawlers.append(
                (
                    "naver_datalab",
                    KeywordCrawler(
                        start_date=_window_iso(crawl_window, "start"),
                        end_date=_window_iso(crawl_window, "end"),
                    ),
                )
            )
        if "global_newsroom" in requested and global_company_ids:
            shared_crawlers.extend(
                (
                    f"global_newsroom[{company_id}]",
                    GlobalNewsroomCrawler(company=company_id),
                )
                for company_id in global_company_ids
            )
        elif "global_newsroom" in requested and not keywords:
            shared_crawlers.append(("global_newsroom", GlobalNewsroomCrawler()))
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
                for month in _window_months(crawl_window)
            )
        if "bcg" in requested:
            from src.crawler.sources.bcg import BcgCrawler

            shared_crawlers.append(
                (
                    "bcg",
                    BcgCrawler(
                        days=_window_lookback_days(crawl_window) or 7,
                        max_articles=100,
                        output_path=DEFAULT_RESULTS_DIR / "bcg_backfill.json",
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

        articles = _filter_window(articles, crawl_window)
        accessible, rejected = await self.link_checker.filter_accessible(articles)
        new_articles = self.dedup.filter_new(accessible)
        inserted = save_articles(new_articles, run_context=run_context) if persist else 0
        self.last_inserted_count = inserted
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


def _hours_cutoff(hours: int) -> datetime | None:
    if hours <= 0:
        return None
    return datetime.now().astimezone() - timedelta(hours=hours)


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
    today = datetime.now().astimezone().date()
    return max((today - start).days, 1)


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
