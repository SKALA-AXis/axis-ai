"""Track A/B 크롤 오케스트레이터 — 원천 수집 + URL 중복 제거 + 저장."""

import logging
from datetime import datetime, timedelta
from typing import Protocol

from src.config.companies import CORP_CODES
from src.config.global_companies import GLOBAL_COMPANY_IDS
from src.crawler.base import CrawlWindow, DailyLimitGuard, RawArticle
from src.crawler.parsers.dedup import DedupStore
from src.crawler.parsers.link_check import LinkChecker
from src.db.article_store import save_articles

log = logging.getLogger(__name__)


class _Crawlable(Protocol):
    async def crawl(self) -> list[RawArticle]: ...


class BatchProcessor:
    def __init__(self) -> None:
        self.limit_guard = DailyLimitGuard()
        self.dedup = DedupStore()
        self.link_checker = LinkChecker()

    async def run_track_a(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        recent_hours: int = 1,
    ) -> list[RawArticle]:
        """Track A — Naver News (1시간 간격)."""
        from src.crawler.sources.naver import NaverNewsCrawler

        cutoff_datetime = _hours_cutoff(recent_hours)
        articles: list[RawArticle] = []
        for peer_id, kws in keywords.items():
            for crawler in [
                NaverNewsCrawler(
                    peer_id=peer_id,
                    aliases=kws,
                    cutoff_datetime=cutoff_datetime,
                ),
            ]:
                try:
                    results = await crawler.crawl()
                    articles.extend(results)
                except Exception as e:
                    log.error(
                        "Track A 크롤 오류 | crawler=%s error=%s",
                        type(crawler).__name__,
                        e,
                    )

        accessible, rejected = await self.link_checker.filter_accessible(articles)
        new_articles = self.dedup.filter_new(accessible)
        inserted = save_articles(new_articles) if persist else 0
        log.info(
            "Track A 완료 | raw=%d accessible=%d rejected_links=%d new=%d db_inserted=%d",
            len(articles),
            len(accessible),
            len(rejected),
            len(new_articles),
            inserted,
        )
        return new_articles

    async def run_track_b(
        self,
        keywords: dict[str, list[str]],
        persist: bool = True,
        crawl_window: CrawlWindow | None = None,
    ) -> list[RawArticle]:
        """Track B — DART, IR, 리서치, 공식 뉴스룸, 채용공고, 검색 트렌드."""
        from src.crawler.sources.company_news import CompanyNewsCrawler
        from src.crawler.sources.dart import DartCrawler
        from src.crawler.sources.global_newsroom import GlobalNewsroomCrawler
        from src.crawler.sources.ir import IRCrawler
        from src.crawler.sources.jobs import JobCrawler
        from src.crawler.sources.keyword import KeywordCrawler
        from src.crawler.sources.naver_research import NaverResearchCrawler

        articles: list[RawArticle] = []
        domestic_keywords = {
            peer_id: kws for peer_id, kws in keywords.items() if peer_id not in GLOBAL_COMPANY_IDS
        }
        global_company_ids = [peer_id for peer_id in keywords if peer_id in GLOBAL_COMPANY_IDS]

        for peer_id, kws in domestic_keywords.items():
            for crawler in [
                DartCrawler(
                    peer_id=peer_id,
                    corp_code=CORP_CODES.get(peer_id, ""),
                    corp_names=kws,
                ),
                IRCrawler(peer_id=peer_id),
                NaverResearchCrawler(peer_id=peer_id),
                JobCrawler(peer_id=peer_id),
            ]:
                try:
                    results = await crawler.crawl()
                    articles.extend(results)
                except Exception as e:
                    log.error(
                        "Track B 크롤 오류 | crawler=%s error=%s",
                        type(crawler).__name__,
                        e,
                    )

        # 아래 크롤러들은 내부에서 전체 company/industry를 처리하므로 1회만 호출
        shared_crawlers: list[tuple[str, _Crawlable]] = [
            ("CompanyNewsCrawler", CompanyNewsCrawler()),
            ("KeywordCrawler", KeywordCrawler()),
        ]
        if global_company_ids:
            shared_crawlers.extend(
                (
                    f"GlobalNewsroomCrawler[{company_id}]",
                    GlobalNewsroomCrawler(company=company_id),
                )
                for company_id in global_company_ids
            )
        elif not keywords:
            shared_crawlers.append(("GlobalNewsroomCrawler", GlobalNewsroomCrawler()))

        for name, shared in shared_crawlers:
            try:
                articles.extend(await shared.crawl())
            except Exception as e:
                log.error("Track B 크롤 오류 | crawler=%s error=%s", name, e)

        accessible, rejected = await self.link_checker.filter_accessible(articles)
        new_articles = self.dedup.filter_new(accessible)
        inserted = save_articles(new_articles) if persist else 0
        log.info(
            "Track B 완료 | raw=%d accessible=%d rejected_links=%d new=%d db_inserted=%d",
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
