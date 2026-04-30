"""Track A/B 크롤 오케스트레이터 — 원천 수집 + URL 중복 제거 + 저장."""

import logging
from typing import Protocol

from src.crawler.base import DailyLimitGuard, RawArticle
from src.crawler.parsers.dedup import DedupStore
from src.crawler.parsers.link_check import LinkChecker
from src.db.article_store import save_articles

log = logging.getLogger(__name__)


class _Crawlable(Protocol):
    async def crawl(self) -> list[RawArticle]: ...


# DART 법인코드 (금융감독원 전자공시시스템 기준)
CORP_CODES: dict[str, str] = {
    "sk_ax": "00111722",  # SK Inc
    "samsung_sds": "00126186",  # 삼성에스디에스 (주식코드 018260)
    "lg_cns": "00139834",  # LG씨엔에스 (주식코드 064400)
    "hyundai_autoever": "00362441",  # 현대오토에버 (주식코드 307950)
    "posco_dx": "00155212",  # 포스코DX (주식코드 022100)
}


class BatchProcessor:
    def __init__(self) -> None:
        self.limit_guard = DailyLimitGuard()
        self.dedup = DedupStore()
        self.link_checker = LinkChecker()

    async def run_track_a(self, keywords: dict[str, list[str]]) -> list[RawArticle]:
        """Track A — Naver, RSS, Google News (1시간 간격)."""
        from src.crawler.sources.naver import NaverNewsCrawler
        from src.crawler.sources.rss import GoogleNewsRssCrawler, RssCrawler

        articles: list[RawArticle] = []
        for peer_id, kws in keywords.items():
            for crawler in [
                NaverNewsCrawler(peer_id, kws, self.limit_guard),
                RssCrawler(peer_id, kws, self.limit_guard),
                GoogleNewsRssCrawler(peer_id, kws, self.limit_guard),
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
        inserted = save_articles(new_articles)
        log.info(
            "Track A 완료 | raw=%d accessible=%d rejected_links=%d new=%d db_inserted=%d",
            len(articles),
            len(accessible),
            len(rejected),
            len(new_articles),
            inserted,
        )
        return new_articles

    async def run_track_b(self, keywords: dict[str, list[str]]) -> list[RawArticle]:
        """Track B — DART, KIPRIS, 공식 뉴스룸, 채용공고 (매일 새벽 2시)."""
        from src.crawler.sources.consensus import HankyungConsensusCrawler
        from src.crawler.sources.dart import DartCrawler
        from src.crawler.sources.jobs import JobsCrawler
        from src.crawler.sources.kipris import KiprisCrawler
        from src.crawler.sources.naver_research import NaverResearchCrawler
        from src.crawler.sources.official import OfficialNewsroomCrawler

        articles: list[RawArticle] = []
        for peer_id, kws in keywords.items():
            for crawler in [
                DartCrawler(peer_id, CORP_CODES.get(peer_id, ""), self.limit_guard),
                OfficialNewsroomCrawler(peer_id, self.limit_guard),
                JobsCrawler(peer_id, kws, self.limit_guard),
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

        # 아래 크롤러들은 내부에서 두 peer_id 모두 처리하므로 1회만 호출
        shared_crawlers: list[tuple[str, _Crawlable]] = [
            ("KiprisCrawler", KiprisCrawler(self.limit_guard)),
            ("HankyungConsensusCrawler", HankyungConsensusCrawler(self.limit_guard)),
            ("NaverResearchCrawler", NaverResearchCrawler(self.limit_guard)),
        ]
        for name, shared in shared_crawlers:
            try:
                articles.extend(await shared.crawl())
            except Exception as e:
                log.error("Track B 크롤 오류 | crawler=%s error=%s", name, e)

        accessible, rejected = await self.link_checker.filter_accessible(articles)
        new_articles = self.dedup.filter_new(accessible)
        inserted = save_articles(new_articles)
        log.info(
            "Track B 완료 | raw=%d accessible=%d rejected_links=%d new=%d db_inserted=%d",
            len(articles),
            len(accessible),
            len(rejected),
            len(new_articles),
            inserted,
        )
        return new_articles
