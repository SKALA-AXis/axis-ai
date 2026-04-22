"""Track A/B 크롤 오케스트레이터 — FastFilter + DedupStore 적용."""

import logging

from src.crawler.base import DailyLimitGuard, RawArticle
from src.crawler.fast_filter import FastFilter
from src.crawler.parsers.dedup import DedupStore
from src.db.article_store import save_articles

log = logging.getLogger(__name__)

# DART 법인코드 (금융감독원 전자공시시스템 기준)
CORP_CODES: dict[str, str] = {
    "samsung_sds": "00126186",  # 삼성에스디에스
    "lg_cns": "00139834",  # LG씨엔에스 (주식코드 064400)
}


class BatchProcessor:
    def __init__(self) -> None:
        self.limit_guard = DailyLimitGuard()
        self.fast_filter = FastFilter()
        self.dedup = DedupStore()

    async def run_track_a(self, keywords: dict[str, list[str]]) -> list[RawArticle]:
        """Track A — Naver, RSS, BigKinds, Google News (1시간 간격)."""
        from src.crawler.sources.bigkinds import BigKindsCrawler
        from src.crawler.sources.naver import NaverNewsCrawler
        from src.crawler.sources.rss import GoogleNewsRssCrawler, RssCrawler

        articles: list[RawArticle] = []
        # BigKindsCrawler는 내부에서 두 peer_id 모두 처리하므로 1회만 호출
        try:
            all_kws = [kw for kws in keywords.values() for kw in kws]
            articles.extend(await BigKindsCrawler(self.limit_guard).crawl(all_kws))
        except Exception as e:
            log.error("Track A 크롤 오류 | crawler=BigKindsCrawler error=%s", e)

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

        filtered, _ = self.fast_filter.filter(articles)
        new_articles = self.dedup.filter_new(filtered)
        inserted = save_articles(new_articles)
        log.info(
            "Track A 완료 | raw=%d filtered=%d new=%d db_inserted=%d",
            len(articles),
            len(filtered),
            len(new_articles),
            inserted,
        )
        return new_articles

    async def run_track_b(self, keywords: dict[str, list[str]]) -> list[RawArticle]:
        """Track B — DART, KIPRIS, 공식 뉴스룸, 채용공고 (매일 새벽 2시)."""
        from src.crawler.sources.dart import DartCrawler
        from src.crawler.sources.jobs import JobsCrawler
        from src.crawler.sources.kipris import KiprisCrawler
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

        # KiprisCrawler는 내부에서 두 peer_id 모두 처리하므로 1회만 호출
        try:
            articles.extend(await KiprisCrawler(self.limit_guard).crawl())
        except Exception as e:
            log.error("Track B 크롤 오류 | crawler=KiprisCrawler error=%s", e)

        filtered, _ = self.fast_filter.filter(articles)
        new_articles = self.dedup.filter_new(filtered)
        inserted = save_articles(new_articles)
        log.info(
            "Track B 완료 | raw=%d filtered=%d new=%d db_inserted=%d",
            len(articles),
            len(filtered),
            len(new_articles),
            inserted,
        )
        return new_articles
