"""채용공고 크롤러"""
import logging

from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)


class JobCrawler(BaseCrawler):
    async def crawl(self) -> list[RawArticle]:
        """LinkedIn·잡플래닛 채용공고 수집 (주 1회)"""
        log.info("채용공고 수집 | peer_id=%s", self.peer_id)
        # TODO: 공식 API 우선, robots.txt 준수
        return []
