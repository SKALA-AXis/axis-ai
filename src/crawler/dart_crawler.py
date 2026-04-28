"""DART 공시 크롤러"""

import logging
import os

from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)


class DartCrawler(BaseCrawler):
    def __init__(self, peer_id: str, corp_code: str):
        super().__init__(peer_id)
        self.corp_code = corp_code
        self.api_key = os.getenv("DART_API_KEY", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.api_key:
            log.warning("DART_API_KEY 미설정. 크롤링 스킵.")
            return []
        # TODO: DART OpenAPI /api/list.json 호출
        return []
