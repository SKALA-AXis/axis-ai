"""네이버 뉴스 API 크롤러"""

import logging
import os
from email.utils import parsedate_to_datetime

import httpx

from src.crawler.article_filter import strip_html
from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverNewsCrawler(BaseCrawler):
    def __init__(self, peer_id: str, aliases: list[str]):
        super().__init__(peer_id)
        self.aliases = aliases
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.client_id:
            log.warning("NAVER_CLIENT_ID 미설정. 크롤링 스킵.")
            return []
        articles = []
        for alias in self.aliases:
            try:
                articles.extend(await self._fetch(alias=alias))
            except Exception as e:
                log.error("네이버 크롤링 실패 | alias=%s error=%s", alias, e)
        return articles

    async def _fetch(self, alias: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                NAVER_API_URL,
                params={"query": alias, "display": 50, "sort": "date"},
                headers={
                    "X-Naver-Client-Id": self.client_id,
                    "X-Naver-Client-Secret": self.client_secret,
                },
            )
            if self._is_blocked(resp.status_code):
                log.warning("네이버 API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [
                RawArticle(
                    url=item["link"],
                    title=strip_html(item["title"]),
                    content=strip_html(item.get("description", "")),
                    published_at=parsedate_to_datetime(item["pubDate"]),
                    source_name="naver_news",
                    peer_id=self.peer_id,
                )
                for item in items
            ]
