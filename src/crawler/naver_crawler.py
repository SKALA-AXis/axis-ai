"""네이버 뉴스 API 크롤러"""

import logging
import os
from email.utils import parsedate_to_datetime

import httpx

from src.crawler.article_filter import strip_html
from src.crawler.base import RawArticle
from src.crawler.base_crawler import BaseCrawler

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverNewsCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        aliases: list[str],
        search_queries: list[dict[str, str]] | None = None,
        display: int = 2,
    ):
        super().__init__(peer_id)
        self.aliases = aliases
        self.search_queries = search_queries
        self.display = display
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.client_id:
            log.warning("NAVER_CLIENT_ID 미설정. 크롤링 스킵.")
            return []
        articles = []
        query_specs = self.search_queries or [
            {"query": alias, "sector": ""} for alias in self.aliases
        ]
        for spec in query_specs:
            try:
                articles.extend(
                    await self._fetch(query=spec["query"], sector=spec.get("sector", ""))
                )
            except Exception as e:
                log.error("네이버 크롤링 실패 | query=%s error=%s", spec["query"], e)
        return articles

    async def _fetch(self, query: str, sector: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                NAVER_API_URL,
                params={"query": query, "display": self.display, "sort": "date"},
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
                    sector=sector,
                    search_query=query,
                )
                for item in items
            ]
