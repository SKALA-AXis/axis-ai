"""네이버 뉴스 API 크롤러"""
import logging
import os
from datetime import datetime

import httpx

from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverNewsCrawler(BaseCrawler):
    def __init__(self, peer_id: str, keywords: list[str]):
        super().__init__(peer_id)
        self.keywords = keywords
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.client_id:
            log.warning("NAVER_CLIENT_ID 미설정. 크롤링 스킵.")
            return []
        articles = []
        for keyword in self.keywords:
            try:
                articles.extend(await self._fetch(keyword))
            except Exception as e:
                log.error("네이버 크롤링 실패 | keyword=%s error=%s", keyword, e)
        return articles

    async def _fetch(self, keyword: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                NAVER_API_URL,
                params={"query": keyword, "display": 20, "sort": "date"},
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
                    title=item["title"].replace("<b>", "").replace("</b>", ""),
                    content=item.get("description", ""),
                    published_at=datetime.strptime(
                        item["pubDate"], "%a, %d %b %Y %H:%M:%S +0900"
                    ),
                    source_name="naver_news",
                    peer_id=self.peer_id,
                )
                for item in items
            ]
