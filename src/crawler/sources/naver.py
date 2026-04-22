"""Naver News API 크롤러 (Tier 1) — REST API, 1시간 간격."""

import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from src.crawler.base import (
    RETRY_POLICY,
    SOURCE_CREDIBILITY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverNewsCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        keywords: list[str],
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.keywords = keywords
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

    async def crawl(self) -> list[RawArticle]:
        if not self.client_id:
            log.warning("NAVER_CLIENT_ID 미설정. 크롤링 스킵.")
            return []
        articles: list[RawArticle] = []
        for keyword in self.keywords:
            if not self.limit_guard.allow("naver_news"):
                break
            try:
                articles.extend(await self._fetch(keyword))
            except Exception as e:
                log.error("네이버 크롤링 실패 | keyword=%s error=%s", keyword, e)
        return articles

    async def _fetch(self, keyword: str) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=RETRY_POLICY["timeout"]) as client:
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
                    content=item.get("description", "").replace("<b>", "").replace("</b>", ""),
                    published_at=_parse_naver_date(item.get("pubDate", "")),
                    source_name="naver_news",
                    peer_id=self.peer_id,
                    credibility_score=SOURCE_CREDIBILITY["naver_news"],
                    source_tier=1,
                )
                for item in items
            ]


def _parse_naver_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S +0900")
    except ValueError:
        return None
