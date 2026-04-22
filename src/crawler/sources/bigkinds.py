"""BigKinds 뉴스 빅데이터 크롤러 (Tier 2) — REST API 요약 + Playwright 본문 2단계."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from src.crawler.base import (
    RETRY_POLICY,
    SOURCE_CREDIBILITY,
    DailyLimitGuard,
    RawArticle,
)
from src.crawler.playwright_client import PlaywrightClient

log = logging.getLogger(__name__)

BIGKINDS_API_URL = "https://tools.kinds.or.kr/search/news"

# 신뢰도 높은 IT 전문 매체 코드
_PROVIDER_CODES = [
    "02100311",  # 전자신문
    "02100117",  # ZDNet Korea
    "02100115",  # 디지털타임스
    "02100271",  # 아이뉴스24
    "02100305",  # 블로터
    "02100270",  # 데이터넷
]


class BigKindsCrawler:
    """
    BigKinds REST API로 요약 수집 →
    경량 분류기에서 urgent·notable 판정된 기사만 Playwright로 본문 전문 수집.
    """

    def __init__(self, limit_guard: Optional[DailyLimitGuard] = None) -> None:
        self.api_key = os.getenv("BIGKINDS_API_KEY", "")
        self.pw = PlaywrightClient()
        self.limit_guard = limit_guard or DailyLimitGuard()

    async def crawl(self, keywords: Optional[list[str]] = None) -> list[RawArticle]:
        if not self.api_key:
            log.warning("BIGKINDS_API_KEY 미설정. 크롤링 스킵.")
            return []
        if not self.limit_guard.allow("bigkinds"):
            return []

        query = " OR ".join(keywords) if keywords else "삼성SDS OR LG CNS"
        try:
            return await self._search(query)
        except Exception as e:
            log.error("BigKinds 크롤링 실패 | error=%s", e)
            return []

    async def _search(self, query: str) -> list[RawArticle]:
        today = datetime.now()
        body: dict[str, Any] = {
            "access_key": self.api_key,
            "argument": {
                "query": query,
                "published_at": {
                    "from": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "until": today.strftime("%Y-%m-%d"),
                },
                "provider_code": _PROVIDER_CODES,
                "result_fields": [
                    "title",
                    "content",
                    "published_at",
                    "provider_link_page",
                    "provider_name",
                ],
                "sort": {"date": "desc"},
                "return_from": 0,
                "return_size": 100,
            },
        }

        _timeout = RETRY_POLICY["source_timeout"]["backoff"][0]
        async with httpx.AsyncClient(timeout=_timeout) as client:
            resp = await client.post(BIGKINDS_API_URL, json=body)
            if resp.status_code in (403, 429):
                log.warning("BigKinds API 접근 차단 | status=%d", resp.status_code)
                return []
            resp.raise_for_status()
            hits = resp.json().get("return_object", {}).get("documents", [])

        return [
            RawArticle(
                url=hit.get("provider_link_page", ""),
                title=hit.get("title", ""),
                content=hit.get("content", "")[:500],  # 빅카인즈 요약 (~3줄)
                source_tier=2,
                source_name=hit.get("provider_name", "bigkinds"),
                credibility_score=SOURCE_CREDIBILITY["bigkinds"],
                published_at=_parse_date(hit.get("published_at", "")),
                metadata={"provider_name": hit.get("provider_name"), "full_content_pending": True},
            )
            for hit in hits
        ]

    async def fetch_full_content(self, article: RawArticle) -> str:
        """
        2단계: FastFilter에서 urgent·notable 판정된 기사만 호출.
        Playwright로 본문 전문 수집 (매체 내 차단 우회).
        실패 시 빅카인즈 요약으로 폴백.
        """
        if not article.url:
            return article.content
        try:
            full = await self.pw.fetch_markdown(article.url)
            return full if full else article.content
        except Exception as e:
            log.warning("BigKinds 본문 크롤링 실패, 요약 사용 | url=%s error=%s", article.url, e)
            return article.content


def _parse_date(date_str: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except ValueError:
            continue
    return None
