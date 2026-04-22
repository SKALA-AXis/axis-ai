"""RSS 피드 크롤러 (Tier 2) — ETnews, ZDNet, IT조선, 연합뉴스 + Google News RSS."""

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import feedparser

from src.crawler.base import (
    SOURCE_CREDIBILITY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

RSS_FEEDS: dict[str, dict[str, str]] = {
    "etnews": {
        "url": "https://www.etnews.com/rss/",
        "credibility_key": "etnews",
    },
    "zdnet": {
        "url": "https://zdnet.co.kr/rss/",
        "credibility_key": "zdnet",
    },
    "itchosun": {
        "url": "https://it.chosun.com/feed/rss/",
        "credibility_key": "itchosun",
    },
    "yonhap_tech": {
        "url": "https://www.yna.co.kr/rss/industry.xml",
        "credibility_key": "yonhap",
    },
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


class RssCrawler(BaseCrawler):
    """국내 IT 미디어 RSS 피드 크롤러."""

    def __init__(
        self,
        peer_id: str,
        keywords: list[str],
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.keywords = keywords

    async def crawl(self) -> list[RawArticle]:
        articles: list[RawArticle] = []
        for source_name, feed_info in RSS_FEEDS.items():
            if not self.limit_guard.allow("rss"):
                break
            try:
                feed = feedparser.parse(feed_info["url"])
                for entry in feed.entries:
                    title = entry.get("title", "")
                    if not any(kw.lower() in title.lower() for kw in self.keywords):
                        continue
                    articles.append(
                        RawArticle(
                            url=entry.get("link", ""),
                            title=title,
                            content=entry.get("summary", ""),
                            published_at=_parse_entry_date(entry),
                            source_name=source_name,
                            peer_id=self.peer_id,
                            credibility_score=SOURCE_CREDIBILITY.get(
                                feed_info["credibility_key"], 0.7
                            ),
                            source_tier=2,
                        )
                    )
            except Exception as e:
                log.error("RSS 크롤링 실패 | source=%s error=%s", source_name, e)
        return articles


class GoogleNewsRssCrawler(BaseCrawler):
    """Google News RSS 크롤러 — 쿼리 기반 (Tier 2)."""

    def __init__(
        self,
        peer_id: str,
        keywords: list[str],
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        super().__init__(peer_id, limit_guard)
        self.keywords = keywords

    async def crawl(self) -> list[RawArticle]:
        articles: list[RawArticle] = []
        for keyword in self.keywords:
            if not self.limit_guard.allow("google_news"):
                break
            try:
                url = GOOGLE_NEWS_RSS.format(query=keyword.replace(" ", "+"))
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles.append(
                        RawArticle(
                            url=entry.get("link", ""),
                            title=entry.get("title", ""),
                            content=entry.get("summary", ""),
                            published_at=_parse_entry_date(entry),
                            source_name="google_news",
                            peer_id=self.peer_id,
                            credibility_score=SOURCE_CREDIBILITY["google_news"],
                            source_tier=2,
                        )
                    )
            except Exception as e:
                log.error("Google News RSS 실패 | keyword=%s error=%s", keyword, e)
        return articles


def _parse_entry_date(entry: Any) -> Optional[datetime]:
    try:
        return parsedate_to_datetime(entry.get("published", ""))
    except Exception:
        return None
