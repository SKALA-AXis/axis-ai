"""RSS 피드 크롤러 (Tier 2) — ETnews, ZDNet, 연합뉴스, 블로터 + Google News RSS."""

import logging
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Optional

import feedparser
import httpx

from src.crawler.base import (
    SOURCE_CREDIBILITY,
    BaseCrawler,
    DailyLimitGuard,
    RawArticle,
)

log = logging.getLogger(__name__)

# feedparser 기본 UA가 차단되는 매체가 많아 httpx로 먼저 fetch 후 feedparser에 넘긴다.
_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
}

RSS_FEEDS: dict[str, dict[str, str]] = {
    "etnews_it": {
        "url": "https://rss.etnews.com/Section901.xml",
        "credibility_key": "etnews",
    },
    "etnews_industry": {
        "url": "https://rss.etnews.com/Section902.xml",
        "credibility_key": "etnews",
    },
    "etnews_economy": {
        "url": "https://rss.etnews.com/Section903.xml",
        "credibility_key": "etnews",
    },
    "zdnet": {
        "url": "https://feeds.feedburner.com/zdkorea",
        "credibility_key": "zdnet",
    },
    "bloter": {
        "url": "https://feeds.feedburner.com/bloter",
        "credibility_key": "zdnet",
    },
    "yonhap_tech": {
        "url": "https://www.yna.co.kr/rss/industry.xml",
        "credibility_key": "yonhap",
    },
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


def _fetch_feed(url: str) -> Any:
    """proper UA로 fetch 후 feedparser에 넘긴다."""
    try:
        r = httpx.get(url, headers=_RSS_HEADERS, timeout=8.0, follow_redirects=True)
        if r.status_code != 200:
            log.warning("RSS fetch 실패 | url=%s status=%d", url, r.status_code)
            return None
        return feedparser.parse(r.text)
    except Exception as e:
        log.warning("RSS fetch 예외 | url=%s error=%s", url, e)
        return None


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
                feed = _fetch_feed(feed_info["url"])
                if feed is None:
                    continue
                total = len(feed.entries)
                matched = 0
                for entry in feed.entries:
                    title = entry.get("title", "")
                    content = entry.get("summary", "") or entry.get("description", "")
                    haystack = f"{title} {content}".lower()
                    if not any(kw.lower() in haystack for kw in self.keywords):
                        continue
                    matched += 1
                    articles.append(
                        RawArticle(
                            url=entry.get("link", ""),
                            title=title,
                            content=content,
                            published_at=_parse_entry_date(entry),
                            source_name=source_name,
                            peer_id=self.peer_id,
                            credibility_score=SOURCE_CREDIBILITY.get(
                                feed_info["credibility_key"], 0.7
                            ),
                            source_tier=2,
                        )
                    )
                log.info(
                    "RSS 수집 | source=%s peer=%s entries=%d matched=%d",
                    source_name,
                    self.peer_id,
                    total,
                    matched,
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
                feed = _fetch_feed(url)
                if feed is None:
                    continue
                before = len(articles)
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
                log.info(
                    "Google News RSS 수집 | keyword=%s peer=%s entries=%d",
                    keyword,
                    self.peer_id,
                    len(articles) - before,
                )
            except Exception as e:
                log.error("Google News RSS 실패 | keyword=%s error=%s", keyword, e)
        return articles


def _parse_entry_date(entry: Any) -> Optional[datetime]:
    try:
        return parsedate_to_datetime(entry.get("published", ""))
    except Exception:
        return None
