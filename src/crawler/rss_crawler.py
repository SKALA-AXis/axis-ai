"""RSS 피드 크롤러"""

import calendar
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser

from src.crawler.article_filter import strip_html
from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

RSS_SOURCES = {
    "etnews": "https://www.etnews.com/rss/",
    "zdnet": "https://zdnet.co.kr/rss/",
}
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


class RssCrawler(BaseCrawler):
    def __init__(
        self,
        peer_id: str,
        aliases: list[str],
        search_queries: list[dict[str, str]] | None = None,
        max_entries_per_query: int = 2,
        include_google_news: bool = False,
    ):
        super().__init__(peer_id)
        self.aliases = aliases
        self.search_queries = search_queries
        self.max_entries_per_query = max_entries_per_query
        self.include_google_news = include_google_news

    async def crawl(self) -> list[RawArticle]:
        articles = []
        for source_name, source in self._source_urls().items():
            try:
                feed = feedparser.parse(source["url"])
                for entry in feed.entries[: self.max_entries_per_query]:
                    articles.append(
                        RawArticle(
                            url=entry.link,
                            title=strip_html(entry.title),
                            content=strip_html(entry.get("summary", "")),
                            published_at=_parse_published_at(entry),
                            source_name=source_name,
                            peer_id=self.peer_id,
                            sector=source.get("sector", ""),
                            search_query=source.get("query", ""),
                        )
                    )
            except Exception as e:
                log.error("RSS 크롤링 실패 | source=%s error=%s", source_name, e)
        return articles

    def _source_urls(self) -> dict[str, dict[str, str]]:
        urls = {
            source_name: {"url": url, "sector": "", "query": ""}
            for source_name, url in RSS_SOURCES.items()
        }

        query_specs = self.search_queries or [
            {"query": f'"{alias}"', "sector": ""} for alias in self.aliases
        ]

        if not self.include_google_news:
            return urls

        for spec in query_specs:
            query_text = spec["query"]
            query = quote_plus(query_text)
            urls[f"google_news:{query_text}"] = {
                "url": GOOGLE_NEWS_RSS_URL.format(query=query),
                "sector": spec.get("sector", ""),
                "query": query_text,
            }
        return urls


def _parse_published_at(entry) -> datetime | None:
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(published), tz=timezone.utc)
