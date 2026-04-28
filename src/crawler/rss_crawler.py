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
    def __init__(self, peer_id: str, aliases: list[str]):
        super().__init__(peer_id)
        self.aliases = aliases

    async def crawl(self) -> list[RawArticle]:
        articles = []
        for source_name, url in self._source_urls().items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles.append(
                        RawArticle(
                            url=entry.link,
                            title=strip_html(entry.title),
                            content=strip_html(entry.get("summary", "")),
                            published_at=_parse_published_at(entry),
                            source_name=source_name,
                            peer_id=self.peer_id,
                        )
                    )
            except Exception as e:
                log.error("RSS 크롤링 실패 | source=%s error=%s", source_name, e)
        return articles

    def _source_urls(self) -> dict[str, str]:
        urls = dict(RSS_SOURCES)
        for alias in self.aliases:
            query = quote_plus(f'"{alias}"')
            urls[f"google_news:{alias}"] = GOOGLE_NEWS_RSS_URL.format(query=query)
        return urls


def _parse_published_at(entry) -> datetime | None:
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if published is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(published), tz=timezone.utc)
