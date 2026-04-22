"""RSS 피드 크롤러"""

import logging

import feedparser

from src.crawler.base_crawler import BaseCrawler, RawArticle

log = logging.getLogger(__name__)

RSS_SOURCES = {
    "etnews": "https://www.etnews.com/rss/",
    "zdnet": "https://zdnet.co.kr/rss/",
}


class RssCrawler(BaseCrawler):
    def __init__(self, peer_id: str, keywords: list[str]):
        super().__init__(peer_id)
        self.keywords = keywords

    async def crawl(self) -> list[RawArticle]:
        articles = []
        for source_name, url in RSS_SOURCES.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if any(kw in entry.title for kw in self.keywords):
                        articles.append(
                            RawArticle(
                                url=entry.link,
                                title=entry.title,
                                content=entry.get("summary", ""),
                                published_at=None,
                                source_name=source_name,
                                peer_id=self.peer_id,
                            )
                        )
            except Exception as e:
                log.error("RSS 크롤링 실패 | source=%s error=%s", source_name, e)
        return articles
