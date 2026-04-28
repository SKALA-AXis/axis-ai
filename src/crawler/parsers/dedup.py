"""URL 해시 기반 인메모리 중복 제거 (재시작 시 초기화)."""

import hashlib
import logging
from collections import OrderedDict

from src.crawler.base import RawArticle

log = logging.getLogger(__name__)

MAX_CACHE_SIZE = 10_000


class DedupStore:
    """URL SHA-256 해시로 수집 중복 기사를 제거한다."""

    def __init__(self, max_size: int = MAX_CACHE_SIZE) -> None:
        self._seen: OrderedDict[str, bool] = OrderedDict()
        self._max_size = max_size

    def _hash(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def is_seen(self, url: str) -> bool:
        return self._hash(url) in self._seen

    def mark_seen(self, url: str) -> None:
        h = self._hash(url)
        self._seen[h] = True
        if len(self._seen) > self._max_size:
            self._seen.popitem(last=False)

    def filter_new(self, articles: list[RawArticle]) -> list[RawArticle]:
        """이미 수집된 URL을 제거하고 새 기사만 반환한다."""
        new_articles = []
        for a in articles:
            if not self.is_seen(a.url):
                new_articles.append(a)
                self.mark_seen(a.url)
        removed = len(articles) - len(new_articles)
        if removed:
            log.debug("중복 제거 | removed=%d", removed)
        return new_articles
