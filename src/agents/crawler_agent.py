"""크롤러 에이전트 — 뉴스·공시·채용공고 수집"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Article:
    url: str
    title: str
    content: str
    published_at: Optional[datetime]
    source_name: str
    peer_id: str


class CrawlerAgent:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id

    async def run(self) -> list[Article]:
        """뉴스·공시·채용공고를 수집한다."""
        log.info("크롤링 시작 | peer_id=%s", self.peer_id)
        # TODO: naver_crawler, dart_crawler, rss_crawler, job_crawler 연결
        return []
