"""크롤러 베이스 클래스"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

CRAWL_TIMEOUT = 10
MAX_RETRIES = 1


@dataclass
class RawArticle:
    url: str
    title: str
    content: str
    published_at: Optional[datetime]
    source_name: str
    peer_id: str


class BaseCrawler(ABC):
    def __init__(self, peer_id: str):
        self.peer_id = peer_id

    @abstractmethod
    async def crawl(self) -> list[RawArticle]:
        """소스에서 기사를 수집한다."""
        ...

    def _is_blocked(self, status_code: int) -> bool:
        return status_code in (403, 429)
