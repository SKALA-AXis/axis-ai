"""크롤러 베이스 클래스"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4

log = logging.getLogger(__name__)

CRAWL_TIMEOUT = 10
MAX_RETRIES = 1

SourceType = Literal[
    "news",
    "ir",
    "securities_report",
    "dart",
    "job",
    "trend_report",
    "search_trend",
    "social",
]
ContentType = Literal["html", "pdf", "api", "rss"]
CrawlStatus = Literal["success", "failed"]


@dataclass
class RawArticle:
    """크롤러 공통 출력 레코드.

    팀 공통 JSON 스키마를 기본으로 두되, 기존 크롤러가 쓰던 peer_id 같은
    내부 호환 필드는 유지한다. 자료 유형별 추가 필드는 extra에 넣는다.
    """

    url: str
    title: str
    content: Optional[str]
    published_at: Optional[datetime]
    source_name: str
    peer_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    source_type: SourceType = "news"
    collected_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    publisher: Optional[str] = None
    company: list[str] = field(default_factory=list)
    language: str = "ko"
    country: str = "KR"
    content_type: ContentType = "html"
    crawl_status: CrawlStatus = "success"
    error_message: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.company and self.peer_id:
            self.company = [self.peer_id]

    def to_common_dict(self) -> dict[str, Any]:
        """팀 공통 JSON 스키마 형태로 직렬화한다."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "collected_at": self.collected_at.isoformat(),
            "publisher": self.publisher,
            "company": self.company,
            "language": self.language,
            "country": self.country,
            "content_type": self.content_type,
            "crawl_status": self.crawl_status,
            "error_message": self.error_message,
            "extra": self.extra,
        }


class BaseCrawler(ABC):
    def __init__(self, peer_id: str):
        self.peer_id = peer_id

    @abstractmethod
    async def crawl(self) -> list[RawArticle]:
        """소스에서 기사를 수집한다."""
        ...

    def _is_blocked(self, status_code: int) -> bool:
        return status_code in (403, 429)
