"""크롤러 공통 기반 — RawArticle, DailyLimitGuard, RetryPolicy."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional
from uuid import uuid4

from src.config.company_tiers import company_tier_map

log = logging.getLogger(__name__)

RETRY_POLICY: dict[str, Any] = {
    "timeout": 10,
    "source_timeout": {"max_retries": 3, "backoff": [10, 60, 300]},
    "db_failure": {"max_retries": 3, "backoff": [5, 30, 120]},
    "playwright_timeout": {"max_retries": 2, "backoff": [15, 60]},
    "playwright_blocked": {"max_retries": 1, "backoff": [300]},
}

SourceType = Literal[
    "news",
    "ir",
    "securities_report",
    "dart",
    "job",
    "trend_report",
    "search_trend",
    "social",
    "official",
    "company_site",
    "market_data",
]

ContentType = Literal[
    "html",
    "pdf",
    "api",
    "text",
    "unknown",
]

CrawlStatus = Literal[
    "success",
    "failed",
    "skipped",
]


@dataclass
class RawArticle:
    """크롤러가 수집한 원천 기사 및 자료 레코드.

    최초 raw_articles INSERT에 필요한 원천 데이터만 담는다.
    신뢰도, 분류, 인사이트 결과는 이후 Agent가 DB row를 업데이트한다.
    """

    url: str
    title: str
    content: Optional[str]
    source_name: str

    published_at: Optional[datetime] = None
    collected_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    id: str = field(default_factory=lambda: str(uuid4()))

    source_type: SourceType = "news"
    content_type: ContentType = "html"
    publisher: Optional[str] = None

    company: list[str] = field(default_factory=list)
    language: str = "ko"

    crawl_status: CrawlStatus = "success"
    error_message: Optional[str] = None

    url_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # 기존 peer_id 기반 크롤러 호환용 필드.
    # 신규 코드는 company를 사용한다.
    peer_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.url_hash:
            self.url_hash = hashlib.md5(self.url.encode()).hexdigest()

        if self.peer_id and not self.company:
            self.company = [self.peer_id]

        self.company = _dedupe_keep_order(self.company)

    def to_common_dict(self) -> dict[str, Any]:
        """공통 JSON 스키마 형태로 변환한다."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "publisher": self.publisher,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "url_hash": self.url_hash,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "collected_at": self.collected_at.isoformat(),
            "company": self.company,
            "company_tier": company_tier_map(self.company),
            "language": self.language,
            "content_type": self.content_type,
            "crawl_status": self.crawl_status,
            "error_message": self.error_message,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class CrawlWindow:
    """크롤링 대상 기간."""

    start: datetime
    end: Optional[datetime] = None

    @classmethod
    def last_days(cls, days: int) -> "CrawlWindow":
        end = datetime.now().astimezone()
        start = end - timedelta(days=max(days, 0))
        return cls(start=start, end=end)

    def contains(self, value: Optional[datetime]) -> bool:
        if value is None:
            return True

        start = _align_tz(self.start, value)
        end = _align_tz(self.end, value) if self.end else None

        if value < start:
            return False
        return end is None or value <= end


@dataclass(frozen=True)
class CrawlRunContext:
    """크롤 실행 메타데이터. V30 이후 raw_articles.crawl_events에 흡수 저장된다."""

    collection_mode: str
    crawl_run_id: str | None = None
    source_name: str | None = None
    track: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None


class DailyLimitGuard:
    """전역 일일 수집 건수 한도와 소스별 개별 한도를 관리한다."""

    GLOBAL_LIMIT = 20_000

    SOURCE_TYPE_LIMITS: dict[str, int] = {
        "news": 10000,
        "official": 100,
        "company_site": 500,
        "ir": 100,
        "dart": 100,
        "securities_report": 100,
        "job": 100,
        "trend_report": 100,
        "search_trend": 100,
        "social": 200,
        "market_data": 100,
    }

    def __init__(self) -> None:
        self._global_count = 0
        self._source_counts: dict[str, int] = {}
        self._reset_date: Optional[date] = None

    def _maybe_reset(self) -> None:
        today = datetime.now().date()

        if self._reset_date is None or self._reset_date != today:
            self._global_count = 0
            self._source_counts = {}
            self._reset_date = today

    def check(self, new_count: int) -> bool:
        """전역 수집 한도를 확인하고 사용량을 반영한다."""
        self._maybe_reset()

        if self._global_count + new_count > self.GLOBAL_LIMIT:
            log.warning(
                "일일 전역 수집 한도 초과 | count=%d limit=%d",
                self._global_count,
                self.GLOBAL_LIMIT,
            )
            return False

        self._global_count += new_count
        return True

    def allow(self, source_type: str) -> bool:
        """소스별 수집 한도를 확인하고 사용량을 반영한다."""
        self._maybe_reset()

        limit = self.SOURCE_TYPE_LIMITS.get(source_type, 500)
        current = self._source_counts.get(source_type, 0)

        if current >= limit:
            log.warning(
                "소스 타입별 수집 한도 초과 | source_type=%s limit=%d",
                source_type,
                limit,
            )
            return False

        self._source_counts[source_type] = current + 1
        return self.check(1)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    """리스트 순서를 유지하면서 중복 값을 제거한다."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def _align_tz(boundary: datetime, value: datetime) -> datetime:
    if value.tzinfo is None and boundary.tzinfo is not None:
        return boundary.replace(tzinfo=None)
    if value.tzinfo is not None and boundary.tzinfo is None:
        return boundary.replace(tzinfo=value.tzinfo)
    return boundary
