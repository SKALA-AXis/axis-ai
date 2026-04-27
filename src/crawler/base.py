"""크롤러 공통 기반 — RawArticle v4, DailyLimitGuard, RetryPolicy, SOURCE_CREDIBILITY"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)

# credibility_score 기준표 (FR-002 사전조건)
SOURCE_CREDIBILITY: dict[str, float] = {
    "dart": 1.00,
    "kipris": 0.95,
    "samsung_sds_newsroom": 0.90,
    "lg_cns_newsroom": 0.90,
    "hyundai_autoever_newsroom": 0.90,
    "posco_dx_newsroom": 0.90,
    "hankyung_consensus": 0.80,
    "naver_research": 0.75,
    "naver_news": 0.75,
    "etnews": 0.70,
    "zdnet": 0.68,
    "itchosun": 0.65,
    "google_news": 0.65,
    "yonhap": 0.85,
    "saramin": 0.50,
    "telegram": 0.30,
}

# 소스별 티어 (credibility_score 범위)
# Tier 1: 0.9~1.0  공식 (DART, KIPRIS, 공식 뉴스룸)
# Tier 2: 0.6~0.9  전문 리서치·언론
# Tier 3: 0.0~0.5  비공식 (텔레그램, SNS)

RETRY_POLICY: dict[str, Any] = {
    "timeout": 10,  # httpx 기본 타임아웃 (초) — 하위 호환
    "source_timeout": {"max_retries": 3, "backoff": [10, 60, 300]},
    "db_failure": {"max_retries": 3, "backoff": [5, 30, 120]},
    "playwright_timeout": {"max_retries": 2, "backoff": [15, 60]},
    "playwright_blocked": {"max_retries": 1, "backoff": [300]},  # 차단 시 5분 대기
}


@dataclass
class RawArticle:
    url: str
    title: str
    content: str  # 요약 or 본문
    source_tier: int  # 1·2·3
    source_name: str
    credibility_score: float = 0.5  # FR-002 사전조건
    peer_id: Optional[str] = None  # samsung_sds · lg_cns · None
    published_at: Optional[datetime] = None
    collected_at: datetime = field(default_factory=datetime.now)
    url_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.url_hash = hashlib.md5(self.url.encode()).hexdigest()


class DailyLimitGuard:
    """전역 일일 수집 건수 한도 + 소스별 개별 한도 관리."""

    GLOBAL_LIMIT = 5_000  # 건/일 (설계서 §11)
    SOURCE_LIMITS: dict[str, int] = {
        "naver_news": 200,
        "dart": 100,
        "kipris": 100,
        "rss": 300,
        "google_news": 200,
        "yonhap": 200,
        "official": 50,
        "jobs": 100,
        "consensus": 100,
        "naver_research": 100,
    }

    def __init__(self) -> None:
        self._global_count = 0
        self._source_counts: dict[str, int] = {}
        self._reset_date: Optional[datetime] = None

    def _maybe_reset(self) -> None:
        today = datetime.now().date()
        if self._reset_date is None or self._reset_date != today:
            self._global_count = 0
            self._source_counts = {}
            self._reset_date = today  # type: ignore[assignment]

    def check(self, new_count: int) -> bool:
        """전역 한도 체크 (설계서 §11 DailyLimitGuard.check)."""
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

    def allow(self, source: str) -> bool:
        """소스별 한도 체크."""
        self._maybe_reset()
        limit = self.SOURCE_LIMITS.get(source, 500)
        current = self._source_counts.get(source, 0)
        if current >= limit:
            log.warning("소스별 수집 한도 초과 | source=%s limit=%d", source, limit)
            return False
        self._source_counts[source] = current + 1
        return self.check(1)


class BaseCrawler(ABC):
    def __init__(
        self, peer_id: Optional[str] = None, limit_guard: Optional[DailyLimitGuard] = None
    ) -> None:
        self.peer_id = peer_id
        self.limit_guard = limit_guard or DailyLimitGuard()

    @abstractmethod
    async def crawl(self) -> list[RawArticle]:
        """소스에서 기사를 수집한다."""
        ...

    def _is_blocked(self, status_code: int) -> bool:
        return status_code in (403, 429)
