# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인 작성 및 ruff format 적용
#   2026-04-30 박지원 — crawler v1 및 크롤러 구현/전처리 agent 수정
"""크롤러 공통 부모 클래스."""

from abc import ABC, abstractmethod
from typing import Optional

from src.crawler.base import DailyLimitGuard, RawArticle


class BaseCrawler(ABC):
    """모든 크롤러가 상속하는 공통 베이스 클래스."""

    def __init__(
        self,
        company: Optional[list[str] | str] = None,
        peer_id: Optional[str] = None,
        limit_guard: Optional[DailyLimitGuard] = None,
    ) -> None:
        if company is None:
            company_values = [peer_id] if peer_id else []
        elif isinstance(company, str):
            company_values = [company]
        else:
            company_values = company

        self.company = _dedupe_keep_order(company_values)
        self.peer_id = peer_id or (self.company[0] if self.company else None)
        self.limit_guard = limit_guard or DailyLimitGuard()

    @abstractmethod
    async def crawl(self) -> list[RawArticle]:
        """소스에서 기사 또는 자료를 수집한다."""
        ...

    def _is_blocked(self, status_code: int) -> bool:
        """차단성 HTTP 상태 코드인지 확인한다."""
        return status_code in (403, 429)


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
