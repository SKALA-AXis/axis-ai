"""CrawlerService / CrawlerJob placeholder.

추후 개발: source별 crawl 실행 경계를 service/job 이름으로 정리한다.
"""

from __future__ import annotations

from typing import Any


class CrawlerService:
    """Planned crawler service facade."""

    status = "planned"
    note = "추후 개발"

    def crawl(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("CrawlerService는 추후 개발 예정입니다.")


class CrawlerJob(CrawlerService):
    """Planned scheduled crawler job alias."""


__all__ = ["CrawlerJob", "CrawlerService"]

