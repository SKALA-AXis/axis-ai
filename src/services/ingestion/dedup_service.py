"""DedupService placeholder.

추후 개발: 중복 제거와 뉴스 클러스터링 경계를 service 이름으로 정리한다.
"""

from __future__ import annotations

from typing import Any


class DedupService:
    """Planned deduplication/clustering service."""

    status = "planned"
    note = "추후 개발"

    def deduplicate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("DedupService는 추후 개발 예정입니다.")


__all__ = ["DedupService"]
