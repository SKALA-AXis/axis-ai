"""RelevanceService placeholder.

추후 개발: raw data 관련성 판단 경계를 service 이름으로 정리한다.
"""

from __future__ import annotations

from typing import Any


class RelevanceService:
    """Planned relevance service."""

    status = "planned"
    note = "추후 개발"

    def judge(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("RelevanceService는 추후 개발 예정입니다.")


__all__ = ["RelevanceService"]

