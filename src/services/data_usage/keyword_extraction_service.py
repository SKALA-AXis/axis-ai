"""KeywordExtractionService placeholder.

추후 개발: 저장된 raw/card 데이터에서 키워드를 추출하고 cache한다.
"""

from __future__ import annotations

from typing import Any


class KeywordExtractionService:
    """Planned keyword extraction service."""

    status = "planned"
    note = "추후 개발"

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("KeywordExtractionService는 추후 개발 예정입니다.")


__all__ = ["KeywordExtractionService"]
