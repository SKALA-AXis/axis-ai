"""SearchSuggestService placeholder.

추후 개발: keyword/search cache를 기반으로 자동완성 후보를 제공한다.
"""

from __future__ import annotations

from typing import Any


class SearchSuggestService:
    """Planned search suggestion service."""

    status = "planned"
    note = "추후 개발"

    def suggest(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("SearchSuggestService는 추후 개발 예정입니다.")


__all__ = ["SearchSuggestService"]
