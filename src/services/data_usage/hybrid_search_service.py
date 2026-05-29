"""HybridSearchService placeholder.

추후 개발: BM25/vector/hybrid retrieval을 제공한다.
"""

from __future__ import annotations

from typing import Any


class HybridSearchService:
    """Planned deterministic retrieval service."""

    status = "planned"
    note = "추후 개발"

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("HybridSearchService는 추후 개발 예정입니다.")


__all__ = ["HybridSearchService"]
