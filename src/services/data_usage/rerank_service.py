"""RerankService placeholder.

추후 개발: 검색 결과를 cross-encoder 또는 LLM-free scoring으로 재정렬한다.
"""

from __future__ import annotations

from typing import Any


class RerankService:
    """Planned reranking service."""

    status = "planned"
    note = "추후 개발"

    async def rerank(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("RerankService는 추후 개발 예정입니다.")


__all__ = ["RerankService"]

