"""EmbedIndexService placeholder.

추후 개발: raw/card/evidence payload를 vector index에 적재한다.
"""

from __future__ import annotations

from typing import Any


class EmbedIndexService:
    """Planned embedding index write service."""

    status = "planned"
    note = "추후 개발"

    def index(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("EmbedIndexService는 추후 개발 예정입니다.")


__all__ = ["EmbedIndexService"]

