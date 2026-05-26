"""AnswerGenerator placeholder.

추후 개발: retrieved/reranked evidence를 기반으로 사용자 답변을 생성한다.
"""

from __future__ import annotations

from typing import Any


class AnswerGenerator:
    """Planned answer generation component."""

    status = "planned"
    note = "추후 개발"

    async def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("AnswerGenerator는 추후 개발 예정입니다.")


class AnswerService(AnswerGenerator):
    """Planned service alias until final API naming is decided."""


__all__ = ["AnswerGenerator", "AnswerService"]

