"""ParserService placeholder.

추후 개발: source별 parser router와 parser output 계약을 service 이름으로 정리한다.
"""

from __future__ import annotations

from typing import Any


class ParserService:
    """Planned parser service facade."""

    status = "planned"
    note = "추후 개발"

    def parse(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("ParserService는 추후 개발 예정입니다.")


__all__ = ["ParserService"]
