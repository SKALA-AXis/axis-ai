"""ClassificationService placeholder.

추후 개발: 기업/섹터/이벤트 매칭 경계를 service 이름으로 정리한다.
"""

from __future__ import annotations

from typing import Any


class ClassificationService:
    """Planned classification service."""

    status = "planned"
    note = "추후 개발"

    def classify(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("ClassificationService는 추후 개발 예정입니다.")


__all__ = ["ClassificationService"]
