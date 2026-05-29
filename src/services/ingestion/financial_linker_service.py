"""FinancialLinkerService placeholder.

추후 개발: 카드/이슈와 DART/IR 재무 metric 근거를 연결한다.
"""

from __future__ import annotations

from typing import Any


class FinancialLinkerService:
    """Planned financial evidence linker."""

    status = "planned"
    note = "추후 개발"

    def link(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("FinancialLinkerService는 추후 개발 예정입니다.")


__all__ = ["FinancialLinkerService"]
