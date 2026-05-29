"""DerivedMetricsService placeholder.

추후 개발: dashboard/report용 파생 지표를 계산한다.
"""

from __future__ import annotations

from typing import Any


class DerivedMetricsService:
    """Planned derived-metrics service."""

    status = "planned"
    note = "추후 개발"

    def calculate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("DerivedMetricsService는 추후 개발 예정입니다.")


__all__ = ["DerivedMetricsService"]
