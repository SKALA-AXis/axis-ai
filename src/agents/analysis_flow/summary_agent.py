"""SummaryAgent placeholder.

추후 개발: AnalysisGraphRunner 내부에서 IntegratedIssue와 content_analysis를 기반으로
카드/리포트 재사용 가능한 요약 payload를 생성한다.
"""

from __future__ import annotations

from typing import Any


class SummaryAgent:
    """Planned summary stage agent."""

    status = "planned"
    note = "추후 개발"

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("SummaryAgent는 추후 개발 예정입니다.")


__all__ = ["SummaryAgent"]

