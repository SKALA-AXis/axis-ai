"""InsightAgent placeholder for AnalysisGraphRunner.

추후 개발: 단일 AnalysisPackage 생성 흐름 안에서 content_analysis와 context를 기반으로
흐름 단위 인사이트 후보를 만든다. 2단계 DataUsageOrchestrator의 InsightAgent와
책임 경계를 확정한 뒤 구현한다.
"""

from __future__ import annotations

from typing import Any


class InsightAgent:
    """Planned in-graph insight stage agent."""

    status = "planned"
    note = "추후 개발"

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("AnalysisGraphRunner용 InsightAgent는 추후 개발 예정입니다.")


class AnalysisFlowInsightAgent(InsightAgent):
    """Explicit alias to avoid confusion with DataUsageOrchestrator InsightAgent."""


__all__ = ["AnalysisFlowInsightAgent", "InsightAgent"]
