"""IntegratedIssue 기반 전략 분석 Agent wrapper."""

from __future__ import annotations

from src.analysis.analyzer import StrategicAnalyzer


class AnalysisAgent(StrategicAnalyzer):
    """DataAnalysisSupervisorAgent가 호출하는 분석 Agent.

    실제 분석 로직은 기존 ``src.analysis.analyzer.StrategicAnalyzer``를 그대로 사용한다.
    원문/클러스터를 다시 읽지 않고 IssueIntegrationAgent가 만든 IntegratedIssue를 분석한다.
    """


__all__ = ["AnalysisAgent"]
