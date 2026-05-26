"""Multi-source content analysis Agent.

AnalysisAgent는 IssueIntegrationAgent가 만든 IntegratedIssue를 입력으로 받아
뉴스/DART/IR/리포트/글로벌 자료를 같은 출력 계약으로 분석한다. 실제 LLM
프롬프트와 정규화 로직은 ``src.analysis.analyzer.StrategicAnalyzer``에 있다.
"""

from __future__ import annotations

from src.analysis.analyzer import StrategicAnalyzer


class AnalysisAgent(StrategicAnalyzer):
    """AnalysisGraphRunner가 호출하는 content analysis Agent.

    원문/클러스터를 다시 읽지 않고 IssueIntegrationAgent가 만든 IntegratedIssue와
    AnalysisContext를 분석한다. 출력은 기존 ``analysis_summary`` /
    ``strategic_meaning`` 계약을 유지하면서, ``content_analysis`` /
    ``detailed_findings`` / ``evidence_map`` / ``handoff`` 을 추가해 이후 요약,
    시사점, SK AX 대응방향 agent가 같은 근거를 재사용할 수 있게 한다.
    """


__all__ = ["AnalysisAgent"]
