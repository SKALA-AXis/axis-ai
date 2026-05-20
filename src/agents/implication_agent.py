"""SK AX 관점 시사점 Agent wrapper."""

from __future__ import annotations

from src.analysis.implication import ImplicationGenerator


class ImplicationAgent(ImplicationGenerator):
    """AnalysisResult, ProfileContext, AnalysisInputBundle을 결합하는 Agent.

    실제 생성 로직은 기존 ``src.analysis.implication.ImplicationGenerator``를
    그대로 사용한다.
    """


__all__ = ["ImplicationAgent"]
