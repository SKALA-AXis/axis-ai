"""StrategicInsightAgent facade.

This class owns the pipeline-level "analysis + implication" step. The current
implementation keeps the proven StrategicAnalyzer and ImplicationAgent as
internal engines, so the graph has one conceptual reasoning node while the
prompts can be merged later without changing graph wiring again.
"""

from __future__ import annotations

from typing import Any

from src.agents.implication_agent import ImplicationAgent
from src.agents.strategic_analyzer import StrategicAnalyzer
from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    AnalysisResult,
    ProfileContext,
)


class StrategicInsightAgent:
    """Generate peer analysis and SK AX implication from an IntegratedIssue."""

    def __init__(
        self,
        *,
        analyzer: StrategicAnalyzer | None = None,
        implication_agent: ImplicationAgent | None = None,
    ) -> None:
        self.analyzer = analyzer or StrategicAnalyzer()
        self.implication_agent = implication_agent or ImplicationAgent()

    def generate(
        self,
        *,
        input_bundle: AnalysisInputBundle,
        integrated_issue: dict[str, Any],
        classification: dict[str, Any] | None = None,
        cluster_metadata: dict[str, Any] | None = None,
        profile_context: ProfileContext | None = None,
        analysis_context: AnalysisContext | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return ``{"analysis": ..., "implication": ...}``.

        ``StrategicAnalyzer`` remains peer-oriented. ``ImplicationAgent`` then
        turns that analysis into SK AX-oriented implications and actions.
        """
        classification = classification or {}
        analysis = self.analyzer.analyze(
            integrated_issue=integrated_issue,
            classification=classification,
            cluster_metadata=cluster_metadata or {},
        )
        implication = self.implication_agent.generate(
            input_bundle=input_bundle,
            integrated_issue=integrated_issue,
            analysis=AnalysisResult.from_dict(analysis).to_dict(),
            profile_context=profile_context,
            analysis_context=analysis_context,
            classification=classification,
        )
        return {"analysis": analysis, "implication": implication}


__all__ = ["StrategicInsightAgent"]
