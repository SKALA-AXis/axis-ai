"""Agent package exports.

무거운 Agent 모듈은 import side effect와 순환 import를 피하기 위해 lazy load한다.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AnalysisGraphRunner": "src.agents.analysis_graph_runner",
    "BriefingGenerationAgent": "src.agents.briefing_generation_agent",
    "CardNewsAgent": "src.agents.card_news_agent",
    "ChatbotAgent": "src.agents.chatbot_agent",
    "IntegrationAgent": "src.agents.integration_agent",
    "ImplicationAgent": "src.agents.implication_agent",
    "InsightCascadeAgent": "src.agents.insight_cascade_agent",
    "InsightAgent": "src.agents.insight_cascade_agent",
    "ITTrendAgent": "src.agents.it_trend_agent",
    "KeywordGraphAgent": "src.agents.keyword_graph_agent",
    "MixerAgent": "src.agents.mixer_analysis_agent",
    "MixerAnalysisAgent": "src.agents.mixer_analysis_agent",
    "PeerProfileAgent": "src.agents.peer_profile_agent",
    "ReportAgent": "src.agents.report_agent",
    "StrategicAnalyzer": "src.agents.strategic_analyzer",
    "StrategicInsightAgent": "src.agents.strategic_insight_agent",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
