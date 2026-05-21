"""2단계 저장 데이터 활용 Orchestrator.

DataUsageOrchestrator는 새 데이터를 수집하거나 분석 패키지를 새로 만들지 않는다.
이미 저장된 카드뉴스, 시사점, 분석 결과를 활용해 믹서/IT 트렌드/리포트/
인사이트/챗봇/키워드 그래프 계열 작업을 라우팅하는 2단계 supervisor이다.
"""

from __future__ import annotations

from typing import Any

from src.agents.chatbot_agent import ChatbotAgent
from src.agents.insight_cascade_agent import InsightAgent
from src.agents.it_trend_agent import ITTrendAgent
from src.agents.keyword_graph_agent import KeywordGraphAgent
from src.agents.mixer_analysis_agent import MixerAgent
from src.agents.report_agent import ReportAgent


class DataUsageOrchestrator:
    """저장 데이터 활용 workflow supervisor."""

    def __init__(
        self,
        *,
        mixer_agent: MixerAgent | None = None,
        it_trend_agent: ITTrendAgent | None = None,
        report_agent: ReportAgent | None = None,
        insight_agent: InsightAgent | None = None,
        chatbot_agent: ChatbotAgent | None = None,
        keyword_graph_agent: KeywordGraphAgent | None = None,
    ) -> None:
        self.mixer_agent = mixer_agent or MixerAgent()
        self.it_trend_agent = it_trend_agent or ITTrendAgent()
        self.report_agent = report_agent or ReportAgent()
        self.insight_agent = insight_agent or InsightAgent()
        self.chatbot_agent = chatbot_agent or ChatbotAgent()
        self.keyword_graph_agent = keyword_graph_agent or KeywordGraphAgent()

    async def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """요청 목적에 맞는 활용 Agent를 실행한다."""
        task = str(request.get("task") or request.get("intent") or "").strip()
        card_ids = [str(card_id) for card_id in request.get("card_ids", []) if card_id]

        if task in {"mixer", "compare", "cross_card_analysis"}:
            return await self.mixer_agent.analyze(
                card_ids=card_ids,
                ratios=request.get("ratios"),
                user_context=str(request.get("user_context") or ""),
            )

        if task in {"insight", "insight_cascade"}:
            return await self.insight_agent.generate(
                card_ids=card_ids,
                context=request.get("context"),
            )

        if task in {"it_trend", "trend", "it_trends"}:
            trend_input = self.it_trend_agent.build_input(
                items=request.get("items") or [],
                period=request.get("period"),
                source_groups=request.get("source_groups"),
                metadata=request.get("metadata"),
            )
            return self.it_trend_agent.generate(trend_input)

        if task in {"report", "briefing"}:
            return self.report_agent.generate(
                report_type=str(request.get("report_type") or task),
                items=request.get("items") or [],
                context=request.get("context"),
            )

        if task in {"chatbot", "chat", "qa"}:
            return self.chatbot_agent.answer(
                question=str(request.get("question") or request.get("message") or ""),
                context=request.get("context"),
            )

        if task in {"keyword_graph", "keywords", "graph"}:
            return self.keyword_graph_agent.build(
                items=request.get("items") or [],
                filters=request.get("filters"),
            )

        return {
            "is_supported": False,
            "orchestrator": "DataUsageOrchestrator",
            "reason": "지원되지 않는 2단계 활용 요청입니다.",
            "requested_task": task,
            "supported_tasks": [
                "mixer",
                "compare",
                "cross_card_analysis",
                "insight",
                "insight_cascade",
                "it_trend",
                "report",
                "chatbot",
                "keyword_graph",
            ],
        }


__all__ = ["DataUsageOrchestrator"]
