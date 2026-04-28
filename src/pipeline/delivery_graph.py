"""전달 파이프라인 — 매일 오전 8:30 실행
ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙
"""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


class DeliveryState(TypedDict):
    issue_card_ids: list[str]
    briefing_text: str
    slack_sent: bool
    errors: list[str]


def load_issues_node(state: DeliveryState) -> DeliveryState:
    """PostgreSQL에서 오늘의 이슈 카드 조회"""
    log.info("이슈 카드 조회")
    # TODO: PostgreSQL에서 오늘 생성된 issue_cards 조회
    return {**state, "issue_card_ids": []}


def build_briefing_node(state: DeliveryState) -> DeliveryState:
    log.info("브리핑 구성 | cards=%d", len(state["issue_card_ids"]))
    # TODO: BriefingAgent 연결
    return {**state, "briefing_text": "(브리핑 생성 예정)"}


def send_slack_node(state: DeliveryState) -> DeliveryState:
    log.info("Slack 발송")
    # TODO: notification_agent.py 연결
    return {**state, "slack_sent": True}


def build_delivery_graph() -> StateGraph:
    graph = StateGraph(DeliveryState)
    graph.add_node("load_issues", load_issues_node)
    graph.add_node("build_briefing", build_briefing_node)
    graph.add_node("send_slack", send_slack_node)

    graph.set_entry_point("load_issues")
    graph.add_edge("load_issues", "build_briefing")
    graph.add_edge("build_briefing", "send_slack")
    graph.add_edge("send_slack", END)

    return graph.compile()  # type: ignore[return-value]


delivery_graph = build_delivery_graph()
