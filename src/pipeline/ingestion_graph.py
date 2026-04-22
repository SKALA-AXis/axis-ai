"""수집 파이프라인 — 1시간마다 실행
ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙
"""

import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


class IngestionState(TypedDict):
    peer_ids: list[str]
    trigger_type: str
    raw_article_ids: list[int]
    credible_ids: list[int]
    cluster_map: dict
    representative_ids: list[int]
    classified_clusters: list[dict]
    issue_cards: list[dict]
    implications: list[dict]
    validation_results: list[dict]
    errors: Annotated[list[str], operator.add]
    human_review_flags: list[int]


def crawl_node(state: IngestionState) -> IngestionState:
    log.info("크롤링 시작 | peer_ids=%s", state["peer_ids"])
    # TODO: crawler_agent.py 연결
    return {**state, "raw_article_ids": []}


def credibility_node(state: IngestionState) -> IngestionState:
    log.info("신뢰도 분류 | articles=%d", len(state["raw_article_ids"]))
    # TODO: credibility_agent.py 연결
    return {**state, "credible_ids": state["raw_article_ids"]}


def dedup_node(state: IngestionState) -> IngestionState:
    log.info("중복 제거 및 클러스터링")
    # TODO: dedup_agent.py 연결
    return {**state, "cluster_map": {}, "representative_ids": []}


def classify_node(state: IngestionState) -> IngestionState:
    log.info("중요도 분류")
    # TODO: classification_agent.py 연결
    return {**state, "classified_clusters": []}


def issue_card_node(state: IngestionState) -> IngestionState:
    log.info("이슈 카드 생성")
    # TODO: issue_card_agent.py 연결
    return {**state, "issue_cards": []}


def implication_node(state: IngestionState) -> IngestionState:
    log.info("시사점 생성")
    # TODO: implication_agent.py 연결
    return {**state, "implications": []}


def validation_node(state: IngestionState) -> IngestionState:
    log.info("SC 검증 (환각 방지)")
    # TODO: validation_agent.py 연결
    return {**state, "validation_results": []}


def build_ingestion_graph() -> StateGraph:  # type: ignore[return]
    graph = StateGraph(IngestionState)
    graph.add_node("crawl", crawl_node)
    graph.add_node("credibility", credibility_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("classify", classify_node)
    graph.add_node("issue_card", issue_card_node)
    graph.add_node("implication", implication_node)
    graph.add_node("validation", validation_node)

    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "credibility")
    graph.add_edge("credibility", "dedup")
    graph.add_edge("dedup", "classify")
    graph.add_edge("classify", "issue_card")
    graph.add_edge("issue_card", "implication")
    graph.add_edge("implication", "validation")
    graph.add_edge("validation", END)

    return graph.compile()


ingestion_graph = build_ingestion_graph()
