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
    """처리 대기 중인 RAW 기사 ID를 DB에서 조회한다.

    실제 수집은 APScheduler → BatchProcessor 경로로 먼저 완료되어 있어야 합니다.
    """
    from src.agents.crawler_agent import CrawlerAgent

    raw_ids = CrawlerAgent().load_raw_ids(state["peer_ids"])
    log.info("RAW 기사 로드 완료 | peer_ids=%s count=%d", state["peer_ids"], len(raw_ids))
    return {**state, "raw_article_ids": raw_ids}


def credibility_node(state: IngestionState) -> IngestionState:
    """Gate 2: credibility_score 기준 신뢰도 필터."""
    from src.agents.credibility_agent import CredibilityAgent

    credible_ids, skipped = CredibilityAgent().filter(state["raw_article_ids"])
    log.info("Gate 2 완료 | credible=%d skipped=%d", len(credible_ids), len(skipped))
    return {**state, "credible_ids": credible_ids}


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


def build_ingestion_graph() -> StateGraph:
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

    return graph.compile()  # type: ignore[return-value]


ingestion_graph = build_ingestion_graph()
