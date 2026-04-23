"""수집 파이프라인 — 1시간마다 실행.
ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.
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
    cluster_map: dict          # {cluster_id: [article_ids]}
    representative_ids: list[int]
    classified_clusters: list[dict]  # [{cluster_id, rep_id, peer_id, importance, ...}]
    issue_cards: list[dict]
    implications: list[dict]
    validation_results: list[dict]
    errors: Annotated[list[str], operator.add]
    human_review_flags: list[int]


# ── 노드 구현 ──────────────────────────────────────────────────

def crawl_node(state: IngestionState) -> IngestionState:
    """처리 대기 중인 RAW 기사 ID를 DB에서 조회한다."""
    from src.agents.crawler_agent import CrawlerAgent
    raw_ids = CrawlerAgent().load_raw_ids(state["peer_ids"])
    log.info("RAW 기사 로드 | peer_ids=%s count=%d", state["peer_ids"], len(raw_ids))
    return {**state, "raw_article_ids": raw_ids}


def credibility_node(state: IngestionState) -> IngestionState:
    """Gate 2: credibility_score 기준 신뢰도 필터."""
    from src.agents.credibility_agent import CredibilityAgent
    credible_ids, skipped = CredibilityAgent().filter(state["raw_article_ids"])
    log.info("Gate 2 완료 | credible=%d skipped=%d", len(credible_ids), len(skipped))
    return {**state, "credible_ids": credible_ids}


def dedup_node(state: IngestionState) -> IngestionState:
    """Gate 3: BGE-M3 코사인 유사도 클러스터링."""
    from src.agents.dedup_agent import DeduplicationAgent
    cluster_map, rep_ids = DeduplicationAgent().deduplicate(state["credible_ids"])
    log.info("Gate 3 완료 | clusters=%d reps=%d", len(cluster_map), len(rep_ids))
    return {**state, "cluster_map": cluster_map, "representative_ids": rep_ids}


def classify_node(state: IngestionState) -> IngestionState:
    """중요도 분류 — 클러스터별 GPT-4o 호출."""
    from src.agents.classification_agent import ClassificationAgent
    from src.db.article_store import get_articles_by_ids

    agent = ClassificationAgent()
    classified: list[dict] = []

    # rep_id → peer_id 매핑
    rep_articles = {a["id"]: a for a in get_articles_by_ids(state["representative_ids"])}

    for cluster_id, article_ids in state["cluster_map"].items():
        rep_id = next(
            (aid for aid in article_ids if aid in rep_articles),
            article_ids[0] if article_ids else None,
        )
        if rep_id is None:
            continue
        result = agent.classify(cluster_id, rep_id)
        classified.append({
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "peer_id": rep_articles.get(rep_id, {}).get(
                "peer_id", state["peer_ids"][0] if state["peer_ids"] else ""
            ),
            **result,
        })

    log.info("중요도 분류 완료 | clusters=%d", len(classified))
    return {**state, "classified_clusters": classified}


def issue_card_node(state: IngestionState) -> IngestionState:
    """이슈 카드 생성 — reference 등급도 생성 (후속 필터링에서 제외 가능)."""
    from src.agents.issue_card_agent import IssueCardAgent

    agent = IssueCardAgent()
    cards: list[dict] = []

    for cluster in state["classified_clusters"]:
        card = agent.generate(
            cluster_id=cluster["cluster_id"],
            representative_id=cluster["representative_id"],
            peer_id=cluster["peer_id"],
            classification=cluster,
        )
        if card:
            cards.append(card)

    log.info("이슈카드 생성 완료 | cards=%d", len(cards))
    return {**state, "issue_cards": cards}


def implication_node(state: IngestionState) -> IngestionState:
    """시사점 생성 — notable 이상 등급만 GPT-4o 호출."""
    from src.agents.implication_agent import ImplicationAgent

    agent = ImplicationAgent()
    implications: list[dict] = []

    for card in state["issue_cards"]:
        if card.get("importance") == "reference":
            # reference 등급은 시사점 생략
            implications.append({"card_id": card["id"], "skipped": True})
            continue
        result = agent.generate(card)
        result["card_id"] = card["id"]
        implications.append(result)
        card["implication"] = result  # 카드에 시사점 붙이기

    log.info("시사점 생성 완료 | total=%d skipped=%d",
             len(implications), sum(1 for i in implications if i.get("skipped")))
    return {**state, "implications": implications}


def validation_node(state: IngestionState) -> IngestionState:
    """SC 검증 + 이슈카드 DB 저장."""
    from src.agents.validation_agent import ValidationAgent
    from src.db.article_store import save_issue_card

    agent = ValidationAgent()
    results: list[dict] = []
    human_review_flags: list[int] = []

    implication_map = {i["card_id"]: i for i in state["implications"] if not i.get("skipped")}

    for card in state["issue_cards"]:
        card_id = card["id"]
        implication = implication_map.get(card_id, {})

        if implication:
            validation = agent.validate(card, implication)
        else:
            validation = {"pass": True, "sc_score": 1.0, "reason": "reference 등급 스킵"}

        card["validation"] = validation
        results.append({"card_id": card_id, **validation})

        # SC 실패 → human review 플래그
        if not validation["pass"]:
            human_review_flags.append(card.get("cluster_id", 0))
            log.warning("SC 검증 실패 — human review 필요 | card_id=%s reason=%s",
                        card_id, validation.get("reason"))

        # DB 저장
        save_issue_card(card)

    log.info(
        "SC 검증 + 저장 완료 | total=%d pass=%d fail=%d",
        len(results),
        sum(1 for r in results if r["pass"]),
        sum(1 for r in results if not r["pass"]),
    )
    return {**state, "validation_results": results, "human_review_flags": human_review_flags}


# ── 그래프 조립 ────────────────────────────────────────────────

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
