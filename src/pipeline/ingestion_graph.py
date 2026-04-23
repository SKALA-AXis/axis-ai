"""수집 파이프라인 — 1시간마다 실행.
ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.
"""

import logging
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)

# GPT-4o rate limit 고려: 분류·카드·시사점·SC 병렬 호출 수
_GPT_WORKERS = 5


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
    """중요도 분류 — 클러스터별 GPT-4o 병렬 호출."""
    from src.agents.classification_agent import ClassificationAgent
    from src.db.article_store import get_articles_by_ids

    agent = ClassificationAgent()
    rep_articles = {a["id"]: a for a in get_articles_by_ids(state["representative_ids"])}

    def _classify_one(cluster_id: int, article_ids: list[int]) -> dict | None:
        rep_id = next(
            (aid for aid in article_ids if aid in rep_articles),
            article_ids[0] if article_ids else None,
        )
        if rep_id is None:
            return None
        result = agent.classify(cluster_id, rep_id)
        return {
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "peer_id": rep_articles.get(rep_id, {}).get(
                "peer_id", state["peer_ids"][0] if state["peer_ids"] else ""
            ),
            **result,
        }

    classified: list[dict] = []
    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {
            ex.submit(_classify_one, cid, aids): cid
            for cid, aids in state["cluster_map"].items()
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                classified.append(result)

    log.info("중요도 분류 완료 | clusters=%d", len(classified))
    return {**state, "classified_clusters": classified}


def issue_card_node(state: IngestionState) -> IngestionState:
    """이슈 카드 생성 — 클러스터 상위 3건 참고, 병렬 GPT-4o 호출."""
    from src.agents.issue_card_agent import IssueCardAgent

    agent = IssueCardAgent()
    cluster_map = state["cluster_map"]

    def _generate_one(cluster: dict) -> dict:
        cluster_id = cluster["cluster_id"]
        return agent.generate(
            cluster_id=cluster_id,
            representative_id=cluster["representative_id"],
            peer_id=cluster["peer_id"],
            classification=cluster,
            cluster_article_ids=cluster_map.get(cluster_id, []),
        )

    cards: list[dict] = []
    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = [ex.submit(_generate_one, c) for c in state["classified_clusters"]]
        for future in as_completed(futures):
            card = future.result()
            if card:
                cards.append(card)

    log.info("이슈카드 생성 완료 | cards=%d", len(cards))
    return {**state, "issue_cards": cards}


def implication_node(state: IngestionState) -> IngestionState:
    """시사점 생성 — notable 이상 등급만 GPT-4o 병렬 호출."""
    from src.agents.implication_agent import ImplicationAgent

    agent = ImplicationAgent()
    notable_cards = [c for c in state["issue_cards"] if c.get("importance") != "reference"]
    reference_cards = [c for c in state["issue_cards"] if c.get("importance") == "reference"]

    implications: list[dict] = [{"card_id": c["id"], "skipped": True} for c in reference_cards]

    def _generate_one(card: dict) -> dict:
        result = agent.generate(card)
        result["card_id"] = card["id"]
        return result

    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {ex.submit(_generate_one, card): card for card in notable_cards}
        for future in as_completed(futures):
            card = futures[future]
            result = future.result()
            implications.append(result)
            card["implication"] = result  # 카드에 시사점 붙이기

    log.info(
        "시사점 생성 완료 | total=%d skipped=%d",
        len(implications),
        len(reference_cards),
    )
    return {**state, "implications": implications}


def validation_node(state: IngestionState) -> IngestionState:
    """SC 검증 + 이슈카드 DB 저장 — notable 카드 병렬 검증."""
    from src.agents.validation_agent import ValidationAgent
    from src.db.article_store import save_issue_card

    agent = ValidationAgent()
    implication_map = {i["card_id"]: i for i in state["implications"] if not i.get("skipped")}

    def _validate_one(card: dict) -> dict:
        card_id = card["id"]
        implication = implication_map.get(card_id, {})
        if implication:
            validation = agent.validate(card, implication)
        else:
            validation = {"pass": True, "sc_score": 1.0, "reason": "reference 등급 스킵"}
        card["validation"] = validation
        save_issue_card(card)
        return {"card_id": card_id, **validation}

    results: list[dict] = []
    human_review_flags: list[int] = []

    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {ex.submit(_validate_one, card): card for card in state["issue_cards"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not result["pass"]:
                card = futures[future]
                human_review_flags.append(card.get("cluster_id", 0))
                log.warning(
                    "SC 검증 실패 — human review 필요 | card_id=%s reason=%s",
                    result["card_id"], result.get("reason"),
                )

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
