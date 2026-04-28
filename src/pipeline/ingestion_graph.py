"""수집 파이프라인 v3 — 1시간마다 실행.

ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.

v3 변경:
- ImplicationAgent 보류 → implication_node 제거
- ValidationAgent (SC 검증) 보류 → EvidenceAgent (근거 첨부)로 전환
- ClassificationAgent v3: sector + 결정적 노출도
- evidence_chain 테이블 persist + vector_index 노드 + pipeline_logs 누적
"""

import logging
import operator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, StateGraph

from src.db.article_store import save_pipeline_log

log = logging.getLogger(__name__)

# GPT-4o rate limit 고려: 분류·카드 병렬 호출 수
_GPT_WORKERS = 5


class IngestionState(TypedDict):
    peer_ids: list[str]
    trigger_type: str
    raw_article_ids: list[int]
    credible_ids: list[int]
    cluster_map: dict  # {cluster_id: [article_ids]}
    representative_ids: list[int]
    classified_clusters: list[dict]
    issue_cards: list[dict]
    evidence_results: list[dict]  # v3: EvidenceAgent 첨부 결과
    indexed_vector_ids: list[str]  # v3: Qdrant axis_main 삽입 vector_id 목록
    errors: Annotated[list[str], operator.add]
    human_review_flags: list[int]


# ── 단계별 통계 헬퍼 ────────────────────────────────────────────


def _logged_step(
    step_name: str,
    input_key: str,
    output_key: str,
) -> Callable[[Callable], Callable]:
    """노드 함수를 감싸 elapsed_ms·input/output 카운트를 pipeline_logs에 기록."""

    def decorator(func: Callable) -> Callable:
        def wrapper(state: IngestionState) -> IngestionState:
            t0 = time.perf_counter()
            err: str | None = None
            new_state: IngestionState
            try:
                new_state = func(state)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                raise
            finally:
                elapsed = int((time.perf_counter() - t0) * 1000)
                input_count = _count_value(state.get(input_key))
                output_count = _count_value((new_state if err is None else state).get(output_key))
                peers = state.get("peer_ids") or []
                peer_label = peers[0] if len(peers) == 1 else None
                save_pipeline_log(
                    step=step_name,
                    peer_id=peer_label,
                    input_count=input_count,
                    output_count=output_count,
                    elapsed_ms=elapsed,
                    error_msg=err,
                )
            return new_state

        return wrapper

    return decorator


def _count_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value)
    return 1


# ── 노드 구현 ──────────────────────────────────────────────────


@_logged_step("crawl", "peer_ids", "raw_article_ids")
def crawl_node(state: IngestionState) -> IngestionState:
    """처리 대기 중인 RAW 기사 ID를 DB에서 조회한다."""
    from src.agents.crawler_agent import CrawlerAgent

    raw_ids = CrawlerAgent().load_raw_ids(state["peer_ids"])
    log.info("RAW 기사 로드 | peer_ids=%s count=%d", state["peer_ids"], len(raw_ids))
    return {**state, "raw_article_ids": raw_ids}


@_logged_step("credibility", "raw_article_ids", "credible_ids")
def credibility_node(state: IngestionState) -> IngestionState:
    """Gate 2: credibility_score 기준 신뢰도 필터."""
    from src.agents.credibility_agent import CredibilityAgent

    credible_ids, skipped = CredibilityAgent().filter(state["raw_article_ids"])
    log.info("Gate 2 완료 | credible=%d skipped=%d", len(credible_ids), len(skipped))
    return {**state, "credible_ids": credible_ids}


@_logged_step("dedup", "credible_ids", "representative_ids")
def dedup_node(state: IngestionState) -> IngestionState:
    """Gate 3: BGE-M3 코사인 유사도 클러스터링."""
    from src.agents.dedup_agent import DeduplicationAgent

    cluster_map, rep_ids = DeduplicationAgent().deduplicate(state["credible_ids"])
    log.info("Gate 3 완료 | clusters=%d reps=%d", len(cluster_map), len(rep_ids))
    return {**state, "cluster_map": cluster_map, "representative_ids": rep_ids}


@_logged_step("classify", "representative_ids", "classified_clusters")
def classify_node(state: IngestionState) -> IngestionState:
    """v3 분류 — 클러스터별 sector + 결정적 노출도 + event_type."""
    from src.agents.classification_agent import ClassificationAgent
    from src.db.article_store import get_articles_by_ids

    agent = ClassificationAgent()
    rep_articles = {a["id"]: a for a in get_articles_by_ids(state["representative_ids"])}
    cluster_map = state["cluster_map"]

    def _classify_one(cluster_id: int, article_ids: list[int]) -> dict | None:
        rep_id = next(
            (aid for aid in article_ids if aid in rep_articles),
            article_ids[0] if article_ids else None,
        )
        if rep_id is None:
            return None
        peer_id = rep_articles.get(rep_id, {}).get(
            "peer_id", state["peer_ids"][0] if state["peer_ids"] else ""
        )
        result = agent.classify(
            cluster_id=cluster_id,
            representative_id=rep_id,
            cluster_article_ids=article_ids,
            peer_id=peer_id,
        )
        return {
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "peer_id": peer_id,
            **result,
        }

    classified: list[dict] = []
    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {ex.submit(_classify_one, cid, aids): cid for cid, aids in cluster_map.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                classified.append(result)

    log.info("v3 분류 완료 | clusters=%d", len(classified))
    return {**state, "classified_clusters": classified}


@_logged_step("issue_card", "classified_clusters", "issue_cards")
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


@_logged_step("evidence", "issue_cards", "evidence_results")
def evidence_node(state: IngestionState) -> IngestionState:
    """v3 검증 체인 첨부 + 카드 DB 저장 + evidence_chain 테이블 persist."""
    from src.agents.evidence_agent import EvidenceAgent
    from src.db.article_store import save_evidence_chain, save_issue_card

    agent = EvidenceAgent()
    cluster_map = state["cluster_map"]

    def _attach_one(card: dict) -> dict:
        cluster_id = card.get("cluster_id")
        result = agent.attach(card, cluster_article_ids=cluster_map.get(cluster_id, []))
        save_issue_card(card)
        save_evidence_chain(
            issue_card_id=card.get("id") or "",
            chain=card.get("evidence_chain", {}),
            passed=bool(result.get("pass")),
            missing=list(result.get("missing", [])),
        )
        return {"card_id": card.get("id"), **result}

    results: list[dict] = []
    human_review_flags: list[int] = []

    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {ex.submit(_attach_one, card): card for card in state["issue_cards"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not result["pass"]:
                card = futures[future]
                human_review_flags.append(card.get("cluster_id", 0))

    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(results) - pass_count
    log.info(
        "검증 체인 첨부 + 저장 완료 | total=%d pass=%d fail=%d",
        len(results),
        pass_count,
        fail_count,
    )
    return {
        **state,
        "evidence_results": results,
        "human_review_flags": human_review_flags,
    }


@_logged_step("vector_index", "issue_cards", "indexed_vector_ids")
def vector_index_node(state: IngestionState) -> IngestionState:
    """검증 통과 카드를 BGE-M3로 임베딩해 Qdrant axis_main에 인덱싱."""
    from src.rag.vector_index import index_card

    passed_card_ids = {r["card_id"] for r in state["evidence_results"] if r.get("pass")}
    targets = [c for c in state["issue_cards"] if c.get("id") in passed_card_ids]

    indexed: list[str] = []
    for card in targets:
        try:
            point_id = index_card(card)
            if point_id:
                indexed.append(point_id)
        except Exception as e:
            log.error("Qdrant 인덱싱 실패 | card=%s error=%s", card.get("id"), e)

    log.info(
        "Qdrant 인덱싱 완료 | targets=%d indexed=%d skipped=%d",
        len(targets),
        len(indexed),
        len(targets) - len(indexed),
    )
    return {**state, "indexed_vector_ids": indexed}


# ── 그래프 조립 ────────────────────────────────────────────────


def build_ingestion_graph() -> StateGraph:
    graph = StateGraph(IngestionState)

    graph.add_node("crawl", crawl_node)
    graph.add_node("credibility", credibility_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("classify", classify_node)
    graph.add_node("issue_card", issue_card_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("vector_index", vector_index_node)

    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "credibility")
    graph.add_edge("credibility", "dedup")
    graph.add_edge("dedup", "classify")
    graph.add_edge("classify", "issue_card")
    graph.add_edge("issue_card", "evidence")
    graph.add_edge("evidence", "vector_index")
    graph.add_edge("vector_index", END)

    return graph.compile()  # type: ignore[return-value]


ingestion_graph = build_ingestion_graph()
