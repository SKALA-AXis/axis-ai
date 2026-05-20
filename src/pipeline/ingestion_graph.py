"""수집 파이프라인 v3 — 1시간마다 실행.

ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.

v3 변경:
- ImplicationAgent 보류 → implication_node 제거
- CardNewsAgent 내부 validation/evidence_chain 사용
- PreprocessingService: RAW 대상 조회 + source 라우팅 + dedup + classification
- card_news persist + vector_index 노드 + pipeline_logs 누적
"""

import logging
import operator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Callable, TypedDict, cast

from langgraph.graph import END, StateGraph

from src.db.article_store import save_pipeline_log

log = logging.getLogger(__name__)

# GPT-4o rate limit 고려: 카드 병렬 호출 수
_GPT_WORKERS = 5


class IngestionState(TypedDict):
    company: list[str]
    trigger_type: str
    collected_since: str | None
    crawl_run_id: str | None
    raw_article_ids: list[int]
    relevant_ids: list[int]
    official_document_ids: list[int]
    parsed_document_ids: list[int]
    industry_document_ids: list[int]
    structured_signal_ids: list[int]
    skipped_preprocess_ids: list[int]
    cluster_map: dict  # {cluster_id: [article_ids]}
    representative_ids: list[int]
    classified_clusters: list[dict]
    card_news: list[dict]
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
                companies = state.get("company") or []
                company_label = companies[0] if len(companies) == 1 else None
                save_pipeline_log(
                    step=step_name,
                    company=company_label,
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


@_logged_step("crawl", "company", "raw_article_ids")
def crawl_node(state: IngestionState) -> IngestionState:
    """처리 대기 중인 RAW 기사 ID를 DB에서 조회한다."""
    from src.preprocessing.preprocessing import PreprocessingService

    raw_ids = PreprocessingService().load_raw_ids(
        state["company"],
        collected_since=state.get("collected_since"),
        crawl_run_id=state.get("crawl_run_id"),
    )
    log.info(
        "RAW 기사 로드 | company=%s collected_since=%s crawl_run_id=%s count=%d",
        state["company"],
        state.get("collected_since"),
        state.get("crawl_run_id"),
        len(raw_ids),
    )
    return {**state, "raw_article_ids": raw_ids}


@_logged_step("preprocess_route", "raw_article_ids", "relevant_ids")
def preprocess_route_node(state: IngestionState) -> IngestionState:
    """source_type별 DB 전처리 라우팅."""
    from src.preprocessing.preprocessing import PreprocessingService

    route_result = PreprocessingService().route_by_source(state.get("raw_article_ids", []))
    return cast(IngestionState, {**state, **route_result})


@_logged_step("dedup", "relevant_ids", "representative_ids")
def dedup_node(state: IngestionState) -> IngestionState:
    """Gate 3: BGE-M3 코사인 유사도 클러스터링."""
    from src.preprocessing.preprocessing import PreprocessingService

    cluster_map, rep_ids = PreprocessingService().deduplicate(state.get("relevant_ids", []))
    return {**state, "cluster_map": cluster_map, "representative_ids": rep_ids}


@_logged_step("classify", "representative_ids", "classified_clusters")
def classify_node(state: IngestionState) -> IngestionState:
    """v3 분류 — 클러스터별 sector + 결정적 노출도 + event_type."""
    from src.preprocessing.preprocessing import PreprocessingService

    classified = PreprocessingService(max_workers=_GPT_WORKERS).classify_clusters(
        representative_ids=state["representative_ids"],
        cluster_map=state["cluster_map"],
        requested_companies=state["company"],
    )
    return {**state, "classified_clusters": classified}


@_logged_step("card_news", "classified_clusters", "card_news")
def card_news_node(state: IngestionState) -> IngestionState:
    """분석 대상 전체를 AnalysisPipelineRunner로 실행해 카드뉴스를 생성한다."""
    from src.config.company_tiers import SELF_COMPANY_IDS
    from src.db.article_store import save_card_news
    from src.pipeline.analysis_pipeline import AnalysisPipelineRunner

    runner = AnalysisPipelineRunner()
    cluster_map = state["cluster_map"]

    def _generate_cluster_card(cluster: dict) -> dict:
        if cluster.get("company") in SELF_COMPANY_IDS:
            log.info(
                "카드뉴스 생성 제외 | cluster=%s company=%s reason=self company",
                cluster.get("cluster_id"),
                cluster.get("company"),
            )
            return {}

        cluster_id = cluster["cluster_id"]
        article_ids = cluster_map.get(cluster_id, [])
        result = runner.run_cluster(
            cluster_id=cluster_id,
            representative_id=cluster["representative_id"],
            cluster_article_ids=article_ids,
            classification=cluster,
            save_card=True,
        )
        analysis_package = result.get("analysis_package") or {}
        integrated_issue = analysis_package.get("integrated_issue") or analysis_package.get(
            "summary",
            {},
        )
        if not integrated_issue.get("is_valid_summary"):
            log.info(
                "카드뉴스 생성 제외 | cluster=%s company=%s reason=invalid_summary:%s",
                cluster_id,
                cluster.get("company"),
                integrated_issue.get("reason"),
            )
            return {}
        return result.get("card_news") or {}

    def _generate_raw_article_card(raw_article_id: int, source_group: str) -> dict:
        result = runner.run_raw_article(raw_article_id=raw_article_id, save_card=False)
        card = result.get("card_news") or {}
        if not card:
            log.info(
                "문서/신호 카드뉴스 생성 제외 | raw_article_id=%s group=%s reason=no_card",
                raw_article_id,
                source_group,
            )
            return {}
        if card.get("company") in SELF_COMPANY_IDS or card.get("peer_id") in SELF_COMPANY_IDS:
            log.info(
                "문서/신호 카드뉴스 생성 제외 | "
                "raw_article_id=%s group=%s company=%s reason=self company",
                raw_article_id,
                source_group,
                card.get("company") or card.get("peer_id"),
            )
            return {}
        save_card_news(card)
        card.setdefault("source_group", source_group)
        return card

    document_targets = _document_analysis_targets(state)
    cards: list[dict] = []
    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = [
            *[ex.submit(_generate_cluster_card, c) for c in state["classified_clusters"]],
            *[
                ex.submit(_generate_raw_article_card, raw_article_id, source_group)
                for source_group, raw_article_ids in document_targets.items()
                for raw_article_id in raw_article_ids
            ],
        ]
        for future in as_completed(futures):
            card = future.result()
            if card:
                cards.append(card)

    log.info(
        "카드 뉴스 생성 완료 | news_clusters=%d document_targets=%d cards=%d",
        len(state["classified_clusters"]),
        sum(len(ids) for ids in document_targets.values()),
        len(cards),
    )
    return {**state, "card_news": cards}


def _document_analysis_targets(state: IngestionState) -> dict[str, list[int]]:
    """뉴스 클러스터 외 분석 대상 문서/구조화 신호 ID를 모은다."""
    return {
        "official_document": _dedupe_positive_ints(state.get("official_document_ids", [])),
        "parsed_document": _dedupe_positive_ints(state.get("parsed_document_ids", [])),
        "industry_document": _dedupe_positive_ints(state.get("industry_document_ids", [])),
        "structured_signal": _dedupe_positive_ints(state.get("structured_signal_ids", [])),
    }


def _dedupe_positive_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@_logged_step("vector_index", "card_news", "indexed_vector_ids")
def vector_index_node(state: IngestionState) -> IngestionState:
    """카드 자체 validation 통과분을 BGE-M3로 임베딩해 Qdrant axis_main에 인덱싱."""
    from src.rag.vector_index import index_card

    targets = [
        c
        for c in state["card_news"]
        if bool(c.get("validation_pass", c.get("validation", {}).get("pass", True)))
    ]

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
    graph.add_node("preprocess_route", preprocess_route_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("classify", classify_node)
    graph.add_node("card_news", card_news_node)
    graph.add_node("vector_index", vector_index_node)

    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "preprocess_route")
    graph.add_edge("preprocess_route", "dedup")
    graph.add_edge("dedup", "classify")
    graph.add_edge("classify", "card_news")
    graph.add_edge("card_news", "vector_index")
    graph.add_edge("vector_index", END)

    return graph.compile()  # type: ignore[return-value]


ingestion_graph = build_ingestion_graph()
