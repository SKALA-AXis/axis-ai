"""수집 파이프라인 v3 — 1시간마다 실행.

ADR 0004: 수집 파이프라인과 전달 파이프라인 분리 원칙.

v3 변경:
- ImplicationAgent 보류 → implication_node 제거
- ValidationAgent (SC 검증) 보류 → EvidenceAgent (근거 첨부)로 전환
- ClassificationAgent v3: sector + 결정적 노출도
- evidence_chain 테이블 persist + vector_index 노드 + pipeline_logs 누적
"""

import json
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
RELEVANCE_SOURCE_TYPES = {"news"}
OFFICIAL_DOCUMENT_SOURCE_TYPES = {"official"}
COMPANY_SITE_DOCUMENT_SOURCE_TYPES = {"company_site"}
PARSED_DOCUMENT_SOURCE_TYPES = {"dart", "ir", "securities_report"}
STRUCTURED_SIGNAL_SOURCE_TYPES = {"job", "market_data", "search_trend", "social"}


class IngestionState(TypedDict):
    company: list[str]
    trigger_type: str
    collected_since: str | None
    crawl_run_id: str | None
    raw_article_ids: list[int]
    credible_ids: list[int]
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


def _first_company(article: dict) -> str:
    company = article.get("company")

    if isinstance(company, list) and company:
        return str(company[0])

    if isinstance(company, str):
        try:
            parsed = json.loads(company)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except json.JSONDecodeError:
            return company

    return ""


def _company_for_context(article: dict, requested_companies: list[str]) -> str:
    article_companies = _company_list(article)

    for company in requested_companies:
        if company in article_companies:
            return company

    return (
        article_companies[0]
        if article_companies
        else (requested_companies[0] if requested_companies else "")
    )


def _company_list(article: dict) -> list[str]:
    company = article.get("company")

    if isinstance(company, list):
        return [str(value) for value in company if value]

    if isinstance(company, str):
        try:
            parsed = json.loads(company)
            if isinstance(parsed, list):
                return [str(value) for value in parsed if value]
        except json.JSONDecodeError:
            stripped = company.strip()
            return [stripped] if stripped else []

    return []


def _source_type(article: dict) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _article_for_agent(article: dict) -> dict:
    item = dict(article)
    metadata = item.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    item["metadata"] = metadata if isinstance(metadata, dict) else {}
    item["extra"] = item["metadata"]
    return item


# ── 노드 구현 ──────────────────────────────────────────────────


@_logged_step("crawl", "company", "raw_article_ids")
def crawl_node(state: IngestionState) -> IngestionState:
    """처리 대기 중인 RAW 기사 ID를 DB에서 조회한다."""
    from src.agents.crawler_agent import CrawlerAgent

    raw_ids = CrawlerAgent().load_raw_ids(
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


@_logged_step("credibility", "raw_article_ids", "credible_ids")
def credibility_node(state: IngestionState) -> IngestionState:
    """Gate 2: credibility_score 기준 신뢰도 필터."""
    from src.agents.credibility_agent import CredibilityAgent

    credible_ids, skipped = CredibilityAgent().filter(state["raw_article_ids"])
    log.info("Gate 2 완료 | credible=%d skipped=%d", len(credible_ids), len(skipped))
    return {**state, "credible_ids": credible_ids}


@_logged_step("preprocess_route", "credible_ids", "relevant_ids")
def preprocess_route_node(state: IngestionState) -> IngestionState:
    """source_type별 DB 전처리 라우팅."""
    from src.parsers.parser_quality import analyze_parser_quality_article
    from src.parsers.parser_router import DocumentParserRouter
    from src.agents.relevance_agent import RelevanceAgent
    from src.db.article_store import get_articles_by_ids, update_preprocess_status

    credible_ids = state.get("credible_ids", [])
    if not credible_ids:
        return {
            **state,
            "relevant_ids": [],
            "official_document_ids": [],
            "parsed_document_ids": [],
            "industry_document_ids": [],
            "structured_signal_ids": [],
            "skipped_preprocess_ids": [],
        }

    articles = get_articles_by_ids(credible_ids)
    by_source: dict[str, list[int]] = {}
    for article in articles:
        by_source.setdefault(_source_type(article), []).append(int(article["id"]))

    relevant_ids: list[int] = []
    official_document_ids: list[int] = []
    parsed_document_ids: list[int] = []
    industry_document_ids: list[int] = []
    structured_signal_ids: list[int] = []
    skipped_ids: list[int] = []

    relevance_ids = [
        article_id
        for source_type in RELEVANCE_SOURCE_TYPES
        for article_id in by_source.get(source_type, [])
    ]
    if relevance_ids:
        passed, skipped = RelevanceAgent().filter(relevance_ids)
        relevant_ids.extend(passed)
        skipped_ids.extend(skipped)

    for article in articles:
        article_id = int(article["id"])
        source_type = _source_type(article)

        if source_type in RELEVANCE_SOURCE_TYPES:
            continue

        agent_article = _article_for_agent(article)

        if source_type in OFFICIAL_DOCUMENT_SOURCE_TYPES:
            official_document_ids.append(article_id)
            update_preprocess_status(
                article_id,
                "PREPROCESSED_OFFICIAL_DOCUMENT",
                {
                    "document_scope": "company_official",
                    "preprocess_note": (
                        "official 문서는 회사별 공식 원문으로 보존. "
                        "기사 relevance/dedup/classification 단계는 생략하고 "
                        "추후 동향 분석에서 사용"
                    ),
                },
            )
            continue

        if source_type in COMPANY_SITE_DOCUMENT_SOURCE_TYPES:
            official_document_ids.append(article_id)
            update_preprocess_status(
                article_id,
                "PREPROCESSED_COMPANY_SITE_DOCUMENT",
                {
                    "document_scope": "company_site",
                    "preprocess_note": (
                        "company_site 문서는 회사 공식 홈페이지의 정적/반정적 원문으로 보존. "
                        "뉴스룸 relevance/dedup/classification 단계는 생략하고 "
                        "회사 지식베이스/과거 분석에서 사용"
                    ),
                },
            )
            continue

        if source_type in PARSED_DOCUMENT_SOURCE_TYPES:
            item, ok, reason = analyze_parser_quality_article(agent_article)
            parser_result = item.get("parser_result") or {}
            metadata_patch = {
                "parser_result": parser_result,
                "parser_quality_score": item.get("parser_quality_score"),
                "parser_quality_label": item.get("parser_quality_label"),
                "parser_quality_reason": item.get("parser_quality_reason"),
            }
            if source_type == "dart":
                metadata_patch.update(
                    {
                        "period": parser_result.get("period"),
                        "period_year": parser_result.get("period_year"),
                        "period_quarter": parser_result.get("period_quarter"),
                        "period_type": parser_result.get("period_type"),
                        "financial_record": parser_result.get("financial_record"),
                        "topics": parser_result.get("topics"),
                        "topic_signals": parser_result.get("topic_signals"),
                        "dart_sections": parser_result.get("sections"),
                        "dart_document_chunks": parser_result.get("document_chunks"),
                    }
                )
            elif source_type == "ir":
                metadata_patch.update(
                    {
                        "period": parser_result.get("period"),
                        "period_year": parser_result.get("period_year"),
                        "period_quarter": parser_result.get("period_quarter"),
                        "period_type": parser_result.get("period_type"),
                        "financial_record": parser_result.get("financial_record"),
                        "topics": parser_result.get("topics"),
                        "topic_signals": parser_result.get("topic_signals"),
                        "ir_sections": parser_result.get("sections"),
                        "ir_document_chunks": parser_result.get("document_chunks"),
                    }
                )
            if ok:
                parsed_document_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    "PREPROCESSED_PARSED_DOCUMENT",
                    {
                        **metadata_patch,
                        "document_scope": "company_document",
                        "preprocess_note": (
                            f"{source_type} 문서는 parser quality check 후 보존. "
                            "기사 relevance/dedup/classification 단계는 생략"
                        ),
                    },
                )
            else:
                skipped_ids.append(article_id)
                update_preprocess_status(
                    article_id,
                    "SKIPPED_PARSER_QUALITY",
                    metadata_patch,
                    error_message=reason,
                )
            continue

        if source_type == "trend_report":
            parser_result = DocumentParserRouter().parse_article(agent_article)
            industry_document_ids.append(article_id)
            update_preprocess_status(
                article_id,
                "PREPROCESSED_INDUSTRY_DOCUMENT",
                {
                    "parser_result": parser_result,
                    "document_scope": "industry_trend",
                    "preprocess_note": (
                        "산업 동향 문서는 기사 relevance/signal 축약 없이 "
                        "추후 본문 분석 대상으로 보존"
                    ),
                },
            )
            continue

        if source_type in STRUCTURED_SIGNAL_SOURCE_TYPES:
            structured_signal_ids.append(article_id)
            update_preprocess_status(
                article_id,
                "PREPROCESSED_STRUCTURED_SIGNAL",
                {
                    "signal_scope": source_type,
                    "preprocess_note": (
                        f"{source_type} 데이터는 기사/문서가 아닌 구조화 신호로 보존. "
                        "급변/급증 탐지 및 종합 분석 단계에서 사용"
                    ),
                },
            )
            continue

        skipped_ids.append(article_id)
        update_preprocess_status(
            article_id,
            "SKIPPED_PREPROCESS_UNSUPPORTED_SOURCE",
            {"skip_reason": f"{source_type or 'unknown'} source_type은 현재 전처리 대상이 아님"},
        )

    log.info(
        (
            "전처리 라우팅 완료 | relevant=%d official_docs=%d parsed_docs=%d "
            "industry_docs=%d structured=%d skipped=%d"
        ),
        len(relevant_ids),
        len(official_document_ids),
        len(parsed_document_ids),
        len(industry_document_ids),
        len(structured_signal_ids),
        len(skipped_ids),
    )
    return {
        **state,
        "relevant_ids": relevant_ids,
        "official_document_ids": official_document_ids,
        "parsed_document_ids": parsed_document_ids,
        "industry_document_ids": industry_document_ids,
        "structured_signal_ids": structured_signal_ids,
        "skipped_preprocess_ids": skipped_ids,
    }


@_logged_step("dedup", "relevant_ids", "representative_ids")
def dedup_node(state: IngestionState) -> IngestionState:
    """Gate 3: BGE-M3 코사인 유사도 클러스터링."""
    from src.agents.dedup_agent import DeduplicationAgent

    cluster_map, rep_ids = DeduplicationAgent().deduplicate(state.get("relevant_ids", []))
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
        company = _company_for_context(rep_articles.get(rep_id, {}), state["company"])
        result = agent.classify(
            cluster_id=cluster_id,
            representative_id=rep_id,
            cluster_article_ids=article_ids,
            company=company,
        )
        return {
            "cluster_id": cluster_id,
            "representative_id": rep_id,
            "company": company,
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


@_logged_step("card_news", "classified_clusters", "card_news")
def card_news_node(state: IngestionState) -> IngestionState:
    """카드 뉴스 생성 — 클러스터 사실 요약 결과 기반. (구 issue_card_node)"""
    from src.agents.issue_card_agent import IssueCardAgent
    from src.agents.news_summary_agent import PeerNewsSummaryAgent

    agent = IssueCardAgent()
    summary_agent = PeerNewsSummaryAgent()
    cluster_map = state["cluster_map"]

    def _generate_one(cluster: dict) -> dict:
        cluster_id = cluster["cluster_id"]
        article_ids = cluster_map.get(cluster_id, [])
        summary = summary_agent.summarize(
            cluster_id=cluster_id,
            representative_id=cluster["representative_id"],
            cluster_article_ids=article_ids,
            max_cluster_articles=max(len(article_ids), 1),
        )
        return agent.generate(
            cluster_id=cluster_id,
            representative_id=cluster["representative_id"],
            company=cluster["company"],
            classification=cluster,
            cluster_article_ids=article_ids,
            summary=summary,
        )

    cards: list[dict] = []
    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = [ex.submit(_generate_one, c) for c in state["classified_clusters"]]
        for future in as_completed(futures):
            card = future.result()
            if card:
                cards.append(card)

    log.info("카드 뉴스 생성 완료 | cards=%d", len(cards))
    return {**state, "card_news": cards}


@_logged_step("evidence", "card_news", "evidence_results")
def evidence_node(state: IngestionState) -> IngestionState:
    """v3 검증 체인 첨부 + 카드 DB 저장 + evidence_chain 테이블 persist."""
    from src.agents.evidence_agent import EvidenceAgent
    from src.db.article_store import save_card_news, save_evidence_chain

    agent = EvidenceAgent()
    cluster_map = state["cluster_map"]

    def _attach_one(card: dict) -> dict:
        cluster_id = card.get("cluster_id")
        result = agent.attach(card, cluster_article_ids=cluster_map.get(cluster_id, []))
        save_card_news(card)
        save_evidence_chain(
            card_news_id=card.get("id") or "",
            chain=card.get("evidence_chain", {}),
            passed=bool(result.get("pass")),
            missing=list(result.get("missing", [])),
        )
        return {"card_id": card.get("id"), **result}

    results: list[dict] = []
    human_review_flags: list[int] = []

    with ThreadPoolExecutor(max_workers=_GPT_WORKERS) as ex:
        futures = {ex.submit(_attach_one, card): card for card in state["card_news"]}
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


@_logged_step("vector_index", "card_news", "indexed_vector_ids")
def vector_index_node(state: IngestionState) -> IngestionState:
    """검증 통과 카드를 BGE-M3로 임베딩해 Qdrant axis_main에 인덱싱."""
    from src.rag.vector_index import index_card

    passed_card_ids = {r["card_id"] for r in state["evidence_results"] if r.get("pass")}
    targets = [c for c in state["card_news"] if c.get("id") in passed_card_ids]

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
    graph.add_node("preprocess_route", preprocess_route_node)
    graph.add_node("dedup", dedup_node)
    graph.add_node("classify", classify_node)
    graph.add_node("card_news", card_news_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("vector_index", vector_index_node)

    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "credibility")
    graph.add_edge("credibility", "preprocess_route")
    graph.add_edge("preprocess_route", "dedup")
    graph.add_edge("dedup", "classify")
    graph.add_edge("classify", "card_news")
    graph.add_edge("card_news", "evidence")
    graph.add_edge("evidence", "vector_index")
    graph.add_edge("vector_index", END)

    return graph.compile()  # type: ignore[return-value]


ingestion_graph = build_ingestion_graph()
