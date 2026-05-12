import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    GenSearchRequest,
    GenSearchResult,
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AXIS AI 서버 시작")
    yield
    log.info("AXIS AI 서버 종료")


app = FastAPI(
    title="AXIS AI Internal API",
    description="SpringBoot에서만 호출하는 내부 AI 서버",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """헬스체크 — docker-compose healthcheck 대상"""
    db_ok = _check_db()
    qdrant_ok = _check_qdrant()
    return HealthResponse(
        status="ok" if (db_ok and qdrant_ok) else "degraded",
        models_loaded={"bge_m3": False, "bge_reranker": False},  # 실제 로드 시 True
        db_connected=db_ok,
        qdrant_connected=qdrant_ok,
    )


@app.post("/pipeline/run", response_model=PipelineRunResponse, status_code=202)
async def run_pipeline(request: PipelineRunRequest):
    """수집 파이프라인 비동기 실행 (SpringBoot 스케줄러가 매시간 호출)"""
    import uuid

    task_id = str(uuid.uuid4())
    log.info("수집 파이프라인 시작 | company=%s task_id=%s", request.company, task_id)
    # TODO: ingestion_graph.py 실행 (백그라운드 태스크)
    return PipelineRunResponse(
        task_id=task_id,
        status="accepted",
        message=f"파이프라인 큐 등록 완료 - company: {request.company}",
    )


@app.post("/pipeline/delivery")
async def run_delivery():
    """전달 파이프라인 실행 (SpringBoot 스케줄러가 08:30에 호출)"""
    log.info("전달 파이프라인 시작")
    # TODO: delivery_graph.py 실행
    return {"status": "accepted", "message": "전달 파이프라인 큐 등록 완료"}


@app.get("/api/cards")
async def list_cards(sort: str = "exposure_desc", limit: int = 30, offset: int = 0):
    """프론트 CardNewsItem schema에 맞는 카드뉴스 목록을 반환한다."""
    del sort
    fetch_limit = max(limit + offset, limit)
    items = _build_card_news_items(limit=fetch_limit, today_only=False)
    page_items = items[offset : offset + limit]
    return _api_response(
        {
            "items": page_items,
            "total": len(items),
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/cards/today")
async def list_today_cards(limit: int = 10):
    """최근 24시간 뉴스 기반 카드뉴스 목록을 반환한다."""
    items = _build_card_news_items(limit=limit, today_only=True)
    return _api_response(
        {
            "date": datetime.now(UTC).date().isoformat(),
            "items": items,
            "total": len(items),
        }
    )


def _api_response(data: dict) -> dict:
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """BGE-M3 하이브리드 검색 (Dense + Sparse RRF)"""
    log.info("검색 요청 | query=%s company=%s", request.query, request.company)
    # TODO: hybrid_search.py 실행
    return SearchResponse(hits=[], total=0)


@app.post("/gen-search", response_model=GenSearchResult)
async def gen_search(request: GenSearchRequest):
    """Generative Search — RAG + GPT-4o + SC 검증"""
    log.info("Generative Search | query=%s", request.query)
    # TODO: RAG + LLM 실행
    return GenSearchResult(
        answer="(AI 서버 초기화 중)",
        sources=[],
        sc_passed=False,
        sc_score=0.0,
    )


@app.post("/weak-signal/run")
async def run_weak_signal():
    """약한 신호 감지기 실행 (주 1회)"""
    log.info("약한 신호 감지기 실행")
    # TODO: weak_signal_agent.py 실행
    return {"status": "accepted"}


def _check_db() -> bool:
    try:
        from sqlalchemy import text

        from src.db.postgres import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("DB 연결 확인 실패: %s", e)
        return False


def _check_qdrant() -> bool:
    try:
        from src.db.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception as e:
        log.warning("Qdrant 연결 확인 실패: %s", e)
        return False


def _build_card_news_items(limit: int, today_only: bool) -> list[dict]:
    """DB 대표 클러스터를 요약·분석·카드뉴스 에이전트 흐름으로 변환한다."""
    from src.agents.card_news_agent import CardNewsAgent
    from src.agents.classification_agent import ClassificationAgent
    from src.agents.news_analysis_agent import PeerNewsAnalysisAgent
    from src.agents.news_summary_agent import PeerNewsSummaryAgent
    from src.db.article_store import get_articles_by_ids, list_card_news_cluster_candidates

    candidate_limit = min(max(limit * 5, limit), 30)
    candidates = list_card_news_cluster_candidates(
        limit=candidate_limit,
        today_only=today_only,
    )
    if not candidates:
        return []

    classifier = ClassificationAgent()
    summary_agent = PeerNewsSummaryAgent()
    analysis_agent = PeerNewsAnalysisAgent()
    card_agent = CardNewsAgent()
    cards: list[dict] = []
    seen_cluster_ids: set[int] = set()

    for candidate in candidates:
        cluster_id = int(candidate["cluster_id"])
        if cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)

        representative_id = int(candidate["representative_id"])
        article_ids = [int(article_id) for article_id in candidate.get("article_ids") or []]
        if representative_id not in article_ids:
            article_ids.insert(0, representative_id)

        articles = get_articles_by_ids(article_ids[:5])
        company = _first_company(candidate.get("company"))
        classification = classifier.classify(
            cluster_id=cluster_id,
            representative_id=representative_id,
            cluster_article_ids=article_ids,
            company=company,
        )
        summary = summary_agent.summarize_articles(
            cluster_id=cluster_id,
            representative_id=representative_id,
            articles=articles,
            cluster_article_ids=article_ids,
        )
        if not summary.get("is_valid_summary"):
            continue

        analysis = analysis_agent.analyze(
            summary=summary,
            classification=classification,
            cluster_metadata={
                "cluster_size": int(candidate.get("cluster_size") or len(article_ids)),
                "source_count": len({article.get("source_name") for article in articles}),
                "source_names": sorted({str(article.get("source_name")) for article in articles}),
            },
        )
        cards.append(
            card_agent.generate(
                summary=summary,
                analysis=analysis,
                classification=classification,
                articles=articles,
            )
        )
        if len(cards) >= limit:
            break

    return cards


def _first_company(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""
