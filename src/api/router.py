import logging
from contextlib import asynccontextmanager

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
    allow_origins=["http://localhost:8080"],
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
    log.info("수집 파이프라인 시작 | peer_ids=%s task_id=%s", request.peer_ids, task_id)
    # TODO: ingestion_graph.py 실행 (백그라운드 태스크)
    return PipelineRunResponse(
        task_id=task_id,
        status="accepted",
        message=f"파이프라인 큐 등록 완료 — peer_ids: {request.peer_ids}",
    )


@app.post("/pipeline/delivery")
async def run_delivery():
    """전달 파이프라인 실행 (SpringBoot 스케줄러가 08:30에 호출)"""
    log.info("전달 파이프라인 시작")
    # TODO: delivery_graph.py 실행
    return {"status": "accepted", "message": "전달 파이프라인 큐 등록 완료"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """BGE-M3 하이브리드 검색 (Dense + Sparse RRF)"""
    log.info("검색 요청 | query=%s peer_id=%s", request.query, request.peer_id)
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
