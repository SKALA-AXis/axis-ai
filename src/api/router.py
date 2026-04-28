import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    CrawlPreviewRequest,
    CrawlPreviewResponse,
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
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
    """로컬 API 헬스체크."""
    return HealthResponse(
        status="ok",
        models_loaded={"bge_m3": False, "bge_reranker": False},  # 실제 로드 시 True
    )


@app.post("/crawl/preview", response_model=CrawlPreviewResponse)
async def crawl_preview(request: CrawlPreviewRequest):
    """현재 크롤러 실행 결과를 crawl_results 폴더에 JSON으로 저장한다."""
    from src.crawler.preview import run_all_crawl_previews, run_crawl_preview

    if request.peer_id:
        output_paths = await run_crawl_preview(
            peer_id=request.peer_id,
            topics=request.topics,
            corp_code=request.corp_code,
            recent_days=request.recent_days,
            mode=request.mode,
        )
    else:
        output_paths = await run_all_crawl_previews(
            recent_days=request.recent_days,
            mode=request.mode,
        )

    return CrawlPreviewResponse(
        status="saved",
        output_paths=[str(output_path) for output_path in output_paths],
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


@app.post("/weak-signal/run")
async def run_weak_signal():
    """약한 신호 감지기 실행 (주 1회)"""
    log.info("약한 신호 감지기 실행")
    # TODO: weak_signal_agent.py 실행
    return {"status": "accepted"}
