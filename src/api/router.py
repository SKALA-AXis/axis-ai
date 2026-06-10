import asyncio
import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.briefing_schemas import BriefingGenerateRequest, BriefingGenerateResponse
from src.api.chat_schemas import ChatTurnRequest, ChatTurnResponse
from src.api.global_trends_schemas import GlobalTrendsRequest, GlobalTrendsResponse
from src.api.insight_schemas import InsightGenerateRequest, InsightGenerateResponse
from src.api.link_verification_schemas import LinkVerificationRequest, LinkVerificationResponse
from src.api.mixer_schemas import MixerAnalysisRequest, MixerAnalysisResponse
from src.api.today_insight_schemas import (
    TodayInsightGenerateRequest,
    TodayInsightGenerateResponse,
)
from src.schemas import (
    BriefingContent,
    BriefingRequest,
    GenSearchRequest,
    GenSearchResult,
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")
SCHEDULED_PREPROCESS_LIMIT = 5000
# Mixer SSE keepalive 주기(초) — nginx/ALB idle timeout(기본 60s)보다 충분히 짧게.
_MIXER_SSE_HEARTBEAT_SEC = 10


def _count_result_items(results: Iterable[Mapping[str, object]], key: str) -> int:
    total = 0
    for result in results:
        value = result.get(key)
        if isinstance(value, list):
            total += len(value)
    return total


def _sum_result_ints(results: Iterable[Mapping[str, object]], key: str) -> int:
    total = 0
    for result in results:
        value = result.get(key)
        if isinstance(value, int):
            total += value
    return total


PREPROCESS_SOURCE_TYPES_BY_SOURCE: dict[str, list[str]] = {
    "naver_news": ["news"],
    "global_newsroom": ["official"],
    "stock": ["market_data"],
    "naver_research": ["securities_report"],
    "jobs": ["job"],
    "company_news": ["official", "company_site"],
    "dart": ["dart"],
    "ir": ["ir"],
    "naver_datalab": ["search_trend"],
    "spri": ["trend_report"],
    "bcg": ["trend_report"],
}

PREPROCESS_SOURCE_TYPES_BY_TRACK: dict[str, list[str]] = {
    "a": ["news", "official"],
    "b": ["market_data", "securities_report"],
    "c": ["job", "official", "company_site"],
    "d": ["dart", "ir", "search_trend", "trend_report"],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AXIS AI 서버 시작")
    # BGE-M3 모델을 startup 시 preload — lazy load 로 인한 첫 cycle 의 메모리 spike
    # (Python heap 확장 + colbert/sparse linear init) 를 제거. 실패해도 첫 호출 시
    # 재시도되므로 서버 startup 자체를 막진 않는다.
    try:
        from src.rag.embedder import preload_embedder

        preload_embedder()
    except Exception as e:
        log.warning("startup preload skipped: %s", e)
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


@app.get("/healthz")
async def healthz():
    """경량 liveness probe — DB/Qdrant 호출 없이 즉시 응답.

    BGE-M3 encode 등 CPU-bound 작업이 event loop 를 막더라도 ASGI 가 응답할 수
    있는 한 200 을 반환한다. kubelet 의 liveness/readiness probe 는 본 endpoint
    를 사용 (k8s manifest 의 probe.path 가 /healthz 로 지정됨).

    상세 헬스 (DB / Qdrant / 모델 로드 상태) 는 /health 로 분리.
    """
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """상세 헬스체크 — DB / Qdrant 연결 + 모델 로드 상태.

    docker-compose healthcheck / 운영 진단용. k8s probe 는 /healthz 사용.
    """
    db_ok = _check_db()
    qdrant_ok = _check_qdrant()
    try:
        from src.rag.embedder import is_loaded as bge_m3_loaded
    except Exception:
        bge_m3_loaded = lambda: False  # noqa: E731
    return HealthResponse(
        status="ok" if (db_ok and qdrant_ok) else "degraded",
        models_loaded={"bge_m3": bge_m3_loaded(), "bge_reranker": False},
        db_connected=db_ok,
        qdrant_connected=qdrant_ok,
    )


@app.post("/pipeline/run", response_model=PipelineRunResponse, status_code=202)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """수집 파이프라인 비동기 실행 (SpringBoot 스케줄러가 매시간 호출)"""
    task_id = str(uuid.uuid4())
    track = request.track.strip().lower()
    if track not in {"a", "b", "c", "d", "all"}:
        raise HTTPException(status_code=400, detail=f"unsupported track: {request.track}")

    log.info(
        ("수집 파이프라인 큐 등록 | track=%s company=%s trigger=%s window=%s~%s task_id=%s"),
        track,
        request.company,
        request.trigger_type,
        request.window_start,
        request.window_end,
        task_id,
    )
    background_tasks.add_task(
        _run_collection_track,
        task_id,
        track,
        request.company,
        request.trigger_type,
        request.window_start,
        request.window_end,
    )
    return PipelineRunResponse(
        task_id=task_id,
        status="accepted",
        message=f"파이프라인 큐 등록 완료 - track: {track}, company: {request.company}",
    )


@app.post("/pipeline/delivery")
async def run_delivery(req: BriefingRequest) -> BriefingContent:
    """전달 파이프라인 — backend Spring @Scheduled (08:30 KST MON-FRI) 가 호출.

    backend 가 PostgreSQL 의 today issue cards 조회 후 cards 전달 → axis-ai 가
    HTML/text 본문 빌더 후 BriefingContent 반환 → backend 의 SesMailService 가
    AWS SES V2 SDK (IRSA) 로 발송.
    """
    from src.pipeline.delivery_graph import delivery_graph

    log.info("전달 파이프라인 시작 | cards=%d", len(req.cards))
    # by_alias=True → JSON camelCase (peerId 등) 유지 — build_briefing_node 와 정합.
    # delivery_graph.invoke 는 sync (LangGraph) — to_thread 로 event loop 보호.
    state = await asyncio.to_thread(
        delivery_graph.invoke,  # type: ignore[attr-defined]
        {
            "cards": [card.model_dump(by_alias=True) for card in req.cards],
            "subject": "",
            "html": "",
            "text": "",
            "errors": [],
        },
    )
    return BriefingContent(
        subject=state["subject"],
        html=state["html"],
        text=state["text"],
        recipients=[],
    )


@app.post("/briefing/generate", response_model=BriefingGenerateResponse)
async def generate_briefing(request: BriefingGenerateRequest) -> BriefingGenerateResponse:
    """기간별 카드뉴스 브리핑 생성.

    backend ``/api/briefings/generate`` 가 호출할 내부 endpoint. 로컬 CLI 와 달리
    mock 입력을 받지 않고 항상 DB ``card_news`` 기간 필터를 기준으로 실행한다.
    """
    from src.agents.briefing_generation_agent import BriefingGenerationAgent

    log.info(
        (
            "Briefing 요청 | type=%s anchor=%s cards=%d integrated_issues=%d "
            "peers=%d sectors=%d save=%s"
        ),
        request.briefing_type,
        request.anchor_date,
        len(request.card_ids or []),
        len(request.integrated_issue_ids or []),
        len(request.peer_ids or []),
        len(request.sectors or []),
        request.save,
    )
    try:
        result = await BriefingGenerationAgent().generate(
            briefing_type=request.briefing_type,
            anchor_date=request.anchor_date,
            card_ids=request.card_ids,
            integrated_issue_ids=request.integrated_issue_ids,
            peer_ids=request.peer_ids,
            sectors=request.sectors,
            requested_by_user_id=request.requested_by_user_id,
            ratios=request.ratios,
            user_context=request.user_context,
            limit=request.limit,
            save=request.save,
            use_mock=False,
            refine_display_copy=request.refine_display_copy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("BriefingGenerationAgent 실행 실패 | error=%s", exc)
        raise HTTPException(status_code=500, detail="briefing generation failed") from exc

    return BriefingGenerateResponse.model_validate(result)


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


async def _run_collection_track(
    task_id: str,
    track: str,
    companies: list[str],
    trigger_type: str = "scheduled",
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> None:
    from src.config.companies import COMPANY_ALIASES
    from src.config.global_companies import GLOBAL_COMPANY_ALIASES, GLOBAL_COMPANY_IDS
    from src.crawler.base import CrawlRunContext
    from src.crawler.batch_processor import (
        TRACK_A_SOURCES,
        TRACK_B_SOURCES,
        TRACK_C_SOURCES,
        TRACK_D_SOURCES,
        BatchProcessor,
    )
    from src.pipeline.analysis_delivery import AnalysisDeliveryResult, run_analysis_delivery
    from src.preprocessing.classification import ClusterClassifier
    from src.preprocessing.preprocessing import PreprocessingResult, PreprocessingService
    from src.preprocessing.relevance import RelevanceEvaluator

    all_aliases = {**COMPANY_ALIASES, **GLOBAL_COMPANY_ALIASES}
    selected = companies or [*COMPANY_ALIASES, *GLOBAL_COMPANY_IDS]
    invalid = sorted({company for company in selected if company not in all_aliases})
    if invalid:
        log.error("수집 파이프라인 실패 | task_id=%s invalid_company=%s", task_id, invalid)
        return

    processor = BatchProcessor()
    started_at = datetime.now(UTC).isoformat()
    crawl_window = _collection_window(track, window_start, window_end)
    results: list[PreprocessingResult] = []
    delivery_results: list[AnalysisDeliveryResult] = []

    async def _preprocess_crawl_record(record: dict[str, str]) -> PreprocessingResult:
        crawl_run_id = record["crawl_run_id"]
        source_name = record["source_name"]
        source_types = _preprocess_source_types(track, source_name)
        preprocessing_service = PreprocessingService(
            relevance_evaluator=RelevanceEvaluator(enable_llm="news" in source_types),
            classifier=ClusterClassifier(enable_llm="news" in source_types),
        )
        return await asyncio.to_thread(
            preprocessing_service.run,
            company=selected,
            source_types=source_types,
            trigger_type=trigger_type,
            collected_since=None,
            crawl_run_id=crawl_run_id,
            limit=SCHEDULED_PREPROCESS_LIMIT,
        )

    async def _run_source_group(track_label: str, source_names: tuple[str, ...]) -> None:
        keywords = {company: all_aliases[company] for company in selected}
        for source_name in source_names:
            before = len(processor.last_crawl_run_records)
            await processor.run_sources(
                (source_name,),
                keywords=keywords,
                persist=True,
                crawl_window=crawl_window,
                run_context=CrawlRunContext(
                    collection_mode="realtime",
                    track=track_label.upper(),
                    source_name=source_name,
                ),
            )
            new_records = processor.last_crawl_run_records[before:]
            if not new_records:
                log.info(
                    "수집 source 후처리 스킵 | task_id=%s track=%s source=%s reason=no_crawl_run",
                    task_id,
                    track_label,
                    source_name,
                )
                continue
            for record in new_records:
                result = await _preprocess_crawl_record(record)
                results.append(result)

    try:
        if track in {"a", "all"}:
            await _run_source_group("a", TRACK_A_SOURCES)
        if track in {"b", "all"}:
            await _run_source_group("b", TRACK_B_SOURCES)
        if track in {"c", "all"}:
            await _run_source_group("c", TRACK_C_SOURCES)
        if track in {"d", "all"}:
            await _run_source_group("d", TRACK_D_SOURCES)

        if not results:
            source_types = _preprocess_source_types(track, None)
            preprocessing_service = PreprocessingService(
                relevance_evaluator=RelevanceEvaluator(enable_llm="news" in source_types),
                classifier=ClusterClassifier(enable_llm="news" in source_types),
            )
            results = [
                await asyncio.to_thread(
                    preprocessing_service.run,
                    company=selected,
                    source_types=source_types,
                    trigger_type=trigger_type,
                    collected_since=started_at,
                    crawl_run_id=None,
                    limit=SCHEDULED_PREPROCESS_LIMIT,
                )
            ]

        news_postprocess = None
        if track in {"a", "all"}:
            news_postprocess = await asyncio.to_thread(_run_recent_news_cluster_postprocess)

        if not delivery_results:
            for result in results:
                delivery_results.append(await asyncio.to_thread(run_analysis_delivery, result))

        log.info(
            (
                "수집 파이프라인 완료 | task_id=%s track=%s raw=%d "
                "analysis_metrics=%d analysis_signals=%d classified=%d "
                "postprocess=%s card_news=%d indexed=%d delivery_errors=%d"
            ),
            task_id,
            track,
            _count_result_items(results, "raw_article_ids"),
            _sum_result_ints(results, "analysis_metric_count"),
            _sum_result_ints(results, "analysis_signal_count"),
            _count_result_items(results, "classified_clusters"),
            news_postprocess,
            _count_result_items(delivery_results, "card_news"),
            _count_result_items(delivery_results, "indexed_card_ids"),
            _count_result_items(delivery_results, "errors"),
        )
    except Exception:
        log.exception("수집 파이프라인 실패 | task_id=%s track=%s", task_id, track)


def _run_recent_news_cluster_postprocess() -> dict:
    from scripts.postprocess_singleton_clusters import run_postprocess
    from src.db.postgres import SessionLocal

    with SessionLocal() as db:
        result = run_postprocess(
            db=db,
            source_type="news",
            lookback_hours=24,
            time_field="published_at",
            max_source_size=4,
            min_target_size=4,
            min_new_cluster_size=2,
            max_time_gap_hours=72,
            min_score=0.45,
            apply=True,
            skip_noise=True,
        )
        db.commit()
        summary = {
            "clusters": result["cluster_count"],
            "sources": result["source_count"],
            "targets": result["target_count"],
            "merge_candidates": len(result["candidates"]),
            "group_merge_candidates": len(result["group_candidates"]),
            "noise_candidates": len(result["noise_ids"]),
            "updated": result["updated"],
            "group_updated": result["group_updated"],
            "noise_updated": result["noise_updated"],
        }
    log.info("뉴스 클러스터 후처리 완료 | %s", summary)
    return summary


def _preprocess_source_types(track: str, source_name: str | None) -> list[str]:
    if source_name:
        normalized_source = source_name.split("[", 1)[0]
        source_types = PREPROCESS_SOURCE_TYPES_BY_SOURCE.get(normalized_source)
        if source_types:
            return source_types

    if track == "all":
        seen: set[str] = set()
        values: list[str] = []
        for source_types in PREPROCESS_SOURCE_TYPES_BY_TRACK.values():
            for source_type in source_types:
                if source_type not in seen:
                    seen.add(source_type)
                    values.append(source_type)
        return values

    return PREPROCESS_SOURCE_TYPES_BY_TRACK.get(track, [])


def _collection_window(
    track: str,
    window_start: datetime | None,
    window_end: datetime | None,
):
    from src.crawler.base import CrawlWindow

    if window_start and window_end:
        return CrawlWindow(
            start=_ensure_kst(window_start),
            end=_ensure_kst(window_end),
        )

    end = datetime.now(KST)
    if track == "a":
        start = end - timedelta(hours=1)
    else:
        start = end - timedelta(days=1)
    return CrawlWindow(start=start, end=end)


def _ensure_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


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


@app.post("/chat", response_model=ChatTurnResponse)
async def chat(request: ChatTurnRequest) -> ChatTurnResponse:
    """Floating assistant chat.

    Page-aware CAG + DB/RAG retrieval 기반으로 답하고, Mixer/Briefing 같은
    multi-step 작업은 실행하지 않고 페이지 handoff 로만 반환한다.
    """
    from src.agents.chat_orchestrator_agent import ChatOrchestratorAgent

    log.info(
        "Assistant chat 요청 | conversation=%s route=%s message_len=%s",
        request.conversation_id,
        request.current_page.route if request.current_page else None,
        len(request.message or ""),
    )
    result = await ChatOrchestratorAgent().answer(request)
    return ChatTurnResponse.model_validate(result)


@app.post("/today-insight/generate", response_model=TodayInsightGenerateResponse)
async def generate_today_insight(
    request: TodayInsightGenerateRequest,
) -> TodayInsightGenerateResponse:
    """Home dashboard Today's Insight 생성.

    integrated_issues + card_news + profile context + prior today_insight_reports
    JSON memory 를 묶어 오늘 달라진 점, 주요 신호, 관찰 포인트, 다음 판단, 근거와
    출처를 UI-ready schema 로 반환한다.
    """
    from src.agents.today_insight_agent import TodayInsightAgent

    log.info(
        "TodayInsight 요청 | anchor=%s window=%s force=%s save=%s",
        request.anchor_date,
        request.window_days,
        request.force_refresh,
        request.save,
    )
    result = await TodayInsightAgent().generate(request)
    return TodayInsightGenerateResponse.model_validate(result)


@app.post("/insight/generate", response_model=InsightGenerateResponse)
async def generate_insight(request: InsightGenerateRequest) -> InsightGenerateResponse:
    """InsightCascade — 4-phase + Synthesis CoT 분석.

    design: ``axis-ai/design/30-analysis/insight-cascade.md``. Walking Skeleton
    phase 2 prototype — cold-start fallback 만 활성 (ContextPack 미주입).

    LLM ~1 호출 (gpt-4o, ~3K tokens). 분석 결과는 ``@with_ledger_writeback`` 으로
    ``analysis_ledger`` 에 자동 INSERT — 다음 분석 호출 시 carry-over.
    """
    from src.agents.insight_cascade_agent import InsightCascadeAgent

    log.info("Insight 요청 | card_ids=%s", request.card_ids)
    result = await InsightCascadeAgent().generate(
        card_ids=request.card_ids,
        context=request.context,
    )
    return InsightGenerateResponse.model_validate(result)


@app.post("/mixer/analyze", response_model=MixerAnalysisResponse)
async def analyze_mixer(request: MixerAnalysisRequest) -> MixerAnalysisResponse:
    """MixerAnalysis — 3-phase per_card / cross_card / synthesis CoT.

    design: ``axis-ai/design/30-analysis/mixer-analysis.md``. Walking Skeleton
    phase 2 prototype — 6축 radar 는 결정적 산식, reasoning 만 LLM 단일 호출.

    cold-start fallback (ContextPack 미주입). ``@with_ledger_writeback`` 으로
    분석 ledger 에 carry-over.
    """
    from src.agents.mixer_analysis_agent import MixerAnalysisAgent

    log.info(
        "Mixer 요청 | mode=%s card_ids=%s integrated_issue_ids=%s",
        request.analysis_mode,
        request.card_ids,
        request.integrated_issue_ids,
    )
    result = await MixerAnalysisAgent().analyze(
        card_ids=request.card_ids,
        integrated_issue_ids=request.integrated_issue_ids,
        ratios=request.ratios,
        user_context=request.user_context,
        analysis_mode=request.analysis_mode,
    )
    return MixerAnalysisResponse.model_validate(result)


@app.post("/mixer/analyze/stream")
async def analyze_mixer_stream(request: MixerAnalysisRequest) -> StreamingResponse:
    """MixerAnalysis SSE — 실행 중 실제 단계(prepare/analyze/synthesize/finalize)를
    실시간 emit 한 뒤 최종 결과를 보낸다.

    이벤트(SSE ``data:`` 한 줄당 JSON 1건):
      - ``{"type":"stage","stage","label","index","total"}`` — 단계 진입 시
      - ``{"type":"result","data":{...MixerAnalysisOutput...}}`` — 분석 완료
      - ``{"type":"error","message":"..."}`` — 실패

    구현 메모: MixerAnalysisAgent 의 LLM 호출은 동기(``.invoke``)라 이벤트 루프를
    막는다. 따라서 분석은 별도 스레드에서 실행하고, progress 콜백은
    ``call_soon_threadsafe`` 로 메인 루프의 asyncio.Queue 에 흘려 SSE 제너레이터가
    실시간으로 drain 한다.
    """
    from src.agents.mixer_analysis_agent import MixerAnalysisAgent

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(stage: str, label: str, index: int, total: int) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "stage", "stage": stage, "label": label, "index": index, "total": total},
        )

    def run_blocking() -> dict:
        # analyze 는 async 지만 내부 호출이 모두 동기라 새 루프에서 안전하게 실행.
        return asyncio.run(
            MixerAnalysisAgent().analyze(
                card_ids=request.card_ids,
                integrated_issue_ids=request.integrated_issue_ids,
                ratios=request.ratios,
                user_context=request.user_context,
                analysis_mode=request.analysis_mode,
                progress=progress,
            )
        )

    async def worker() -> None:
        try:
            result = await asyncio.to_thread(run_blocking)
            payload = MixerAnalysisResponse.model_validate(result).model_dump(mode="json")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": payload})
        except Exception as e:  # noqa: BLE001 — 모든 실패를 SSE error 로 전달
            log.warning("Mixer stream 실패: %s", e)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_gen():
        task = asyncio.create_task(worker())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_MIXER_SSE_HEARTBEAT_SEC)
                except asyncio.TimeoutError:
                    # 무이벤트 구간(메인+repair LLM, 수십 초)에 keepalive 핑.
                    # 주석(": ")은 백엔드 SSE 디코더가 버리므로 data 이벤트로 보낸다.
                    # 그래야 백엔드→ingress→브라우저 전 구간이 살아
                    # nginx/ALB idle timeout(기본 60s)을 피한다.
                    # 프론트는 stage/result/error 만 처리하므로 ping 은 무시된다.
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/global/trends/run", response_model=GlobalTrendsResponse)
async def run_global_trends(request: GlobalTrendsRequest) -> GlobalTrendsResponse:
    """GlobalTrendsAgent — 5-phase 글로벌 IT 트렌드 + peer alignment.

    design: ``axis-ai/design/30-analysis/global-trends.md``.
    contract: ``axis-infra/api/openapi.yaml`` ``/global/trends/run`` (operationId
    ``runGlobalTrends``).

    Phase 1 (Snapshot) + Phase 2 (Trend Detection) 결정적 산식,
    Phase 3 (Peer Alignment) + Phase 4 (Impact Mapping) + Phase 5 (Synthesis) LLM 3 호출.

    ``previous_trend_context`` 는 ITTrendAgent.generate() 가 ``global_industry_trends`` 직전
    batch 를 self-read 해 delta 를 계산한다 (design §16).

    결과는 ``global_industry_trends`` 에 keyword 별 row 로 직접 upsert. (V30 이후
    ``analysis_ledger`` DROP 되어 ``@with_ledger_writeback`` 미사용 — 설계서 §7.)
    """
    from src.agents.it_trend_agent import ITTrendAgent, ITTrendInput

    log.info(
        "GlobalTrends 요청 | window_days=%s peers=%s themes=%s",
        request.window_days,
        request.peer_company_ids,
        request.focus_themes,
    )
    trend_input = ITTrendInput(
        trend_items=[],
        period=None,
        source_groups=[],
        previous_trend_context=None,
        reference_issue_results=[],
        metadata={
            "window_days": request.window_days,
            "company_ids": request.company_ids,
            "focus_themes": request.focus_themes,
            "sk_ax_business_lines": request.sk_ax_business_lines,
            "peer_company_ids": request.peer_company_ids,
            "include_peer_alignment": request.include_peer_alignment,
            "min_mention_count": request.min_mention_count,
            "max_trend_count": request.max_trend_count,
        },
    )
    # ITTrendAgent.generate 는 sync (5-phase 합산 ~70s, LLM 3 calls + DB 호출) —
    # event loop 를 막으면 liveness probe /healthz 도 응답 못해 SIGKILL.
    result = await asyncio.to_thread(ITTrendAgent().generate, trend_input)
    return GlobalTrendsResponse.model_validate(
        {
            "analysis_id": result.get("analysis_id", ""),
            "analysis_period": result.get("analysis_period", {}),
            "snapshots": result.get("snapshots", []),
            "trend_detections": result.get("trend_detections", []),
            "peer_alignment": result.get("peer_alignment", {}),
            "impact_matrix": result.get("impact_matrix", []),
            "forecasts": result.get("forecasts", []),
            "final_one_liner": result.get("final_one_liner", ""),
            "sk_ax_implication": result.get("sk_ax_implication", ""),
            "reasoning_steps": result.get("reasoning_steps", []),
            "confidence": result.get("confidence", 0.0),
            "persisted_row_count": result.get("persisted_row_count", 0),
            "validation": result.get("validation", {}),
            "warning": result.get("warning"),
            "company_ids": request.company_ids or [],
            "provenance": {
                "prompt_version": result.get("prompt_version"),
                "agent": result.get("agent"),
            },
        }
    )


@app.post("/link/verify", response_model=LinkVerificationResponse)
async def verify_link(request: LinkVerificationRequest) -> LinkVerificationResponse:
    """LinkVerification — 카드 source URL 들의 HTTP HEAD 검증 + 옵션 GET hash diff.

    design: ``axis-ai/design/30-analysis/link-verification.md``. Walking Skeleton
    phase 2 prototype — deterministic, LLM 미사용. ``link_verification_logs``
    테이블 저장은 Day 90+ 후속 작업.
    """
    from src.services.link_verification import LinkVerificationService

    log.info("LinkVerify 요청 | card_id=%s", request.card_id)
    result = await LinkVerificationService().verify(card_id=request.card_id)
    return LinkVerificationResponse.model_validate(result)


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
    """DB 대표 클러스터를 요약·분석·카드뉴스 composer 흐름으로 변환한다."""
    from src.agents.analysis_graph_runner import AnalysisGraphRunner
    from src.composers.card_news_composer import CardNewsComposer
    from src.config.company_tiers import SELF_COMPANY_IDS
    from src.db.article_store import get_articles_by_ids, list_card_news_cluster_candidates
    from src.preprocessing.classification import ClusterClassifier

    candidate_limit = min(max(limit * 5, limit), 30)
    candidates = list_card_news_cluster_candidates(
        limit=candidate_limit,
        today_only=today_only,
    )
    if not candidates:
        return []

    classifier = ClusterClassifier()
    analysis_runner = AnalysisGraphRunner()
    card_agent = CardNewsComposer()
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

        company = _first_company(candidate.get("company"))
        if company in SELF_COMPANY_IDS:
            continue

        articles = get_articles_by_ids(article_ids)
        classification = classifier.classify(
            cluster_id=cluster_id,
            representative_id=representative_id,
            cluster_article_ids=article_ids,
            company=company,
        )
        analysis_package = analysis_runner.analyze_cluster(
            cluster_id=cluster_id,
            representative_id=representative_id,
            classification=classification,
            articles=articles,
            cluster_article_ids=article_ids,
        )
        integrated_issue = analysis_package.get("integrated_issue") or analysis_package["summary"]
        if not integrated_issue.get("is_valid_summary"):
            continue
        cards.append(
            card_agent.generate_from_analysis_package(
                analysis_package,
                classification=classification,
            )
        )
        if len(cards) >= limit:
            break

    return cards


def _first_company(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
        return value
    return ""
