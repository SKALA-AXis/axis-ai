import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.insight_schemas import InsightGenerateRequest, InsightGenerateResponse
from src.api.link_verification_schemas import LinkVerificationRequest, LinkVerificationResponse
from src.api.mixer_schemas import MixerAnalysisRequest, MixerAnalysisResponse
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
    state = delivery_graph.invoke(  # type: ignore[attr-defined]
        {
            "cards": [card.model_dump(by_alias=True) for card in req.cards],
            "subject": "",
            "html": "",
            "text": "",
            "errors": [],
        }
    )
    return BriefingContent(
        subject=state["subject"],
        html=state["html"],
        text=state["text"],
        recipients=[],
    )


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
    from src.crawler.batch_processor import BatchProcessor
    from src.preprocessing.classification import ClusterClassifier
    from src.preprocessing.preprocessing import PreprocessingService
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

    try:
        if track in {"a", "all"}:
            await processor.run_track_a(
                {company: all_aliases[company] for company in selected},
                crawl_window=crawl_window,
            )
        if track in {"b", "all"}:
            await processor.run_track_b(
                {company: all_aliases[company] for company in selected},
                crawl_window=crawl_window,
            )
        if track in {"c", "all"}:
            await processor.run_track_c(
                {company: all_aliases[company] for company in selected},
                crawl_window=crawl_window,
            )
        if track in {"d", "all"}:
            await processor.run_track_d(
                {company: all_aliases[company] for company in selected},
                crawl_window=crawl_window,
            )

        preprocessing_service = PreprocessingService(
            relevance_evaluator=RelevanceEvaluator(enable_llm=True),
            # Classification is rule-first; LLM is only a last-resort fallback
            # when OpenAI calls are explicitly enabled by policy.
            classifier=ClusterClassifier(enable_llm=True),
        )
        crawl_run_ids = processor.last_crawl_run_ids
        if crawl_run_ids:
            results = [
                preprocessing_service.run(
                    company=selected,
                    trigger_type=trigger_type,
                    collected_since=None,
                    crawl_run_id=crawl_run_id,
                )
                for crawl_run_id in crawl_run_ids
            ]
        else:
            results = [
                preprocessing_service.run(
                    company=selected,
                    trigger_type=trigger_type,
                    collected_since=started_at,
                    crawl_run_id=None,
                )
            ]
        log.info(
            (
                "수집 파이프라인 완료 | task_id=%s track=%s raw=%d "
                "analysis_metrics=%d analysis_signals=%d classified=%d"
            ),
            task_id,
            track,
            sum(len(result.get("raw_article_ids", [])) for result in results),
            sum(result.get("analysis_metric_count", 0) for result in results),
            sum(result.get("analysis_signal_count", 0) for result in results),
            sum(len(result.get("classified_clusters", [])) for result in results),
        )
    except Exception:
        log.exception("수집 파이프라인 실패 | task_id=%s track=%s", task_id, track)


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

    log.info("Mixer 요청 | card_ids=%s", request.card_ids)
    result = await MixerAnalysisAgent().analyze(
        card_ids=request.card_ids,
        ratios=request.ratios,
        user_context=request.user_context,
    )
    return MixerAnalysisResponse.model_validate(result)


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
    """DB 대표 클러스터를 요약·분석·카드뉴스 에이전트 흐름으로 변환한다."""
    from src.agents.analysis_supervisor_agent import AnalysisSupervisorAgent
    from src.agents.card_news_agent import CardNewsAgent
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
    analysis_supervisor = AnalysisSupervisorAgent()
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
        analysis_package = analysis_supervisor.analyze_cluster(
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
