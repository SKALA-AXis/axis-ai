import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.api.briefing_schemas import BriefingGenerateRequest, BriefingGenerateResponse
from src.api.chat_schemas import ChatPdfRequest, ChatTurnRequest, ChatTurnResponse
from src.api.classify_schemas import ClassifyRequest, ClassifyResponse
from src.api.global_trends_schemas import GlobalTrendsRequest, GlobalTrendsResponse
from src.api.insight_schemas import InsightGenerateRequest, InsightGenerateResponse
from src.api.link_verification_schemas import LinkVerificationRequest, LinkVerificationResponse
from src.api.mixer_schemas import MixerAnalysisRequest, MixerAnalysisResponse
from src.api.today_insight_schemas import (
    TodayInsightGenerateRequest,
    TodayInsightGenerateResponse,
)
from src.llm import LLMSpec, build_chat_llm
from src.schemas import (
    BriefingContent,
    BriefingRequest,
    GenSearchRequest,
    GenSearchResult,
    HealthResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

_LOG_LEVEL_NAME = os.getenv("AXIS_AI_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
_LOG_LEVEL = logging.getLevelNamesMapping().get(_LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger().setLevel(_LOG_LEVEL)
log = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")
SCHEDULED_PREPROCESS_LIMIT = 5000
# Mixer SSE keepalive 주기(초) — nginx/ALB idle timeout(기본 60s)보다 충분히 짧게.
_MIXER_SSE_HEARTBEAT_SEC = 10
_CHAT_PDF_MAX_BYTES = 15 * 1024 * 1024
_CHAT_PDF_MAX_TEXT_CHARS = 80_000
_AGENT_CALL_FAILED_MESSAGE = "호출에 실패했다"
_USER_STRATEGY_FILE_MAX_BYTES = 5 * 1024 * 1024


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


_USER_STRATEGY_OCR_MAX_PAGES = _safe_int_env("USER_STRATEGY_OCR_MAX_PAGES", 8)


class UserSkaxOverlayRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(..., min_length=1)
    title: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserStrategyFileOcrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_name: str | None = None
    content_type: str | None = None
    file_base64: str = Field(..., min_length=1)


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


def _raise_if_agent_failure(agent: str, result: Mapping[str, object] | None) -> None:
    reason = _agent_failure_reason(result)
    if not reason:
        return
    raise HTTPException(
        status_code=502,
        detail={
            "code": _agent_failure_code(agent, result),
            "message": _AGENT_CALL_FAILED_MESSAGE,
            "detail": reason,
        },
    )


def _is_today_insight_status_placeholder(result: Mapping[str, object] | None) -> bool:
    """today-insight 결과가 '신호 없음' status placeholder 인지 — 장애가 아니라 정상 빈 결과.

    당일 통합 이슈·신규 카드가 모두 없을 때 agent 가 만드는 placeholder("오늘은 주목할
    동향 없음")는 502 가 아니라 200 으로 내려야 대시보드가 "오늘 중요한 뉴스 없음" 을
    표시한다. LLM 실패(generated_fallback)·status=failed/error 같은 진짜 실패는 제외해
    502 가시성을 유지한다.
    """
    if not result:
        return False
    if str(result.get("status") or "").strip().lower() in {"failed", "error"}:
        return False
    if str(result.get("error") or "").strip():
        return False
    provenance = result.get("provenance")
    if isinstance(provenance, Mapping):
        if str(provenance.get("error") or "").strip():
            return False
        if provenance.get("is_status_placeholder") is True:
            return True
        kind = str(provenance.get("result_kind") or "").strip().lower()
        if kind in {"no_current_signals", "scheduled_pending"}:
            return True
    top_kind = str(result.get("result_kind") or "").strip().lower()
    return top_kind in {"no_current_signals", "scheduled_pending"}


def _agent_failure_event(agent: str, result: Mapping[str, object] | None) -> dict[str, str]:
    return {
        "type": "error",
        "message": _AGENT_CALL_FAILED_MESSAGE,
        "error_code": _agent_failure_code(agent, result),
        "detail": _agent_failure_reason(result) or "agent response failed",
    }


def _agent_failure_reason(result: Mapping[str, object] | None) -> str:
    if not result:
        return "empty_response"
    status = str(result.get("status") or "").strip().lower()
    if status in {"failed", "error"}:
        return status
    top_error = str(result.get("error") or "").strip()
    if top_error:
        return top_error

    provenance = result.get("provenance")
    provenance_error = ""
    provenance_mode = ""
    provenance_kind = ""
    if isinstance(provenance, Mapping):
        provenance_error = str(provenance.get("error") or "").strip()
        provenance_mode = str(provenance.get("mode") or "").strip().lower()
        provenance_kind = str(provenance.get("result_kind") or "").strip().lower()
    if provenance_error:
        return provenance_error

    result_kind = str(result.get("result_kind") or provenance_kind).strip().lower()
    if "unavailable" in result_kind or "empty_axis_ai_response" in result_kind:
        return result_kind
    if "fallback" in result_kind or "fallback" in provenance_mode:
        return result_kind or provenance_mode

    warning = str(result.get("warning") or "").strip()
    warning_lower = warning.lower()
    if (
        "llm generation failed" in warning_lower
        or "llm 호출 실패" in warning_lower
        or "generation failed" in warning_lower
        or "source data unavailable" in warning_lower
    ):
        return warning
    return ""


def _agent_failure_code(agent: str, result: Mapping[str, object] | None) -> str:
    prefix = _normalize_error_code(agent)
    if result:
        top_code = str(result.get("error_code") or "").strip()
        if top_code:
            return _normalize_error_code(top_code)
        provenance = result.get("provenance")
        if isinstance(provenance, Mapping):
            provenance_code = str(provenance.get("error_code") or "").strip()
            if provenance_code:
                return _normalize_error_code(provenance_code)
            provenance_error = str(provenance.get("error") or "").strip()
            if provenance_error:
                return f"{prefix}_{_normalize_error_code(provenance_error)}"
            provenance_kind = str(provenance.get("result_kind") or "").strip()
            if provenance_kind:
                return f"{prefix}_{_normalize_error_code(provenance_kind)}"
        result_kind = str(result.get("result_kind") or "").strip()
        if result_kind:
            return f"{prefix}_{_normalize_error_code(result_kind)}"
    return f"{prefix}_AI_RESPONSE_FAILED"


def _normalize_error_code(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_").upper()
    return normalized or "AI_RESPONSE_FAILED"


def _build_user_skax_overlay_llm():
    model = os.getenv(
        "USER_SKAX_OVERLAY_MODEL",
        os.getenv("FRONTEND_READY_MODEL", "gpt-5.5"),
    )
    return build_chat_llm(
        LLMSpec(
            model=model,
            temperature=1,
            max_tokens=5000,
            max_tokens_reasoning=8000,
            reasoning_effort="low",
            timeout=120,
            max_retries=1,
            json_object=True,
        )
    )


def _user_strategy_ocr_model_name() -> str:
    return os.getenv(
        "USER_STRATEGY_OCR_MODEL",
        os.getenv("FRONTEND_READY_MODEL", "gpt-5.5"),
    )


def _build_user_strategy_ocr_llm():
    return build_chat_llm(
        LLMSpec(
            model=_user_strategy_ocr_model_name(),
            temperature=0,
            max_tokens=6000,
            max_tokens_reasoning=10000,
            reasoning_effort="low",
            timeout=180,
            max_retries=1,
            json_object=False,
        )
    )


def _decode_user_strategy_file(file_base64: str) -> bytes:
    try:
        data = base64.b64decode(file_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="file_base64 is invalid") from exc
    if not data:
        raise HTTPException(status_code=400, detail="file is empty")
    if len(data) > _USER_STRATEGY_FILE_MAX_BYTES:
        raise HTTPException(status_code=400, detail="file must be 5MB or smaller")
    return data


def _normalize_file_content_type(value: str | None, file_name: str | None) -> str:
    content_type = str(value or "").strip().lower()
    lower_name = str(file_name or "").strip().lower()
    if content_type:
        return content_type
    if lower_name.endswith(".pdf"):
        return "application/pdf"
    if lower_name.endswith(".png"):
        return "image/png"
    if lower_name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower_name.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def _image_data_url(content_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _render_pdf_pages_for_ocr(file_bytes: bytes) -> tuple[list[str], int]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyMuPDF is required for scanned PDF OCR") from exc

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_count = int(getattr(doc, "page_count", 0) or 0)
        urls: list[str] = []
        for page_index in range(min(page_count, max(1, _USER_STRATEGY_OCR_MAX_PAGES))):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            urls.append(_image_data_url("image/png", pix.tobytes("png")))
        return urls, page_count
    finally:
        doc.close()


def _ocr_user_strategy_images(
    *,
    image_urls: list[str],
    file_name: str,
    page_count: int,
) -> str:
    if not image_urls:
        return ""
    llm = _build_user_strategy_ocr_llm()
    pages_note = (
        f"총 {page_count}페이지 중 앞 {len(image_urls)}페이지를 처리합니다."
        if page_count > len(image_urls)
        else f"총 {page_count or len(image_urls)}페이지를 처리합니다."
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "아래 파일 이미지에서 보이는 텍스트를 원문 순서대로 추출하세요.\n"
                "- 요약, 해석, 전략 문장 작성 금지\n"
                "- 표는 읽기 쉬운 줄 단위 텍스트로 변환\n"
                "- 보이지 않는 내용은 만들지 않기\n"
                f"- file_name: {file_name}\n"
                f"- page_info: {pages_note}\n"
                "출력은 추출 텍스트만 작성하세요."
            ),
        }
    ]
    user_content.extend(
        {"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls
    )
    result = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "당신은 OCR 텍스트 추출기입니다. 이미지에 보이는 글자만 "
                    "가능한 원문 순서대로 전사하고, 보이지 않는 정보는 만들지 마세요."
                ),
            },
            {"role": "user", "content": user_content},
        ]
    )
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


PREPROCESS_SOURCE_TYPES_BY_SOURCE: dict[str, list[str]] = {
    "naver_news": ["news"],
    "naver_industry_news": ["news"],
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
    # langchain 경로 선로딩 — 에이전트들은 ChatOpenAI 를 지연 임포트하는데(transformers
    # 체인 회피), 상주 서버에서는 첫 LLM 요청이 import 비용까지 떠안아 이벤트 루프를
    # 막고 readiness 플랩을 유발했음 (2026-06-11 배포 순단 실측). startup 에서 1회 선로딩.
    try:
        import langchain_openai  # noqa: F401
    except Exception as e:
        log.warning("startup langchain preload skipped: %s", e)
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


@app.post("/classify", response_model=ClassifyResponse)
async def classify_article(request: ClassifyRequest) -> ClassifyResponse:
    """단일 기사 텍스트 분류 — 운영 수집 파이프라인과 동일 판정(event_type·중요도).

    backend 데모(``/api/demo/publish``)가 호출한다. rule 우선 → GPT-4o fallback 이라
    동기 LLM 호출이 event-loop 를 막지 않도록 ``to_thread`` 로 격리한다.
    """
    from src.preprocessing.classification import classify_article_text

    result = await asyncio.to_thread(
        classify_article_text,
        request.title,
        request.content,
        request.company,
        source_type=request.source_type,
    )
    return ClassifyResponse(**result)


@app.post("/profile/user-skax-overlay")
async def summarize_user_skax_overlay(request: UserSkaxOverlayRequest) -> dict[str, Any]:
    """사용자 입력 전략 자료를 SK AX profile overlay JSON으로 구조화한다."""
    from src.services.profile_snapshot_summarizer import ProfileSnapshotSummarizer

    raw_text = request.raw_text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is required")
    try:
        overlay = await asyncio.to_thread(
            ProfileSnapshotSummarizer(
                llm=_build_user_skax_overlay_llm()
            ).summarize_user_skax_overlay,
            raw_text,
            user_id=request.user_id,
            title=request.title,
            metadata=request.metadata,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("user SK AX overlay 구조화 실패 | error=%s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "USER_SKAX_OVERLAY_GENERATION_FAILED",
                "message": _AGENT_CALL_FAILED_MESSAGE,
                "detail": str(exc)[:500],
            },
        ) from exc
    return {"overlay": overlay}


@app.post("/profile/user-strategy-file-ocr")
async def ocr_user_strategy_file(request: UserStrategyFileOcrRequest) -> dict[str, Any]:
    """스캔 PDF/이미지 전략 자료에서 보이는 텍스트만 추출한다."""
    file_name = str(request.file_name or "uploaded-file").strip() or "uploaded-file"
    content_type = _normalize_file_content_type(request.content_type, file_name)
    file_bytes = _decode_user_strategy_file(request.file_base64)

    try:
        if content_type == "application/pdf" or file_name.lower().endswith(".pdf"):
            image_urls, page_count = await asyncio.to_thread(
                _render_pdf_pages_for_ocr,
                file_bytes,
            )
        elif content_type.startswith("image/"):
            image_urls = [_image_data_url(content_type, file_bytes)]
            page_count = 1
        else:
            raise HTTPException(
                status_code=400,
                detail="OCR supports scanned PDF and image files only",
            )
        extracted_text = await asyncio.to_thread(
            _ocr_user_strategy_images,
            image_urls=image_urls,
            file_name=file_name,
            page_count=page_count,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("user strategy file OCR 실패 | file=%s error=%s", file_name, exc)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "USER_STRATEGY_FILE_OCR_FAILED",
                "message": _AGENT_CALL_FAILED_MESSAGE,
                "detail": str(exc)[:500],
            },
        ) from exc

    if not extracted_text.strip():
        raise HTTPException(
            status_code=502,
            detail={
                "code": "USER_STRATEGY_FILE_OCR_EMPTY",
                "message": _AGENT_CALL_FAILED_MESSAGE,
                "detail": "OCR returned empty text",
            },
        )

    return {
        "extractedText": extracted_text.strip(),
        "extracted_text": extracted_text.strip(),
        "extractionMethod": "openai_vision_ocr",
        "extraction_method": "openai_vision_ocr",
        "model": _user_strategy_ocr_model_name(),
        "pageCount": page_count,
        "page_count": page_count,
        "processedPageCount": len(image_urls),
        "processed_page_count": len(image_urls),
    }


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
            reuse_saved=request.reuse_saved,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("BriefingGenerationAgent 실행 실패 | error=%s", exc)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "BRIEFING_GENERATION_FAILED",
                "message": _AGENT_CALL_FAILED_MESSAGE,
                "detail": str(exc),
            },
        ) from exc

    _raise_if_agent_failure("BRIEFING", result)
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
            _count_result_items(delivery_results, "indexed_vector_ids"),
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
            max_source_size=0,
            min_target_size=2,
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
    hits, _timings = await asyncio.to_thread(_run_search_pipeline, request)
    return SearchResponse(hits=[SearchHit.model_validate(hit) for hit in hits], total=len(hits))


@app.post("/gen-search", response_model=GenSearchResult)
async def gen_search(request: GenSearchRequest):
    """Generative Search — RAG + GPT-4o + SC 검증"""
    log.info("Generative Search | query=%s", request.query)
    search_request = SearchRequest(
        query=request.query,
        company=request.company,
        event_type=None,
        top_k=request.top_k,
    )
    hits, _timings = await asyncio.to_thread(_run_search_pipeline, search_request)
    if not hits:
        return GenSearchResult(
            answer="검색 인덱스에서 관련 근거를 찾지 못했습니다. 검색어를 더 구체화해 주세요.",
            sources=[],
            sc_passed=False,
            sc_score=0.0,
        )

    llm_answer = await asyncio.to_thread(_try_gen_search_llm_answer, request.query, hits)
    if llm_answer:
        return GenSearchResult(
            answer=llm_answer,
            sources=hits,
            sc_passed=True,
            sc_score=0.72,
        )

    return GenSearchResult(
        answer=_deterministic_gen_search_answer(request.query, hits),
        sources=hits,
        sc_passed=False,
        sc_score=0.42,
    )


def _run_search_pipeline(request: SearchRequest) -> tuple[list[dict[str, object]], dict[str, int]]:
    from src.rag.hybrid_search import hybrid_search
    from src.rag.reranker import rerank

    top_k = max(1, min(int(request.top_k or 10), 50))
    timings: dict[str, int] = {}
    started = time.perf_counter()
    try:
        candidates = hybrid_search(
            query=request.query,
            top_k=max(top_k * 3, top_k),
            company=request.company,
            event_type=request.event_type,
            raise_on_failure=True,
        )
        timings["search_ms"] = int((time.perf_counter() - started) * 1000)
        rerank_started = time.perf_counter()
        ranked = rerank(request.query, candidates, top_k=top_k)
        timings["rerank_ms"] = int((time.perf_counter() - rerank_started) * 1000)
    except Exception as exc:  # noqa: BLE001 - 내부 API는 장애를 빈 결과로 숨기지 않는다.
        log.exception("검색 실행 실패 | query=%s error=%s", request.query, exc)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "SEARCH_RAG_UNAVAILABLE",
                "message": _AGENT_CALL_FAILED_MESSAGE,
                "detail": str(exc),
            },
        ) from exc
    return [_normalize_search_hit(hit) for hit in ranked], timings


def _normalize_search_hit(hit: Mapping[str, object]) -> dict[str, object]:
    rdb_id = _safe_int(
        hit.get("rdb_id")
        or hit.get("raw_article_id")
        or hit.get("article_id")
        or hit.get("source_id")
    )
    company = _first_text(hit.get("company"), hit.get("peer_id"), hit.get("peer"), "unknown")
    title = _first_text(hit.get("title"), hit.get("card_title"), f"검색 결과 {rdb_id}")
    summary = _first_text(hit.get("summary"), hit.get("snippet"), hit.get("text"), "")
    event_type = _first_text(hit.get("event_type"), hit.get("type"), "unknown")
    published_at = hit.get("pub_date") or hit.get("published_at") or hit.get("updated_at")
    score = _safe_float(hit.get("rerank_score"), hit.get("score"), 0.0)
    return {
        "rdb_id": rdb_id,
        "company": company,
        "title": title,
        "summary": summary,
        "importance": _first_text(hit.get("importance"), hit.get("exposure_band"), "unknown"),
        "event_type": event_type,
        "pub_date": _search_date_string(published_at),
        "rerank_score": score,
        "source_url": _optional_text(hit.get("source_url") or hit.get("url") or hit.get("link")),
    }


def _try_gen_search_llm_answer(query: str, hits: list[dict[str, object]]) -> str:
    from src.services.llm_env import llm_credentials_ready

    if os.getenv("AXIS_GEN_SEARCH_ENABLE_LLM", "1").lower() in {"0", "false", "no"}:
        return ""
    if not llm_credentials_ready():
        return ""
    try:
        payload = {
            "query": query,
            "sources": [
                {
                    "title": hit.get("title"),
                    "summary": hit.get("summary"),
                    "company": hit.get("company"),
                    "event_type": hit.get("event_type"),
                    "pub_date": hit.get("pub_date"),
                    "score": hit.get("rerank_score"),
                }
                for hit in hits[:6]
            ],
        }
        prompt = f"""\
AXIS Generative Search 답변을 작성합니다.

규칙:
- 아래 JSON의 sources 안에 있는 사실만 사용합니다.
- 출처에 없는 수치, 고객명, 계약명, 날짜를 만들지 않습니다.
- 답변은 한국어 4~7문장으로 작성합니다.
- 마지막 문장에는 추가로 확인해야 할 검색어 1개를 제안합니다.
- JSON object 하나만 반환합니다.

입력 JSON:
{json.dumps(payload, ensure_ascii=False)}

출력 JSON:
{{"answer":"근거 기반 답변"}}
"""
        model = os.getenv("GEN_SEARCH_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
        # env 로 모델 지정 가능 → gpt-5 라도 reasoning_effort 미전달(기존 동작) 위해 None.
        llm = build_chat_llm(
            LLMSpec(
                model=model,
                temperature=0.1,
                max_tokens=800,
                json_object=True,
                reasoning_effort=None,
            )
        )
        from src.observability.langfuse_client import tracing_config

        result = llm.invoke(
            prompt,
            config=tracing_config(agent="GenSearch", phase="answer_compose"),
        )
        parsed = json.loads(str(getattr(result, "content", result) or "{}"))
        answer = str(parsed.get("answer") or "").strip()
        return answer
    except Exception as exc:  # noqa: BLE001 - 검색 결과 요약 fallback 을 사용한다.
        log.warning("GenSearch LLM compose failed; deterministic fallback used | error=%s", exc)
        return ""


def _deterministic_gen_search_answer(query: str, hits: list[dict[str, object]]) -> str:
    lead = hits[0]
    bullets = []
    for index, hit in enumerate(hits[:3], start=1):
        title = _first_text(hit.get("title"), f"근거 {index}")
        company = _first_text(hit.get("company"), "unknown")
        summary = _first_text(hit.get("summary"), "")
        bullets.append(f"{index}. {company}: {title}" + (f" — {summary}" if summary else ""))
    return (
        f"'{query}'에 대해 Qdrant 하이브리드 검색과 rerank 결과를 기준으로 요약했습니다. "
        f"가장 관련도가 높은 근거는 {lead.get('company')}의 '{lead.get('title')}'입니다. "
        "현재 응답은 LLM 생성 단계가 비활성화되었거나 실패해 "
        "deterministic fallback으로 작성되었습니다.\n" + "\n".join(bullets)
    )


def _first_text(*values: object) -> str:
    for value in values:
        text = _optional_text(value)
        if text:
            return text
    return ""


def _optional_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _safe_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _safe_float(*values: object) -> float:
    for value in values:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            continue
    return 0.0


def _search_date_string(value: object) -> str:
    if isinstance(value, (int, float)):
        if value <= 0:
            return ""
        return datetime.fromtimestamp(value, tz=UTC).isoformat()
    text = _optional_text(value)
    return text


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


@app.post("/chat/pdf", response_model=ChatTurnResponse)
async def chat_pdf(request: ChatPdfRequest) -> ChatTurnResponse:
    """Analyze a user-uploaded PDF inside the floating assistant flow."""

    try:
        pdf_bytes = base64.b64decode(request.pdf_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ASSISTANT_PDF_INVALID_BASE64",
                "message": "pdf_base64 is not valid base64",
            },
        ) from exc

    if len(pdf_bytes) > _CHAT_PDF_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "ASSISTANT_PDF_FILE_TOO_LARGE",
                "message": "PDF file is too large",
            },
        )
    if not _looks_like_pdf(request.file_name, request.content_type, pdf_bytes):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ASSISTANT_PDF_UNSUPPORTED_TYPE",
                "message": "Only PDF attachments are supported",
            },
        )

    from src.crawler.parsers.pdf_payload import extract_pdf_payload

    pdf_payload = extract_pdf_payload(
        pdf_bytes,
        max_text_chars=_CHAT_PDF_MAX_TEXT_CHARS,
    )
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    log.info(
        "Assistant PDF chat 요청 | conversation=%s file=%s bytes=%s text_chars=%s",
        request.request.conversation_id,
        request.file_name,
        len(pdf_bytes),
        len(str(pdf_payload.get("text") or "")),
    )
    result = _build_pdf_chat_response(
        request.request,
        file_name=request.file_name,
        content_type=request.content_type or "application/pdf",
        file_hash=file_hash,
        pdf_payload=pdf_payload,
    )
    return ChatTurnResponse.model_validate(result)


def _looks_like_pdf(file_name: str, content_type: str | None, pdf_bytes: bytes) -> bool:
    name_ok = (file_name or "").lower().endswith(".pdf")
    type_ok = (content_type or "").lower() in {"application/pdf", "application/x-pdf"}
    bytes_ok = pdf_bytes.startswith(b"%PDF")
    return bytes_ok or (name_ok and type_ok)


def _build_pdf_chat_response(
    request: ChatTurnRequest,
    *,
    file_name: str,
    content_type: str,
    file_hash: str,
    pdf_payload: Mapping[str, object],
) -> dict[str, object]:
    conversation_id = request.conversation_id or request.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    text = str(pdf_payload.get("text") or "").strip()
    page_count = _safe_pdf_int(pdf_payload.get("page_count"))
    parsed_page_count = _safe_pdf_int(pdf_payload.get("parsed_page_count"))
    source_id = f"pdf:{file_hash[:16]}"

    if not text:
        error_code = "ASSISTANT_PDF_TEXT_EXTRACTION_FAILED"
        return {
            "conversation_id": conversation_id,
            "session_id": conversation_id,
            "message_id": message_id,
            "reply": (
                "기능에 문제가 생겼습니다.\n"
                f"에러코드: {error_code}\n"
                "PDF에서 분석 가능한 텍스트를 추출하지 못했습니다."
            ),
            "intent": "pdf_attachment_analysis",
            "scope": "uploaded_pdf",
            "answer_blocks": [
                {
                    "type": "warning",
                    "title": "PDF 분석 실패",
                    "items": [
                        f"파일명: {file_name}",
                        f"파싱 전략: {pdf_payload.get('pdf_parse_strategy') or 'unknown'}",
                    ],
                }
            ],
            "sources": [],
            "follow_up_suggestions": ["다른 PDF로 다시 분석해줘"],
            "confidence": 0.15,
            "blocked": True,
            "blocked_reason": error_code,
            "error_code": error_code,
            "handoff": None,
            "provenance": {
                "retrieval_mode": "uploaded_pdf_text_extraction",
                "error_code": error_code,
                "attachment": {
                    "file_name": file_name,
                    "content_type": content_type,
                    "sha256": file_hash,
                    "page_count": page_count,
                    "parsed_page_count": parsed_page_count,
                },
            },
        }

    title = _pdf_title(file_name, text)
    bullets = _pdf_key_points(text, limit=4)
    evidence = _pdf_evidence_lines(text, limit=4)
    llm_payload = _try_pdf_llm_payload(
        request=request,
        file_name=file_name,
        text=text,
        bullets=bullets,
        evidence=evidence,
    )
    if llm_payload:
        bullets = _string_list_from_payload(
            llm_payload.get("key_points"), fallback=bullets, limit=4
        )
        evidence = _string_list_from_payload(
            llm_payload.get("evidence"), fallback=evidence, limit=4
        )
    question = (request.message or "첨부 PDF를 분석해줘").strip()
    reply = str(llm_payload.get("reply") or "").strip() if llm_payload else ""
    if not reply:
        reply = (
            f"{file_name}에서 {parsed_page_count or page_count}개 페이지의 텍스트를 확인했습니다. "
            f"요청 '{question}' 기준으로 핵심은 {bullets[0] if bullets else title} 입니다."
        )
    raw_report_draft = llm_payload.get("report_draft") if llm_payload else None
    report_draft_payload: Mapping[str, object] = (
        raw_report_draft if isinstance(raw_report_draft, dict) else {}
    )
    report_draft = {
        "title": str(report_draft_payload.get("title") or "").strip() or f"{title} 분석 보고서",
        "sections": [
            {
                "title": "목차 및 구성",
                "body": "\n".join(
                    [
                        "1. Executive Summary",
                        "2. 문서 주요 내용",
                        "3. 문서 근거",
                        "4. 해석 한계",
                        "5. SK AX 관점 검토 포인트",
                    ]
                ),
            },
            {
                "title": "Executive Summary",
                "body": _section_body(report_draft_payload, "핵심 요약")
                or _pdf_executive_summary(
                    title=title, bullets=bullets, page_count=parsed_page_count or page_count
                ),
            },
            {
                "title": "문서 주요 내용",
                "body": _section_body(report_draft_payload, "문서 주요 내용")
                or "\n".join(f"{index}. {item}" for index, item in enumerate(bullets[:6], start=1)),
            },
            {
                "title": "문서 근거",
                "body": _section_body(report_draft_payload, "문서 근거")
                or "\n".join(f"- {item}" for item in evidence[:8]),
            },
            {
                "title": "해석 한계",
                "body": (
                    "이 초안은 업로드된 PDF에서 추출 가능한 텍스트만 바탕으로 작성되었습니다. "
                    "표, 이미지, 각주, 스캔본 OCR 품질에 따라 일부 문맥이 누락될 수 있으므로 "
                    "최종 보고 전 원문 페이지와 수치·고유명사를 대조해야 합니다."
                ),
            },
            {
                "title": "SK AX 관점 검토 포인트",
                "body": _section_body(report_draft_payload, "SK AX 관점 검토 포인트")
                or _pdf_skax_review_point(bullets, evidence),
            },
        ],
    }
    return {
        "conversation_id": conversation_id,
        "session_id": conversation_id,
        "message_id": message_id,
        "reply": reply,
        "intent": "pdf_attachment_analysis",
        "scope": "uploaded_pdf",
        "answer_blocks": [
            {"type": "summary", "title": "PDF 핵심 요약", "items": bullets},
            {"type": "evidence", "title": "문서 근거", "items": evidence},
        ],
        "report_draft": report_draft,
        "sources": [
            {
                "type": "pdf_attachment",
                "id": source_id,
                "title": file_name,
                "snippet": _compact_text(text, limit=420),
                "score": 1.0,
                "source_name": "uploaded_pdf",
            }
        ],
        "follow_up_suggestions": [
            "이 PDF를 임원 보고서 형식으로 다시 정리해줘",
            "문서에서 SK AX가 확인해야 할 리스크만 뽑아줘",
        ],
        "confidence": _safe_pdf_confidence(
            llm_payload.get("confidence") if llm_payload else None, text
        ),
        "blocked": False,
        "blocked_reason": None,
        "handoff": None,
        "provenance": {
            "retrieval_mode": "uploaded_pdf_text_extraction",
            "attachment": {
                "file_name": file_name,
                "content_type": content_type,
                "sha256": file_hash,
                "page_count": page_count,
                "parsed_page_count": parsed_page_count,
                "text_chars": len(text),
                "parse_strategy": pdf_payload.get("pdf_parse_strategy"),
                "llm_used": bool(llm_payload),
            },
        },
    }


def _try_pdf_llm_payload(
    *,
    request: ChatTurnRequest,
    file_name: str,
    text: str,
    bullets: list[str],
    evidence: list[str],
) -> dict[str, object]:
    if os.getenv("AXIS_CHAT_PDF_ENABLE_LLM", "1").lower() in {"0", "false", "no"}:
        return {}
    try:
        from src.agents.chat_orchestrator_agent import _get_llm, _parse_json_object
        from src.observability.langfuse_client import tracing_config, with_session

        prompt = _pdf_llm_prompt(
            request=request,
            file_name=file_name,
            text=text,
            bullets=bullets,
            evidence=evidence,
        )
        session_id = request.conversation_id or request.session_id or str(uuid.uuid4())
        with with_session(session_id):
            result = _get_llm().invoke(
                prompt,
                config=tracing_config(
                    agent="ChatOrchestratorAgent",
                    phase="pdf_attachment_analysis",
                    prompt_version="chat-pdf-v1",
                    session_id=session_id,
                ),
            )
        parsed = _parse_json_object(str(getattr(result, "content", result) or ""))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:  # noqa: BLE001 - PDF chat must still answer without LLM.
        log.debug("assistant PDF LLM compose skipped | file=%s error=%s", file_name, exc)
        return {}


def _pdf_llm_prompt(
    *,
    request: ChatTurnRequest,
    file_name: str,
    text: str,
    bullets: list[str],
    evidence: list[str],
) -> str:
    payload = {
        "question": request.message,
        "file_name": file_name,
        "extracted_key_points": bullets,
        "extracted_evidence": evidence,
        "pdf_text": text[:18_000],
    }
    return f"""\
당신은 SK AX AXIS 챗봇의 PDF 분석 모듈입니다.

규칙:
- 아래 입력 JSON의 pdf_text와 extracted_evidence에 있는 사실만 사용합니다.
- 문서에 없는 고객명, 금액, 일정, 계약명은 만들지 않습니다.
- 사용자의 질문에 먼저 답하고, 임원이 바로 출력할 수 있는 보고서 초안을 함께 작성합니다.
- report_draft는 내부적으로 "목차 및 구성 설계 → 초안 작성 → 문서 근거로 내용 채우기" 순서로
  작성하되 내부 단계명은 출력하지 않고 완성본만 반환합니다.
- report_draft는 5~7개 섹션으로 구성하고 각 body는 3~6문장 또는 3~5개 bullet을 포함합니다.
- SK AX 관점은 "무엇을 확인/판단/조치해야 하는지"로 씁니다.
- SK AX가 이미 알고 있을 내부 행동 묘사는 쓰지 않습니다.
- 출력은 JSON object 하나만 반환합니다.

입력 JSON:
{json.dumps(payload, ensure_ascii=False)}

출력 JSON:
{{
  "reply": "PDF 기반 답변 3~6문장",
  "key_points": ["핵심 포인트 1", "핵심 포인트 2"],
  "evidence": ["문서 안 근거 문장 또는 수치"],
  "report_draft": {{
    "title": "보고서 제목",
    "sections": [
      {{"title": "Executive Summary", "body": "출력 가능한 본문"}},
      {{"title": "문서 주요 내용", "body": "출력 가능한 본문"}},
      {{"title": "문서 근거", "body": "출력 가능한 본문"}},
      {{"title": "해석 한계", "body": "출력 가능한 본문"}},
      {{"title": "SK AX 관점 검토 포인트", "body": "출력 가능한 본문"}}
    ]
  }},
  "confidence": 0.0
}}
"""


def _pdf_title(file_name: str, text: str) -> str:
    for line in _pdf_lines(text):
        cleaned = re.sub(r"^\[PAGE\s+\d+\]\s*", "", line, flags=re.IGNORECASE).strip()
        if 8 <= len(cleaned) <= 80 and not cleaned.lower().startswith("page "):
            return cleaned
    return re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE).strip() or "첨부 PDF"


def _pdf_key_points(text: str, *, limit: int) -> list[str]:
    lines = _rank_pdf_lines(text)
    if not lines:
        lines = _pdf_sentences(text)
    return [_compact_text(line, limit=180) for line in lines[:limit]] or [
        "문서에서 식별 가능한 핵심 문장이 부족합니다."
    ]


def _pdf_evidence_lines(text: str, *, limit: int) -> list[str]:
    candidates = [
        line
        for line in _pdf_lines(text)
        if re.search(
            r"\d|%|억원|매출|영업|계약|투자|AI|AX|cloud|클라우드", line, flags=re.IGNORECASE
        )
    ]
    if len(candidates) < limit:
        candidates.extend(_pdf_sentences(text))
    deduped = _dedupe_preserve_order(candidates)
    return [_compact_text(line, limit=200) for line in deduped[:limit]] or [
        "본문에서 직접 인용 가능한 근거 문장을 충분히 찾지 못했습니다."
    ]


def _pdf_executive_summary(*, title: str, bullets: list[str], page_count: int) -> str:
    lead = bullets[0] if bullets else title
    supporting = bullets[1:4]
    lines = [
        f"이 보고서는 '{title}' 문서에서 추출한 텍스트를 기준으로 작성한 초안입니다.",
        f"문서 범위는 약 {page_count or 1}개 페이지이며, 핵심 논점은 {lead}입니다.",
        (
            "주요 내용을 의사결정 관점에서 빠르게 검토할 수 있도록 문서 주요 내용, "
            "근거, 해석 한계, SK AX 관점 검토 포인트로 재구성했습니다."
        ),
    ]
    lines.extend(f"- {item}" for item in supporting)
    return "\n".join(lines)


def _pdf_skax_review_point(bullets: list[str], evidence: list[str]) -> str:
    lead = bullets[0] if bullets else "문서의 핵심 변화"
    basis = evidence[0] if evidence else "문서 근거"
    return (
        f"{lead}를 기준으로 고객·산업·기술 실행 영향이 SK AX의 제안, 운영, "
        f"보안 검토 항목에 연결되는지 확인해야 합니다. 판단 근거는 '{basis}'이며, "
        "후속 검토에서는 문서 안 수치와 일정이 실제 고객 대응 우선순위를 바꾸는지 분리해 보세요."
    )


def _string_list_from_payload(value: object, *, fallback: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return fallback
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized[:limit] or fallback


def _section_body(report_draft: object, title: str) -> str:
    if not isinstance(report_draft, dict):
        return ""
    sections = report_draft.get("sections")
    if not isinstance(sections, list):
        return ""
    for section in sections:
        if not isinstance(section, dict):
            continue
        if str(section.get("title") or "").strip() != title:
            continue
        return str(section.get("body") or "").strip()
    return ""


def _safe_pdf_confidence(value: object, text: str) -> float:
    parsed = _safe_pdf_float(value)
    if parsed is not None and 0.0 <= parsed <= 1.0:
        return parsed
    return 0.74 if len(text) >= 800 else 0.58


def _safe_pdf_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _safe_pdf_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _rank_pdf_lines(text: str) -> list[str]:
    keywords = (
        "AI",
        "AX",
        "cloud",
        "클라우드",
        "계약",
        "투자",
        "매출",
        "전략",
        "보안",
        "운영",
        "고객",
    )
    scored: list[tuple[int, str]] = []
    for line in _pdf_lines(text):
        keyword_score = sum(1 for keyword in keywords if keyword.lower() in line.lower())
        digit_score = 1 if re.search(r"\d", line) else 0
        length_score = 1 if 30 <= len(line) <= 180 else 0
        score = keyword_score * 3 + digit_score + length_score
        if score > 0:
            scored.append((score, line))
    scored.sort(key=lambda item: (-item[0], _pdf_lines(text).index(item[1])))
    return _dedupe_preserve_order([line for _, line in scored])


def _pdf_lines(text: str) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [
        line
        for line in lines
        if len(line) >= 12 and not re.fullmatch(r"\[PAGE\s+\d+\]", line, flags=re.IGNORECASE)
    ]


def _pdf_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?。？！다])\s+", normalized)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _compact_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "..."


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


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
    # 당일 신호·신규 카드가 없어 생성된 status placeholder("오늘은 주목할 동향 없음")는
    # 장애가 아니라 정상 빈 결과 → 502 가 아니라 200 으로 그대로 내려보내, 대시보드가
    # "오늘 중요한 뉴스 없음" 을 표시하고 평일 사전생성 cron 도 죽지 않게 한다.
    # LLM 실패 등 진짜 실패는 _raise_if_agent_failure 가 그대로 502 로 처리한다.
    if not _is_today_insight_status_placeholder(result):
        _raise_if_agent_failure("TODAY_INSIGHT", result)
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
    _raise_if_agent_failure("INSIGHT", result)
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
    _raise_if_agent_failure("MIXER", result)
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
            failure_reason = _agent_failure_reason(result)
            if failure_reason:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    _agent_failure_event("MIXER", result),
                )
                return
            payload = MixerAnalysisResponse.model_validate(result).model_dump(mode="json")
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": payload})
        except Exception as e:  # noqa: BLE001 — 모든 실패를 SSE error 로 전달
            log.warning("Mixer stream 실패: %s", e)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "error",
                    "message": _AGENT_CALL_FAILED_MESSAGE,
                    "error_code": "MIXER_STREAM_FAILED",
                    "detail": str(e),
                },
            )
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
            "global_only": request.global_only,
            "min_mention_count": request.min_mention_count,
            "max_trend_count": request.max_trend_count,
        },
    )
    # ITTrendAgent.generate 는 sync (5-phase 합산 ~70s, LLM 3 calls + DB 호출) —
    # event loop 를 막으면 liveness probe /healthz 도 응답 못해 SIGKILL.
    result = await asyncio.to_thread(ITTrendAgent().generate, trend_input)
    _raise_if_agent_failure("GLOBAL_TRENDS", result)
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
    """약한 신호 감지기 실행 (주 1회).

    W7 weak-signal read model 이 V30 이후 제거된 상태라 accepted 로 위장하지 않는다.
    """
    log.info("약한 신호 감지기 실행")
    raise HTTPException(
        status_code=501,
        detail={
            "code": "WEAK_SIGNAL_NOT_IMPLEMENTED",
            "message": "WeakSignalAgent는 현재 운영 경로에 연결되어 있지 않습니다.",
            "result_kind": "not_implemented",
        },
    )


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
    """저장된 카드뉴스를 우선 반환하고, 없을 때만 즉석 생성 경로를 사용한다."""
    saved_cards = _load_saved_card_news_items(limit=limit, today_only=today_only)
    if saved_cards:
        return saved_cards[:limit]

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


def _load_saved_card_news_items(limit: int, today_only: bool) -> list[dict[str, Any]]:
    from sqlalchemy import text

    from src.db.postgres import SessionLocal

    today_clause = ""
    params: dict[str, Any] = {"limit": max(1, limit)}
    if today_only:
        today_clause = "AND created_at >= :window_start"
        params["window_start"] = datetime.now(UTC) - timedelta(hours=24)

    query = text(f"""
        WITH active_cards AS (
            SELECT
                id, company, peer_company_id, cluster_id, title, summary_lines,
                event_type, importance, importance_score, implication, sources,
                validation_pass, validation_sc_score, primary_keyword_category,
                source_raw_article_ids, source_articles, image_assets, created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY cluster_id
                    ORDER BY created_at ASC, id ASC
                ) AS rn
            FROM card_news
            WHERE COALESCE(status, 'ACTIVE') = 'ACTIVE'
              {today_clause}
        )
        SELECT *
        FROM active_cards
        WHERE rn = 1
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
    """)
    try:
        with SessionLocal() as db:
            rows = db.execute(query, params).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.warning("저장 카드뉴스 조회 실패, 즉석 생성 경로로 대체 | error=%s", exc)
        return []
    return [_saved_card_news_item_from_row(dict(row)) for row in rows]


def _saved_card_news_item_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    created_at = row.get("created_at")
    published_date = _card_news_date_string(created_at)
    sources = _json_list(row.get("sources"))
    source_articles = _json_list(row.get("source_articles"))
    image_assets = _json_list(row.get("image_assets"))
    implication = _json_dict(row.get("implication"))
    frontend = _json_dict(implication.get("frontend"))
    summary_lines = _string_list(row.get("summary_lines"))
    insight_items = _frontend_text_items(
        frontend,
        ("key_implications", "peer_implications"),
        ("why_important", "potential_impact"),
    )
    action_items = _frontend_text_items(
        frontend,
        ("suggested_actions", "response_directions", "recommended_actions"),
        ("opportunities",),
    )
    cover_image = _cover_image_url(image_assets, sources, source_articles)
    peer_id = row.get("peer_company_id") or row.get("company")
    return {
        "id": row.get("id"),
        "company": row.get("company"),
        "peer_id": peer_id,
        "cluster_id": row.get("cluster_id"),
        "title": row.get("title"),
        "summary_lines": summary_lines,
        "event_type": row.get("event_type") or "tech",
        "sector": row.get("primary_keyword_category") or "other",
        "category_label": str(row.get("primary_keyword_category") or "AX").upper(),
        "date": published_date,
        "display_date": published_date,
        "published_date": published_date,
        "created_at": _iso_datetime(created_at),
        "importance": row.get("importance") or "low",
        "importance_score": row.get("importance_score") or 0.0,
        "exposure_band": row.get("importance") or "low",
        "exposure_score": row.get("importance_score") or 0.0,
        "implication": implication,
        "frontend_implication": frontend,
        "sources": sources,
        "source_articles": source_articles,
        "source_raw_article_ids": list(row.get("source_raw_article_ids") or []),
        "source_count": len(sources) or len(source_articles),
        "image_assets": image_assets,
        "display_sections": [
            {"type": "summary", "title": "요약", "items": summary_lines},
            {
                "type": "insight",
                "title": "시사점",
                "items": insight_items,
                "structured_items": _structured_display_items(insight_items),
            },
            {
                "type": "action",
                "title": "대응방안",
                "items": action_items,
                "structured_items": _structured_display_items(action_items),
            },
        ],
        "display": {
            "theme": "default",
            "background_asset_url": cover_image,
            "image_url": cover_image,
        },
        "is_human_reviewed": False,
        "is_bookmarked": False,
        "bookmark_count": 0,
        "share_count": 0,
        "validation_pass": bool(row.get("validation_pass")),
        "validation_sc_score": row.get("validation_sc_score") or 0.0,
    }


def _card_news_date_string(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(KST).date().isoformat()
    text_value = str(value or "").strip()
    if not text_value:
        return datetime.now(KST).date().isoformat()
    try:
        normalized = text_value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(KST).date().isoformat()
    except ValueError:
        return text_value[:10]


def _iso_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value or "")


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _frontend_text_items(
    frontend: Mapping[str, Any],
    primary_keys: tuple[str, ...],
    fallback_keys: tuple[str, ...],
) -> list[str]:
    for key in primary_keys:
        items = _string_list(frontend.get(key))
        if items:
            return items
    for key in fallback_keys:
        items = _string_list(frontend.get(key))
        if items:
            return items
    return []


def _structured_display_items(items: list[str]) -> list[dict[str, str]]:
    structured: list[dict[str, str]] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        conclusion, evidence = _split_labeled_display_text(text)
        structured.append({"main": conclusion or text, "detail": evidence})
    return structured


def _split_labeled_display_text(value: str) -> tuple[str, str]:
    text_value = re.sub(r"\s+", " ", value).strip()
    text_value = re.sub(r"^핵심\s*(시사점|대응)\s*[:：]\s*", "", text_value)
    if "근거/설명:" in text_value:
        main, detail = text_value.split("근거/설명:", 1)
        return main.strip(), detail.strip()
    return text_value, ""


def _cover_image_url(
    image_assets: list[Any],
    sources: list[Any],
    source_articles: list[Any],
) -> str:
    for collection in (image_assets, sources, source_articles):
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("url") or "").strip()
            if item.get("type") == "image" and url:
                return url
            image_url = str(item.get("image_url") or "").strip()
            if image_url:
                return image_url
            image_urls = item.get("image_urls")
            if isinstance(image_urls, list):
                for candidate in image_urls:
                    candidate_text = str(candidate or "").strip()
                    if candidate_text:
                        return candidate_text
    return ""


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
