"""FastAPI 요청/응답 Pydantic 스키마."""

from __future__ import annotations

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    peer_ids: list[str]
    trigger_type: str = "scheduled"


class PipelineRunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


class CrawlPreviewRequest(BaseModel):
    peer_id: str | None = None
    topics: list[str] | None = None
    corp_code: str | None = None
    recent_days: int = 1
    mode: str = "raw"


class CrawlPreviewResponse(BaseModel):
    status: str
    output_paths: list[str]
