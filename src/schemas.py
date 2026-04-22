"""Pydantic 스키마 — axis-infra/api/ai-internal-api.yaml 기반 자동 생성
수동 수정 금지. datamodel-codegen으로 재생성:
  datamodel-codegen --input ../axis-infra/api/ai-internal-api.yaml --output src/schemas.py
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    peer_ids: list[str]
    trigger_type: str = "scheduled"


class PipelineRunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class SearchRequest(BaseModel):
    query: str
    peer_id: Optional[str] = None
    event_type: Optional[str] = None
    top_k: int = 10


class SearchHit(BaseModel):
    rdb_id: int
    peer_id: str
    title: str
    summary: str
    importance: str
    event_type: str
    pub_date: str
    rerank_score: float
    source_url: Optional[str] = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    total: int


class GenSearchRequest(BaseModel):
    query: str
    peer_id: Optional[str] = None
    top_k: int = 10


class GenSearchResult(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    sc_passed: bool
    sc_score: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    db_connected: bool
    qdrant_connected: bool
