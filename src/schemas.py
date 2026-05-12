"""Pydantic 스키마 — axis-infra/api/ai-internal-api.yaml 기반 자동 생성
수동 수정 금지. datamodel-codegen으로 재생성:
  datamodel-codegen --input ../axis-infra/api/ai-internal-api.yaml --output src/schemas.py
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunRequest(BaseModel):
    company: list[str]
    track: str = "A"
    trigger_type: str = "scheduled"


class PipelineRunResponse(BaseModel):
    task_id: str
    status: str
    message: str


class SearchRequest(BaseModel):
    query: str
    company: Optional[str] = None
    event_type: Optional[str] = None
    top_k: int = 10


class SearchHit(BaseModel):
    rdb_id: int
    company: str
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
    company: Optional[str] = None
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


# ── 일일 브리핑 (v4 — backend SES 통합) ─────────────────────────────────────
# axis-backend 가 PostgreSQL 의 today issue cards 조회 후 POST /pipeline/delivery
# 로 전달 → axis-ai 가 HTML/text 본문 빌더 후 BriefingContent 반환 → backend
# 의 SesMailService 가 AWS SES V2 SDK (IRSA) 로 발송.
# 자세한 spec: axis-infra/docs/SES_INTEGRATION.md


class BriefingCard(BaseModel):
    """backend 에서 받는 카드 1건 — IssueCardResponse 호환 (camelCase JSON ↔ snake_case)."""

    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    peer_id: Optional[str] = Field(default=None, alias="peerId")
    cluster_id: Optional[int] = Field(default=None, alias="clusterId")
    title: str
    event_type: Optional[str] = Field(default=None, alias="eventType")
    importance: Optional[str] = None
    importance_score: Optional[float] = Field(default=None, alias="importanceScore")
    created_at: Optional[str] = Field(default=None, alias="createdAt")


class BriefingRequest(BaseModel):
    cards: list[BriefingCard]


class BriefingContent(BaseModel):
    subject: str
    html: Optional[str] = None
    text: Optional[str] = None
    recipients: list[str] = []
