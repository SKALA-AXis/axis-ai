# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — InsightCascade prototype 스키마 신설 및 output validation helper 추가
"""Insight endpoint 의 request / response Pydantic 모델.

``src/schemas.py`` 는 datamodel-codegen 자동 생성 — 분석 prototype 의 신규 schema
는 본 모듈에 분리하여 보관. ai-internal-api.yaml 에 추가되면 자동 generation 으로
통합 가능.

design: ``axis-ai/design/30-analysis/insight-cascade.md`` §4 입력 / §5 출력.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class InsightGenerateRequest(BaseModel):
    """``POST /insight/generate`` 요청 body.

    Args:
        card_ids: 분석 대상 카드 id 목록 (최소 2개 권장, 최대 10개).
        context: frontend 가 추가 컨텍스트 제공 (예: 사용자 관심사 — 현재 prototype 미사용).
    """

    model_config = ConfigDict(extra="ignore")

    card_ids: list[str] = Field(..., min_length=1, description="카드 id 목록")
    context: Optional[dict[str, Any]] = Field(default=None)


class ReasoningTrailItem(BaseModel):
    """3-tier observability Tier 1 — 사용자 default 노출."""

    model_config = ConfigDict(extra="allow")

    seq: int = 0
    label: str = ""
    one_liner: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    langfuse_observation_id: Optional[str] = None


class CoTStep(BaseModel):
    """3-tier observability Tier 2 — 상세."""

    model_config = ConfigDict(extra="allow")

    step_idx: int = 0
    phase: str = ""
    question: str = ""
    inputs_used: list[str] = Field(default_factory=list)
    answer: str = ""
    intermediate_conclusion: str = ""
    confidence: float = 0.0
    langfuse_observation_id: Optional[str] = None


class InsightResponseAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str = ""
    priority: int = 99
    rationale: str = ""


class InsightGenerateResponse(BaseModel):
    """``POST /insight/generate`` 응답.

    design ``§5 InsightCascadeOutput`` 의 schema. 모든 필드 default 가 있어 LLM
    parse 실패 시에도 stub 반환 가능.
    """

    model_config = ConfigDict(extra="allow")

    cause: list[str] = Field(default_factory=list)
    change: list[str] = Field(default_factory=list)
    impact: list[str] = Field(default_factory=list)
    response: list[InsightResponseAction] = Field(default_factory=list)

    final_one_liner: str = ""
    sk_ax_implication: str = ""

    # 3-tier observability
    reasoning_trail: list[ReasoningTrailItem] = Field(default_factory=list)
    reasoning_steps: list[CoTStep] = Field(default_factory=list)
    langfuse_trace_id: Optional[str] = None

    follow_up_questions: list[str] = Field(default_factory=list)
    risk_assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    sources_used: list[str] = Field(default_factory=list)
    peer_ids: list[str] = Field(default_factory=list)

    provenance: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
