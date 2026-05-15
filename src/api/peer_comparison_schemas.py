"""PeerComparison endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/30-analysis/peer-comparison.md`` §4 입력 / §5 출력.

prototype 범위 (Walking Skeleton Phase 2): Phase 1 (Current) + Phase 2 (Trend) +
Phase 4 (Strategic). Phase 3 (Forecast) 는 Day 90+ deferred — 응답 schema 는 보존
하되 ``forecasts`` 는 빈 배열로 반환.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PeerComparisonRequest(BaseModel):
    """``POST /peer/compare`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    peer_id: str = Field(..., description="비교 대상 peer id (예: samsung_sds)")
    window_days: int = Field(default=30, ge=1, le=365)
    focus_sector: Optional[str] = Field(default=None)


class TrendDelta(BaseModel):
    """Phase 2 — 정량 변화 지표. peer_financials 기반 deterministic 계산."""

    model_config = ConfigDict(extra="allow")

    metric: str
    qoq_pct: Optional[float] = None
    yoy_pct: Optional[float] = None
    band: Literal["normal", "유의", "급변"] = "normal"
    direction: Literal["up", "down", "flat"] = "flat"
    source: str = ""


class Differentiator(BaseModel):
    model_config = ConfigDict(extra="allow")

    aspect: str
    peer_position: str = ""
    skax_position: str = ""
    opportunity: str = ""


class Forecast(BaseModel):
    """Phase 3 — PDF §4 (1Q / 6M / 1Y). prototype 에서는 비활성."""

    model_config = ConfigDict(extra="allow")

    horizon: Literal["1Q", "6M", "1Y"]
    scenario: Literal["optimistic", "baseline", "pessimistic"]
    summary: str = ""
    drivers: list[str] = Field(default_factory=list)
    quantitative_estimate: Optional[str] = None
    risk_assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ReasoningTrailItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    seq: int
    label: str
    one_liner: str
    evidence_refs: list[str] = Field(default_factory=list)
    langfuse_observation_id: Optional[str] = None


class CoTStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step_idx: int
    phase: Literal["current", "trend", "forecast", "strategic"]
    question: str
    inputs_used: list[str] = Field(default_factory=list)
    answer: str
    intermediate_conclusion: str
    confidence: float
    langfuse_observation_id: Optional[str] = None


class PeerComparisonResponse(BaseModel):
    """``POST /peer/compare`` 응답. design §5 PeerComparisonOutput schema."""

    model_config = ConfigDict(extra="allow")

    peer_id: str
    # Phase 1 — Current
    strategy_label: str = ""
    differentiators: list[Differentiator] = Field(default_factory=list)
    strengths_of_peer: list[str] = Field(default_factory=list)
    weaknesses_of_peer: list[str] = Field(default_factory=list)
    collaboration_potential: list[str] = Field(default_factory=list)
    # Phase 2 — Trend
    trend_deltas: list[TrendDelta] = Field(default_factory=list)
    # Phase 3 — Forecast (prototype: 빈 배열)
    forecasts: list[Forecast] = Field(default_factory=list)
    # Phase 4 — Synthesis
    sk_ax_implication: str = ""
    final_one_liner: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)
    # Meta — 3-tier observability
    reasoning_trail: list[ReasoningTrailItem] = Field(default_factory=list)
    reasoning_steps: list[CoTStep] = Field(default_factory=list)
    langfuse_trace_id: Optional[str] = None
    confidence: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    analysis_period: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
