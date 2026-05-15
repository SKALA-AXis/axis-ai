"""GlobalTrends endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/30-analysis/global-trends.md`` §4 입력 / §5 출력.

prototype 범위 (Walking Skeleton Phase 2): Phase 1 (Snapshot) + Phase 2 (Trend Detection)
는 결정적 산식, Phase 3 (Impact Mapping) + Phase 4 (Forecast) + Phase 5 (Synthesis) 는
LLM 단일 호출. 글로벌 카드 데이터 부재 시 graceful 빈 응답 + warning.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GlobalTrendsRequest(BaseModel):
    """``POST /global/trends/run`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    company_ids: Optional[list[str]] = Field(
        default=None,
        description="None 이면 default 글로벌 6사 (NVIDIA / Apple / MS / Google / Amazon / Meta).",
    )
    focus_themes: Optional[list[str]] = Field(default=None)
    window_days: int = Field(default=30, ge=1, le=365)
    sk_ax_business_lines: Optional[list[str]] = Field(default=None)


class GlobalSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    company_id: str = ""
    card_count: int = 0
    top_themes: list[str] = Field(default_factory=list)
    headline_announcements: list[dict[str, Any]] = Field(default_factory=list)
    source_marker: str = ""


class TrendDetection(BaseModel):
    model_config = ConfigDict(extra="allow")

    theme: str = ""
    frequency_delta_pct: float = 0.0
    intensity: Literal["weak", "moderate", "strong"] = "weak"
    leading_companies: list[str] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)


class SKAXImpactCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    trend_theme: str = ""
    sk_ax_line: str = ""
    direction: Literal["positive", "neutral", "negative"] = "neutral"
    magnitude: Literal["low", "medium", "high"] = "low"
    channel: str = ""
    quant_hint: Optional[str] = None
    source_marker: str = ""


class GlobalForecast(BaseModel):
    model_config = ConfigDict(extra="allow")

    horizon: Literal["1Q", "6M", "1Y"] = "1Q"
    scenario: Literal["optimistic", "baseline", "pessimistic"] = "baseline"
    narrative: str = ""
    sk_ax_impact: str = ""
    drivers: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    recommended_response: str = ""


class ReasoningTrailItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    seq: int = 0
    label: str = ""
    one_liner: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    langfuse_observation_id: Optional[str] = None


class CoTStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step_idx: int = 0
    phase: Literal["snapshot", "trend_detect", "impact_map", "forecast", "synthesis"] = "synthesis"
    question: str = ""
    inputs_used: list[str] = Field(default_factory=list)
    answer: str = ""
    intermediate_conclusion: str = ""
    confidence: float = 0.0
    langfuse_observation_id: Optional[str] = None


class GlobalTrendsResponse(BaseModel):
    """``POST /global/trends/run`` 응답. design §5 GlobalTrendsOutput schema."""

    model_config = ConfigDict(extra="allow")

    analysis_period: dict[str, Any] = Field(default_factory=dict)
    snapshots: list[GlobalSnapshot] = Field(default_factory=list)
    trend_detections: list[TrendDetection] = Field(default_factory=list)
    impact_matrix: list[SKAXImpactCell] = Field(default_factory=list)
    forecasts: list[GlobalForecast] = Field(default_factory=list)

    final_one_liner: str = ""
    sk_ax_implication: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)

    reasoning_trail: list[ReasoningTrailItem] = Field(default_factory=list)
    reasoning_steps: list[CoTStep] = Field(default_factory=list)
    langfuse_trace_id: Optional[str] = None

    risk_assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    company_ids: list[str] = Field(default_factory=list)
    warning: Optional[str] = None
