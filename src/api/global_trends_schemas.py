# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — GlobalTrends prototype 스키마 신설 및 output validation helper 추가, 이후 5-phase agent·peer alignment·Layer B 파이프라인 반영
#   2026-06-12 박지원 — 글로벌 트렌드 종합 요약 및 글로벌 기업 동향 필드 추가
"""GlobalTrends endpoint 의 request / response Pydantic 모델 (정본).

design: ``axis-ai/design/30-analysis/global-trends.md`` §4 입력 / §5 출력.
source-of-truth: ``axis-infra/api/openapi.yaml`` 의 ``GlobalTrendsRequest`` /
``GlobalTrendsResult`` components (git tracked, line 2933 / 6846 / 6874).

prototype 범위 (Walking Skeleton): Phase 1 (Snapshot) + Phase 2 (Trend Detection)
는 결정적 산식, Phase 3 (Peer Alignment) + Phase 4 (Impact Mapping) + Phase 5
(Forecast/Synthesis) 는 LLM 호출. 글로벌 카드 데이터 부재 시 graceful 빈 응답 + warning.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GlobalTrendsRequest(BaseModel):
    """``POST /global/trends/run`` 요청 body. openapi.yaml ``GlobalTrendsRequest`` 와 정합."""

    model_config = ConfigDict(extra="ignore")

    company_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "분석 대상 글로벌 회사 ids. 비어 있으면 default 6사 "
            "(NVIDIA / Apple / MS / Google / Amazon / Meta)."
        ),
    )
    focus_themes: Optional[list[str]] = Field(
        default=None,
        description="특정 theme 필터 (옵션)",
    )
    window_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="분석 window (일). default 30.",
    )
    sk_ax_business_lines: Optional[list[str]] = Field(
        default=None,
        description=(
            "SK AX 사업라인 (옵션). 비어 있으면 default "
            "[cloud_ax, manufacturing_ax, data_platform, smart_factory]."
        ),
    )
    peer_company_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Peer alignment 대상 (옵션). 비어 있으면 "
            "[sk_ax, samsung_sds, lg_cns, posco_dx, hyundai_autoever] 5사."
        ),
    )
    include_peer_alignment: bool = Field(
        default=True,
        description="Phase 3 peer alignment 포함 여부. design §4 metadata 옵션.",
    )
    global_only: bool = Field(
        default=False,
        description="글로벌 탭 전용 모드. SK AX 관점 정렬/영향 분석은 생략하고 글로벌 요약만 생성.",
    )
    min_mention_count: int = Field(
        default=3,
        ge=1,
        description="trend 후보 keyword 컷오프. design §4 metadata 옵션.",
    )
    max_trend_count: int = Field(
        default=8,
        ge=1,
        le=20,
        description="출력 keyword 수 cap (LLM 비용 보호). design §4 metadata 옵션.",
    )


class GlobalSnapshot(BaseModel):
    """Phase 1 산출. design §3.1 ``payload.snapshots`` 에 저장."""

    model_config = ConfigDict(extra="allow")

    company_id: str
    card_count: int = 0
    top_themes: list[str] = Field(default_factory=list)
    headline_announcements: list[dict[str, Any]] = Field(default_factory=list)
    source_marker: str = ""


class TrendDetection(BaseModel):
    """Phase 2 산출. design §3.1 ``payload.trend_detections`` 에 저장."""

    model_config = ConfigDict(extra="allow")

    theme: str
    frequency_delta_pct: float = 0.0
    intensity: Literal["weak", "moderate", "strong"] = "weak"
    leading_companies: list[str] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)
    mention_count: int = 0


class PeerAlignment(BaseModel):
    """Phase 3 산출. design §3.1 ``payload.peer_alignment[]`` 에 저장."""

    model_config = ConfigDict(extra="allow")

    peer_id: str
    alignment_type: Literal["aligned", "lagging", "missing", "diverging"] = "missing"
    alignment_score: float = 0.0
    peer_mention_count: int = 0
    global_mention_count: int = 0
    recency_gap_days: Optional[int] = None
    evidence_card_ids: list[str] = Field(default_factory=list)
    strategic_note: str = ""


class SKAXImpactCell(BaseModel):
    """Phase 4 산출. design §3.1 ``payload.impact_matrix[]`` 에 저장."""

    model_config = ConfigDict(extra="allow")

    trend_theme: str
    sk_ax_line: str
    direction: Literal["positive", "neutral", "negative"] = "neutral"
    magnitude: Literal["low", "medium", "high"] = "low"
    channel: str = ""
    quant_hint: Optional[str] = None
    source_marker: str = ""


class GlobalForecast(BaseModel):
    """Phase 5 산출. design §3.1 ``payload.forecasts[]`` 에 저장."""

    model_config = ConfigDict(extra="allow")

    horizon: Literal["1Q", "6M", "1Y"]
    scenario: Literal["optimistic", "baseline", "pessimistic"] = "baseline"
    narrative: str = ""
    sk_ax_impact: str = ""
    drivers: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    recommended_response: str = ""


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
    phase: Literal[
        "snapshot", "trend_detect", "peer_alignment", "impact_map", "forecast", "synthesis"
    ]
    question: str
    inputs_used: list[str] = Field(default_factory=list)
    answer: str
    intermediate_conclusion: str = ""
    confidence: float = 0.0
    langfuse_observation_id: Optional[str] = None


class GlobalTrendsResponse(BaseModel):
    """``POST /global/trends/run`` 응답. openapi.yaml ``GlobalTrendsResult`` 와 정합.

    design §5 출력 spec. Phase 별 산출물 + reasoning trail + provenance.
    """

    model_config = ConfigDict(extra="allow")

    analysis_id: str = ""
    analysis_period: dict[str, Any] = Field(default_factory=dict)
    snapshots: list[GlobalSnapshot] = Field(default_factory=list)
    trend_detections: list[TrendDetection] = Field(default_factory=list)
    peer_alignment: dict[str, list[PeerAlignment]] = Field(
        default_factory=dict,
        description="trend keyword → per-peer alignment list (5 peer × 1 keyword).",
    )
    impact_matrix: list[SKAXImpactCell] = Field(default_factory=list)
    forecasts: list[GlobalForecast] = Field(default_factory=list)

    final_one_liner: str = ""
    overall_summary: str = ""
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
    persisted_row_count: int = 0
    warning: Optional[str] = None

    validation: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "GlobalTrendsRequest",
    "GlobalSnapshot",
    "TrendDetection",
    "PeerAlignment",
    "SKAXImpactCell",
    "GlobalForecast",
    "ReasoningTrailItem",
    "CoTStep",
    "GlobalTrendsResponse",
]
