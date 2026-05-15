"""Mixer endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/30-analysis/mixer-analysis.md`` §4 입력 / §5 출력.

InsightCascade 와 마찬가지로 ``src/schemas.py`` (datamodel-codegen 자동 생성) 에는
prototype 단계라 분리. ``ai-internal-api.yaml`` 에 추가되면 통합 가능.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MixerAnalysisRequest(BaseModel):
    """``POST /mixer/analyze`` 요청 body.

    Args:
        card_ids: 분석 대상 카드 id 목록 (2 ≤ N ≤ 20).
        ratios: peer / industry / keyword 비율 (frontend Mixer UI 슬라이더 결과).
        user_context: 사용자 자유 입력 컨텍스트.
    """

    model_config = ConfigDict(extra="ignore")

    card_ids: list[str] = Field(..., min_length=1, description="카드 id 목록 (2~20 권장)")
    ratios: Optional[dict[str, Any]] = Field(default=None)
    user_context: Optional[str] = Field(default=None)


class RadarAxis(BaseModel):
    model_config = ConfigDict(extra="allow")

    axis: Literal[
        "peer_strategic_shift",
        "tech_investment",
        "market_position",
        "partnership_momentum",
        "regulatory_risk",
        "talent_movement",
    ] = "peer_strategic_shift"
    score: float = 0.0
    explanation: str = ""


class Connection(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_card_id: str = ""
    target_card_id: str = ""
    label: Literal["cause", "effect", "similar", "contrast", "reinforce"] = "similar"
    weight: float = 1.0
    reason: str = ""  # v2: 왜 이 연결이 non-obvious 인지 (≤ 60자)


class CrossCardFinding(BaseModel):
    """v2 신규 — 카드들을 함께 봐야 보이는 비명백한 발견.

    bullet_signals 와 분리: bullet_signals 가 cross-card 'signal' 이라면 본 필드는
    'finding' (의미 해석). pattern_type 으로 발견의 성격 분류.
    """

    model_config = ConfigDict(extra="allow")

    finding: str = ""
    evidence_card_ids: list[str] = Field(default_factory=list)
    pattern_type: Literal[
        "convergent_strategy",
        "divergent_strategy",
        "gap_in_market",
        "acceleration_signal",
        "timing_mismatch",
        "market_baseline",
    ] = "convergent_strategy"


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
    phase: Literal["per_card", "cross_card", "synthesis"] = "synthesis"
    question: str = ""
    inputs_used: list[str] = Field(default_factory=list)
    answer: str = ""
    intermediate_conclusion: str = ""
    confidence: float = 0.0
    langfuse_observation_id: Optional[str] = None


class MixerAnalysisResponse(BaseModel):
    """``POST /mixer/analyze`` 응답. design ``§5 MixerAnalysisOutput`` schema."""

    model_config = ConfigDict(extra="allow")

    mix_id: str
    insight: str = ""
    final_one_liner: str = ""
    sk_ax_implication: str = ""
    bullet_signals: list[str] = Field(default_factory=list)
    radar_axes: list[RadarAxis] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    cross_card_findings: list[CrossCardFinding] = Field(default_factory=list)

    reasoning_trail: list[ReasoningTrailItem] = Field(default_factory=list)
    reasoning_steps: list[CoTStep] = Field(default_factory=list)
    langfuse_trace_id: Optional[str] = None

    follow_up_questions: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    sources_used: list[str] = Field(default_factory=list)
    peer_ids: list[str] = Field(default_factory=list)

    provenance: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
