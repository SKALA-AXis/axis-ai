# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — MixerAnalysis prototype 스키마 신설
#   2026-05-22 심유정 — 에이전트 아키텍처·mixer 인사이트 개선 및 사용자 전략 액션 투영 필드 추가
#   2026-06-04 박진 — 통합 이슈 기반 mixer/브리핑 플로우와 챗봇 today insight 플로우 반영
"""Mixer endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/30-analysis/mixer-analysis.md`` §4 입력 / §5 출력.

InsightCascade 와 마찬가지로 ``src/schemas.py`` (datamodel-codegen 자동 생성) 에는
prototype 단계라 분리. ``ai-internal-api.yaml`` 에 추가되면 통합 가능.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MixerAnalysisRequest(BaseModel):
    """``POST /mixer/analyze`` 요청 body.

    Args:
        card_ids: 분석 대상 카드 id 목록 (2 ≤ N ≤ 20). 호환 입력.
        integrated_issue_ids: canonical 분석 단위 id 목록 (2 ≤ N ≤ 20 권장).
        ratios: peer / industry / keyword 비율 (frontend Mixer UI 슬라이더 결과).
        user_context: 사용자 자유 입력 컨텍스트.
        analysis_mode: 빠른 실행(quick) 또는 정확 분석(deep).
    """

    model_config = ConfigDict(extra="ignore")

    card_ids: Optional[list[str]] = Field(default=None, description="카드 id 목록 (2~20 권장)")
    integrated_issue_ids: Optional[list[str]] = Field(
        default=None,
        description="integrated_issues.id 목록 (card_ids보다 우선)",
    )
    ratios: Optional[dict[str, Any]] = Field(default=None)
    user_context: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(
        default=None, description="사용자별 맞춤 전략 projection 조회용 user id"
    )
    analysis_mode: Literal["quick", "deep"] = Field(default="quick")

    @model_validator(mode="after")
    def require_analysis_ids(self) -> "MixerAnalysisRequest":
        if not (self.integrated_issue_ids or self.card_ids):
            raise ValueError("integrated_issue_ids 또는 card_ids 중 하나는 필요합니다.")
        return self


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
    calculation: str = ""
    meaning: str = ""
    prompted_interpretation: str = ""
    analysis_prompt: str = ""
    support_count: int = 0
    total_count: int = 0
    matched_card_ids: list[str] = Field(default_factory=list)


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


class MixerEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    card_id: str = ""
    text: str = ""


class MixerInsightBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    finding: str = ""
    rationale: str = ""
    evidence: list[MixerEvidenceRef] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)


class MixerActionDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str = ""
    why: str = ""
    use_case: str = ""
    evidence: list[MixerEvidenceRef] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)


class ReasoningTrailItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    seq: int = 0
    label: str = ""
    one_liner: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    langfuse_observation_id: Optional[str] = None


class FollowUpCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = ""
    answer: str = ""
    purpose: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MixerAnalysisDepth(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["quick", "deep"] = "quick"
    label: str = ""
    summary: str = ""
    included_steps: list[str] = Field(default_factory=list)
    omitted_steps: list[str] = Field(default_factory=list)


class MixerDeepDiveDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    text: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MixerDeepDiveSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = ""
    summary: str = ""
    details: list[MixerDeepDiveDetail] = Field(default_factory=list)


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
    mix_insight: str = ""
    common_pattern: MixerInsightBlock = Field(default_factory=MixerInsightBlock)
    comparison_point: MixerInsightBlock = Field(default_factory=MixerInsightBlock)
    hidden_conclusion: MixerInsightBlock = Field(default_factory=MixerInsightBlock)
    action_details: list[MixerActionDetail] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    # Legacy UI-compatible fields.
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
    follow_up_checks: list[FollowUpCheck] = Field(default_factory=list)
    analysis_depth: MixerAnalysisDepth = Field(default_factory=MixerAnalysisDepth)
    deep_dive_sections: list[MixerDeepDiveSection] = Field(default_factory=list)
    confidence: float = 0.0

    sources_used: list[str] = Field(default_factory=list)
    source_integrated_issue_ids: list[str] = Field(default_factory=list)
    peer_ids: list[str] = Field(default_factory=list)

    provenance: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
