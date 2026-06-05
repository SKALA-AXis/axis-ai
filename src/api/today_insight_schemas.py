"""Today's Insight endpoint Pydantic models.

Home dashboard 전용 요약이지만, 단순 UI fixture 가 아니라 매일의 통합 이슈와
과거 today_insight_reports JSONB 를 비교해 만든 executive-facing result schema 다.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TodayInsightGenerateRequest(BaseModel):
    """``POST /today-insight/generate`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    anchor_date: Optional[date] = Field(
        default=None,
        description="인사이트 기준일. 비어 있으면 Asia/Seoul 오늘.",
    )
    window_days: int = Field(
        default=60,
        ge=1,
        le=90,
        description="통합 이슈와 카드뉴스 입력 조회 window.",
    )
    max_issues: int = Field(default=8, ge=1, le=20, description="LLM 입력 통합 이슈 cap.")
    max_cards: int = Field(default=12, ge=1, le=30, description="연결 카드뉴스 cap.")
    use_cached: bool = Field(default=True, description="같은 기준일 최신 저장 결과 재사용.")
    force_refresh: bool = Field(default=False, description="캐시 무시 후 새로 생성.")
    refresh_policy: Literal["cache_first", "urgent_only"] = Field(
        default="cache_first",
        description=(
            "캐시가 있을 때의 refresh 정책. 현재 운영 정책은 cache_first 이며 "
            "urgent_only 는 하위 호환 입력으로만 허용하고 재생성 조건으로 쓰지 않는다."
        ),
    )
    urgent_importance_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="하위 호환 필드. 현재 Today's Insight 일중 refresh 에는 사용하지 않는다.",
    )
    cache_only: bool = Field(
        default=False,
        description=(
            "저장된 결과만 반환한다. 캐시가 없으면 생성하지 않고 scheduled pending 상태를 반환."
        ),
    )
    preload_model: bool = Field(
        default=False,
        description="캐시 반환 전에도 LLM client 를 선초기화해 로그인 직후 latency 를 낮춘다.",
    )
    save: bool = Field(default=True, description="today_insight_reports 에 결과 저장.")
    context: Optional[dict[str, Any]] = Field(default=None)


class TodayInsightReasoningStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    stage: str = ""
    detail: str = ""


class TodayInsightEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    grounds: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    related_keywords: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class TodayInsightSignal(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    label: str = ""
    value: str = ""
    reasoning: list[TodayInsightReasoningStep] = Field(default_factory=list)
    evidence: TodayInsightEvidence = Field(default_factory=TodayInsightEvidence)


class TodayInsightAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str = ""
    decision_owner: str = ""
    time_horizon: str = ""
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class TodayInsightSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    title: str = ""
    source_name: str = ""
    publisher: str = ""
    url: str = ""
    published_at: Optional[str] = None


class TodayInsightChangeSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    value: str = ""


class TodayInsightGenerateResponse(BaseModel):
    """Home dashboard Today's Insight response."""

    model_config = ConfigDict(extra="allow")

    report_date: str = ""
    generated_at: str = ""
    headline: str = ""
    executive_summary: str = ""
    executive_implication: str = ""
    change_summary: list[TodayInsightChangeSummary] = Field(default_factory=list)
    signals: list[TodayInsightSignal] = Field(default_factory=list)
    response_direction: list[TodayInsightAction] = Field(default_factory=list)
    sources: list[TodayInsightSource] = Field(default_factory=list)
    source_integrated_issue_ids: list[str] = Field(default_factory=list)
    source_card_ids: list[str] = Field(default_factory=list)
    peer_ids: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
