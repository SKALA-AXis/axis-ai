"""BriefingGenerationAgent endpoint request / response models.

``src/schemas.py`` 는 datamodel-codegen 자동 생성 대상이라, 브리핑 생성 prototype
schema 는 본 모듈에 분리한다. ai-internal-api.yaml 에 계약이 추가되면 자동 생성
schema 로 통합할 수 있다.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BriefingGenerateRequest(BaseModel):
    """``POST /briefing/generate`` 요청 body.

    Args:
        briefing_type: 기준 기간 타입. custom 은 사용하지 않는다.
        anchor_date: 기간 계산 기준 날짜. daily 는 해당 일, weekly 는 주 시작일부터
            기준일까지, monthly 는 월초부터 기준일까지 계산한다. ``YYYY-MM`` 문자열도
            월간 anchor 로 허용하며 해당 월 1일로 해석된다.
        card_ids: 직접 선택한 카드 id. 들어와도 기간 밖 카드는 제외된다.
        integrated_issue_ids: 직접 선택한 integrated issue id. 들어와도 기간 밖 항목은 제외된다.
        peer_ids: 경쟁사 필터.
        sectors: 섹터 필터.
        requested_by_user_id: 요청 사용자 id.
        ratios: 향후 화면 비율/가중치 확장용 입력. 현재 브리핑 생성에서는 판단
            근거로 사용하지 않는다.
        user_context: 사용자가 추가로 넘긴 관심 맥락.
        limit: 기간 내 선택할 최대 카드 수.
        save: true 이면 briefing_reports 와 매핑 테이블에 저장한다.
        refine_display_copy: true 이면 화면 문장 정제 LLM 단계를 수행한다.
        reuse_saved: true(기본) 이면 필터 없는 기본형 요청에서 저장된 동일 기간
            브리핑을 재사용한다 (과거 기간 무기한·진행 중 기간 30분 TTL).
            false 면 강제 재생성.
    """

    model_config = ConfigDict(extra="ignore")

    briefing_type: Literal["daily", "weekly", "monthly"] = Field(default="daily")
    anchor_date: Optional[str] = Field(default=None)
    card_ids: Optional[list[str]] = Field(default=None)
    integrated_issue_ids: Optional[list[str]] = Field(default=None)
    peer_ids: Optional[list[str]] = Field(default=None)
    sectors: Optional[list[str]] = Field(default=None)
    requested_by_user_id: Optional[int] = Field(default=None)
    ratios: Optional[dict[str, Any]] = Field(default=None)
    user_context: Optional[str] = Field(default=None)
    limit: int = Field(default=20, ge=1, le=50)
    save: bool = Field(default=False)
    refine_display_copy: bool = Field(default=True)
    reuse_saved: bool = Field(default=True)


class BriefingGenerateResponse(BaseModel):
    """BriefingGenerationAgent output passthrough.

    Agent payload 는 화면/저장 스키마가 함께 담긴 확장 JSON 이므로, endpoint 에서는
    필드 확장을 허용하고 그대로 반환한다.
    """

    model_config = ConfigDict(extra="allow")

    id: str = ""
    agent: str = "BriefingGenerationAgent"
    prompt_version: str = ""
    title: str = ""
    briefing_type: Literal["daily", "weekly", "monthly"] = "daily"
    date_from: Any = None
    date_to: Any = None
    period_label: str = ""
    created_at: Any = None
    status: str = ""
    briefing_lead: str = ""
    key_summary: str = ""
    source_card_ids: list[str] = Field(default_factory=list)
    executive_summary: str = ""
    immediate_trends: list[dict[str, Any]] = Field(default_factory=list)
    watch_trends: list[dict[str, Any]] = Field(default_factory=list)
    sections: list[dict[str, Any]] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    dailySnapshot: dict[str, Any] = Field(default_factory=dict)  # noqa: N815
    weeklySnapshot: dict[str, Any] = Field(default_factory=dict)  # noqa: N815
    evidenceSources: list[str] = Field(default_factory=list)  # noqa: N815
    history: list[dict[str, Any]] = Field(default_factory=list)
    frontend_briefings: dict[str, Any] = Field(default_factory=dict)
    briefingReport: dict[str, Any] = Field(default_factory=dict)  # noqa: N815
    flowSteps: list[dict[str, Any]] = Field(default_factory=list)  # noqa: N815
    selected_cards: list[dict[str, Any]] = Field(default_factory=list)
    key_change_cards: list[dict[str, Any]] = Field(default_factory=list)
    interpretation_flow: dict[str, Any] = Field(default_factory=dict)
    related_card_ids: list[str] = Field(default_factory=list)
    source_integrated_issue_ids: list[str] = Field(default_factory=list)
    primary_card_news_id: Optional[str] = None
    hidden_details: list[dict[str, Any]] = Field(default_factory=list)
    briefing_basis: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    provenance: dict[str, Any] = Field(default_factory=dict)
