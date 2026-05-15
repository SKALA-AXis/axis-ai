"""ChatOrchestrator endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/40-user-query/chat-orchestrator.md`` §4 입력 / §5 출력.

prototype 범위 (Walking Skeleton Phase 2):
- intent 분류 (gpt-4o-mini) + 분석 agent 라우팅 + compose
- 지원 intent: insight / mixer / peer_compare / global_trends / link_verify / smalltalk
- deep_dive / alternative_view / search / summary 는 Day 90+ deferred
- typed follow_up_suggestions (5 lens 만)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

IntentEnum = Literal[
    "insight",
    "mixer",
    "peer_compare",
    "global_trends",
    "link_verify",
    "search",
    "summary",
    "smalltalk",
]


class ChatTurnRequest(BaseModel):
    """``POST /chat`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: Optional[str] = Field(default=None, description="세션 식별 (frontend localStorage)")
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="최근 turn 들 [{role, content}]"
    )


class ChatEntities(BaseModel):
    model_config = ConfigDict(extra="allow")

    peer_ids: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    card_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    date_range: Optional[dict[str, Any]] = None


class FollowUpSuggestion(BaseModel):
    """typed follow-up — frontend deep_dive_context 자동 구성용 (PDF §6)."""

    model_config = ConfigDict(extra="allow")

    label: str
    intent: IntentEnum
    deep_dive: bool = False
    topic_anchor: Optional[str] = None
    lens: Optional[Literal["technical", "financial", "competitive", "regulatory", "customer"]] = (
        None
    )


class ChatTurnResponse(BaseModel):
    """``POST /chat`` 응답. design §5 ChatTurnOutput schema."""

    model_config = ConfigDict(extra="allow")

    reply: str
    intent: IntentEnum
    entities: ChatEntities = Field(default_factory=ChatEntities)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_suggestions: list[FollowUpSuggestion] = Field(default_factory=list)
    final_one_liner: Optional[str] = None
    sk_ax_implication: Optional[str] = None
    deep_dive_depth: int = 1
    reasoning_steps: Optional[list[dict[str, Any]]] = None
    confidence: float = 0.0
    session_id: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    warning: Optional[str] = None
