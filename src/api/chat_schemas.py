from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatHistoryTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str


class ChatPageContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    route: str | None = None
    title: str | None = None
    visible_item_ids: dict[str, list[str]] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class ChatClientContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    locale: str = "ko-KR"
    timezone: str = "Asia/Seoul"


class ChatTurnRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    conversation_id: str | None = None
    session_id: str | None = None
    history: list[ChatHistoryTurn] = Field(default_factory=list)
    current_page: ChatPageContext | None = None
    client_context: ChatClientContext = Field(default_factory=ChatClientContext)
    stream: bool = False

    @model_validator(mode="after")
    def normalize_conversation_id(self) -> "ChatTurnRequest":
        if not self.conversation_id and self.session_id:
            self.conversation_id = self.session_id
        if not self.session_id and self.conversation_id:
            self.session_id = self.conversation_id
        return self


class ChatSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    id: str
    title: str | None = None
    snippet: str | None = None
    url: str | None = None
    score: float | None = None


class ChatHandoff(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["navigate"] = "navigate"
    target_route: str
    label: str
    handoff_id: str
    payload_preview: dict[str, Any] = Field(default_factory=dict)


class ChatTurnResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    conversation_id: str
    session_id: str
    message_id: str
    reply: str
    intent: str
    scope: str
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[ChatSource] = Field(default_factory=list)
    related_items: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    blocked: bool = False
    blocked_reason: str | None = None
    handoff: ChatHandoff | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
