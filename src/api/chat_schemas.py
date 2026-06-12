"""호환 shim — 정주소는 src.contracts.chat_schemas (2-A1 agents→api 의존 절단)."""

from src.contracts.chat_schemas import (  # noqa: F401
    ChatClientContext,
    ChatHandoff,
    ChatHistoryTurn,
    ChatPageContext,
    ChatPdfRequest,
    ChatSource,
    ChatTurnRequest,
    ChatTurnResponse,
)
