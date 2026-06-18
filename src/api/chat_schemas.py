# 작성일: 2026-06-08
# 작성자: 박진
# 변경이력:
#   2026-06-08 박진 — 챗봇 에이전트·어시스턴트 RAG용 스키마 추가, 이후 PDF export·mixer 자격증명 및 챗봇 분석 워크플로 개선
#   2026-06-12 최종민 — 공용 스키마를 src.contracts 로 이전하는 호환 shim 으로 전환 (agents→api 의존 절단)
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
