"""chat _constants — extracted from facade (move-only)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypedDict

from src.contracts.chat_schemas import ChatTurnRequest  # noqa: F401

_PROMPT_VERSION = "chat-orchestrator-v2-peer-market-detail"


_RAG_PREFETCH_K = int(os.getenv("CHAT_RAG_PREFETCH_K", "24"))


_RAG_VECTOR_K = int(os.getenv("CHAT_RAG_VECTOR_K", "12"))


_RAG_RERANK_K = int(os.getenv("CHAT_RAG_RERANK_K", "12"))


_RAG_FINAL_K = int(os.getenv("CHAT_RAG_FINAL_K", "5"))


_RECENT_NEWS_DAYS = int(os.getenv("CHAT_RECENT_NEWS_DAYS", "7"))


_RECENT_NEWS_LIMIT = int(os.getenv("CHAT_RECENT_NEWS_LIMIT", "6"))


_MARKET_TREND_DAYS = int(os.getenv("CHAT_MARKET_TREND_DAYS", "30"))


_MARKET_TREND_LIMIT = int(os.getenv("CHAT_MARKET_TREND_LIMIT", "8"))


_DEFAULT_LLM_MODEL = "gpt-4o-mini"


_SECURITY_PATTERNS = (
    r"시스템\s*프롬프트",
    r"system\s*prompt",
    r"api[_\s-]*key",
    r"비밀번호|패스워드|password",
    r"secret|token|jwt",
    r"전체\s*db|db\s*덤프|database\s*dump",
    r"db\s*전체|덤프",
    r"select\s+.*\s+from",
    r"drop\s+table|truncate\s+table",
)


_AXIS_SCOPE_HINTS = (
    "axis",
    "sk ax",
    "skax",
    "카드",
    "카드뉴스",
    "인사이트",
    "브리핑",
    "믹서",
    "mixer",
    "보고서",
    "리포트",
    "대시보드",
    "동향",
    "경쟁",
    "경쟁사",
    "peer",
    "피어",
    "피어사",
    "lg cns",
    "lgcns",
    "엘지씨엔에스",
    "삼성sds",
    "samsung sds",
    "현대오토에버",
    "포스코dx",
    "금융",
    "제조",
    "클라우드",
    "ai",
    "수주",
    "계약",
    "핵심 신호",
)


_PAGE_REFERENCE_HINTS = (
    "현재 화면",
    "현재 페이지",
    "이 화면",
    "이 페이지",
    "여기",
    "보고 있는",
)


_SEARCH_STOPWORDS = {
    "오늘",
    "어제",
    "요약",
    "알려줘",
    "알려줘~",
    "찾아줘",
    "보여줘",
    "정리해줘",
    "설명해줘",
    "기사",
    "뉴스",
    "news",
    "pdf",
    "피디에프",
    "최근",
    "관련",
    "내용",
    "정보",
    "피어",
    "피어사",
    "위주",
    "만들어줘",
    "생성해줘",
    "출력해줘",
    "설명",
    "설명해줘",
    "설ㅈ명해줘",
}


_PEER_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("lg_cns", "LG CNS", "lg cns"),
    ("lg_cns", "LG CNS", "lgcns"),
    ("lg_cns", "LG CNS", "엘지씨엔에스"),
    ("samsung_sds", "삼성SDS", "삼성sds"),
    ("samsung_sds", "삼성SDS", "samsung sds"),
    ("samsung_sds", "삼성SDS", "samsungsds"),
    ("hyundai_autoever", "현대오토에버", "현대오토에버"),
    ("hyundai_autoever", "현대오토에버", "hyundai autoever"),
    ("posco_dx", "포스코DX", "포스코dx"),
    ("posco_dx", "포스코DX", "posco dx"),
    ("sk_ax", "SK AX", "sk ax"),
    ("sk_ax", "SK AX", "skax"),
    ("nvidia", "NVIDIA", "nvidia"),
    ("apple", "Apple", "apple"),
    ("microsoft", "Microsoft", "microsoft"),
    ("google", "Google", "google"),
    ("amazon", "Amazon", "amazon"),
    ("meta", "Meta", "meta"),
)


_SELECTED_PEER_IDS = ("samsung_sds", "lg_cns", "hyundai_autoever", "posco_dx")


_VISIBLE_ID_ALIASES = {
    "card_ids": (
        "card_ids",
        "card_news_ids",
        "card_news",
        "cards",
        "source_card_ids",
        "selected_card_ids",
    ),
    "integrated_issue_ids": (
        "integrated_issue_ids",
        "integrated_issues",
        "issue_ids",
        "issues",
        "source_integrated_issue_ids",
        "selected_integrated_issue_ids",
    ),
}


_CHAT_GRAPH_MERMAID = """\
flowchart TD
  START([START]) --> Security[security]
  Security -->|blocked| Blocked[blocked response]
  Security -->|ok| Intent[intent parser]
  Intent -->|off_topic| OffTopic[out-of-scope response]
  Intent -->|today_insight_summary| TodayLookup[today insight lookup]
  TodayLookup -->|found| TodayResponse[today insight response]
  TodayLookup -->|missing| Retrieve[page CAG + DB/RAG retrieve]
  Intent -->|mixer_handoff| HandoffRetrieve[candidate card lookup]
  HandoffRetrieve --> HandoffResponse[/mixer handoff]
  Intent -->|page_qa/compare/report_lookup/market_trend| Retrieve
  Retrieve --> Grounded[grounded answer composer]
  Grounded -->|sources| LLM[ChatOpenAI JSON answer]
  Grounded -->|no sources or LLM error| Template[deterministic fallback]
  Blocked --> END([END])
  OffTopic --> END
  TodayResponse --> END
  HandoffResponse --> END
  LLM --> END
  Template --> END
"""


@dataclass(slots=True)
class RetrievalCandidate:
    source_type: str
    source_id: str
    title: str
    snippet: str
    score: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_source(self) -> dict[str, Any]:
        payload = {
            "type": self.source_type,
            "id": self.source_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
        }
        if self.metadata:
            payload.update({k: v for k, v in self.metadata.items() if v is not None})
        return payload


class ChatGraphState(TypedDict, total=False):
    request: ChatTurnRequest
    message: str
    conversation_id: str
    message_id: str
    security_reason: str | None
    intent: str
    insight: dict[str, Any] | None
    candidates: list[RetrievalCandidate]
    response: dict[str, Any]
