"""Grounded chat orchestrator for the AXIS floating assistant.

The orchestrator is intentionally conservative:
* page context and exact DB lookups are preferred over broad retrieval,
* "run a tool" requests return page handoffs instead of executing agents inline,
* retrieved evidence is returned with the answer so the backend/frontend can store
  provenance in a compact assistant message row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text

from src.api.chat_schemas import ChatTurnRequest
from src.db.postgres import SessionLocal
from src.observability.langfuse_client import (
    get_current_trace_id,
    tracing_config,
    with_session,
)
from src.services.llm_env import ensure_llm_env_loaded

log = logging.getLogger(__name__)

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


_llm: Any | None = None


def _chat_llm_model() -> str:
    return os.getenv("CHAT_ORCHESTRATOR_MODEL", _DEFAULT_LLM_MODEL)


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        ensure_llm_env_loaded()
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model=_chat_llm_model(),
            temperature=0.12,
            max_completion_tokens=1100,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _llm


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


class ChatOrchestratorAgent:
    prompt_version = _PROMPT_VERSION

    def __init__(self, *, llm: Any | None = None, enable_llm: bool = True) -> None:
        self._llm = llm
        self._enable_llm = enable_llm
        self._graph = self._build_graph()

    async def answer(self, request: ChatTurnRequest) -> dict[str, Any]:
        initial: ChatGraphState = {
            "request": request,
            "message": request.message.strip(),
            "conversation_id": request.conversation_id or str(uuid.uuid4()),
            "message_id": str(uuid.uuid4()),
        }
        final_state = await asyncio.to_thread(self._graph.invoke, initial)
        return final_state["response"]

    def graph_mermaid(self) -> str:
        try:
            return str(self._graph.get_graph().draw_mermaid())
        except Exception:  # noqa: BLE001 - visualization helper only.
            return _CHAT_GRAPH_MERMAID

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)
        graph.add_node("security", self._node_security)
        graph.add_node("intent", self._node_intent)
        graph.add_node("blocked", self._node_blocked)
        graph.add_node("off_topic", self._node_off_topic)
        graph.add_node("print_help_response", self._node_print_help_response)
        graph.add_node("today_lookup", self._node_today_lookup)
        graph.add_node("today_response", self._node_today_response)
        graph.add_node("handoff_retrieve", self._node_handoff_retrieve)
        graph.add_node("handoff_response", self._node_handoff_response)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("grounded_response", self._node_grounded_response)

        graph.set_entry_point("security")
        graph.add_conditional_edges(
            "security",
            lambda state: "blocked" if state.get("security_reason") else "intent",
            {"blocked": "blocked", "intent": "intent"},
        )
        graph.add_conditional_edges(
            "intent",
            _route_after_intent,
            {
                "off_topic": "off_topic",
                "print_help": "print_help_response",
                "today_lookup": "today_lookup",
                "handoff_retrieve": "handoff_retrieve",
                "retrieve": "retrieve",
            },
        )
        graph.add_conditional_edges(
            "today_lookup",
            lambda state: "today_response" if state.get("insight") else "retrieve",
            {"today_response": "today_response", "retrieve": "retrieve"},
        )
        graph.add_edge("handoff_retrieve", "handoff_response")
        graph.add_edge("retrieve", "grounded_response")
        terminal_nodes = (
            "blocked",
            "off_topic",
            "print_help_response",
            "today_response",
            "handoff_response",
            "grounded_response",
        )
        for node in terminal_nodes:
            graph.add_edge(node, END)
        return graph.compile()

    def _node_security(self, state: ChatGraphState) -> dict[str, Any]:
        return {"security_reason": _security_block_reason(state["message"])}

    def _node_intent(self, state: ChatGraphState) -> dict[str, Any]:
        return {"intent": _classify_intent(state["message"], state["request"])}

    def _node_blocked(self, state: ChatGraphState) -> dict[str, Any]:
        return {
            "response": _blocked_response(
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
                reason=state.get("security_reason") or "security_policy",
            )
        }

    def _node_off_topic(self, state: ChatGraphState) -> dict[str, Any]:
        return {
            "response": _blocked_response(
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
                reason="out_of_scope",
                reply=(
                    "AXIS 화면, 카드뉴스, 인사이트, 브리핑, 믹서 결과와 관련된 질문에만 "
                    "답변할 수 있습니다. 현재 화면의 내용이나 경쟁사 동향 기준으로 질문해 주세요."
                ),
            )
        }

    def _node_print_help_response(self, state: ChatGraphState) -> dict[str, Any]:
        history_report = _history_report_response(
            request=state["request"],
            conversation_id=state["conversation_id"],
            message_id=state["message_id"],
        )
        if history_report:
            return {"response": history_report}
        return {
            "response": _print_help_response(
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
            )
        }

    def _node_today_lookup(self, state: ChatGraphState) -> dict[str, Any]:
        return {"insight": self._lookup_today_insight()}

    def _node_today_response(self, state: ChatGraphState) -> dict[str, Any]:
        return {
            "response": _today_insight_response(
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
                insight=state["insight"] or {},
            )
        }

    def _node_handoff_retrieve(self, state: ChatGraphState) -> dict[str, Any]:
        return {"candidates": self._retrieve_for_handoff(state["request"])}

    def _node_handoff_response(self, state: ChatGraphState) -> dict[str, Any]:
        return {
            "response": _handoff_response(
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
                candidates=state.get("candidates") or [],
            )
        }

    def _node_retrieve(self, state: ChatGraphState) -> dict[str, Any]:
        return {"candidates": self._retrieve(state["request"])}

    def _node_grounded_response(self, state: ChatGraphState) -> dict[str, Any]:
        if state.get("intent") == "news_lookup":
            return {
                "response": _news_lookup_response(
                    conversation_id=state["conversation_id"],
                    message_id=state["message_id"],
                    candidates=state.get("candidates") or [],
                    query=state["message"],
                )
            }
        if state.get("intent") == "lookup":
            return {
                "response": _grounded_response(
                    conversation_id=state["conversation_id"],
                    message_id=state["message_id"],
                    intent="lookup",
                    candidates=state.get("candidates") or [],
                )
            }
        return {
            "response": self._compose_grounded_response(
                request=state["request"],
                conversation_id=state["conversation_id"],
                message_id=state["message_id"],
                intent=state.get("intent") or "page_qa",
                candidates=state.get("candidates") or [],
            )
        }

    def _lookup_today_insight(self) -> dict[str, Any] | None:
        try:
            with SessionLocal() as db:
                row = (
                    db.execute(
                        text(
                            """
                        SELECT id::text AS id,
                               report_date::text AS report_date,
                               headline,
                               executive_summary,
                               executive_implication,
                               output_payload,
                               source_card_ids,
                               source_integrated_issue_ids,
                               peer_ids,
                               sectors,
                               confidence,
                               created_at
                          FROM today_insight_reports
                         WHERE status = 'active'
                         ORDER BY report_date DESC, created_at DESC
                         LIMIT 1
                        """
                        )
                    )
                    .mappings()
                    .first()
                )
        except Exception as exc:  # noqa: BLE001 - optional table in local/dev DBs.
            log.debug("today insight lookup skipped | error=%s", exc)
            return None
        return dict(row) if row else None

    def _retrieve_for_handoff(self, request: ChatTurnRequest) -> list[RetrievalCandidate]:
        visible = _visible_ids(request, "card_ids")
        candidates = self._lookup_cards(visible, limit=10) if visible else []
        if len(candidates) < 2:
            candidates = _dedupe_candidates(
                [*candidates, *self._lexical_card_search(request.message, limit=10)]
            )
        return candidates[:10]

    def _retrieve(self, request: ChatTurnRequest) -> list[RetrievalCandidate]:
        if _is_news_lookup_request(request.message):
            return self._recent_news_search(
                request.message,
                days=_RECENT_NEWS_DAYS,
                limit=_RECENT_NEWS_LIMIT,
            )

        report_cards = (
            self._recent_card_report_search(request.message, limit=10)
            if _is_card_news_report_request(request.message)
            else []
        )
        if report_cards and _is_pdf_or_report_export_request(request.message):
            return report_cards

        market_trend_candidates = (
            self._recent_peer_market_trend_search(
                days=_MARKET_TREND_DAYS,
                limit=_MARKET_TREND_LIMIT,
            )
            if _is_market_trend_request(request.message)
            else []
        )
        lexical = [
            *report_cards,
            *self._lexical_card_search(request.message, limit=8),
            *self._lexical_integrated_issue_search(request.message, limit=5),
            *self._lexical_briefing_search(request.message, limit=4),
            *self._lexical_peer_search(request.message, limit=4),
        ]
        if len(lexical) >= _RAG_FINAL_K and _is_direct_lookup_request(request.message):
            return _dedupe_candidates(lexical)[:_RAG_FINAL_K]

        vector = self._vector_search(request.message, top_k=_RAG_VECTOR_K)
        candidates = _dedupe_candidates([*market_trend_candidates, *lexical, *vector])
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:_RAG_RERANK_K]
        reranked = _rerank_candidates(request.message, candidates, top_k=_RAG_FINAL_K)
        if market_trend_candidates:
            return _dedupe_candidates([*market_trend_candidates, *reranked])[:_MARKET_TREND_LIMIT]
        return reranked

    def _lookup_cards(self, card_ids: list[str], *, limit: int) -> list[RetrievalCandidate]:
        ids = [card_id for card_id in card_ids if card_id]
        if not ids:
            return []
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            """
                        SELECT id,
                               title,
                               summary_lines,
                               event_type,
                               importance,
                               COALESCE(peer_company_id, company) AS peer_id,
                               created_at,
                               source_raw_article_ids,
                               sources,
                               source_articles
                          FROM card_news
                         WHERE id = ANY(CAST(:ids AS text[]))
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        {"ids": ids, "limit": int(limit)},
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("card lookup skipped | error=%s", exc)
            return []
        return [_card_row_to_candidate(row, score=1.0) for row in rows]

    def _lookup_integrated_issues(
        self, issue_ids: list[str], *, limit: int
    ) -> list[RetrievalCandidate]:
        ids = [issue_id for issue_id in issue_ids if issue_id]
        if not ids:
            return []
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            """
                        SELECT id::text AS id,
                               headline,
                               one_line_summary,
                               main_company,
                               event_type,
                               sectors,
                               confidence,
                               created_at
                          FROM integrated_issues
                         WHERE id = ANY(CAST(:ids AS uuid[]))
                           AND status = 'active'
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        {"ids": ids, "limit": int(limit)},
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("integrated issue lookup skipped | error=%s", exc)
            return []
        return [_issue_row_to_candidate(row, score=1.0) for row in rows]

    def _lexical_card_search(self, query: str, *, limit: int) -> list[RetrievalCandidate]:
        terms = _search_terms(query)
        if not terms:
            return []
        where = " OR ".join([f"global_search_text ILIKE :kw{i}" for i, _ in enumerate(terms)])
        params: dict[str, Any] = {f"kw{i}": f"%{term}%" for i, term in enumerate(terms)}
        params["limit"] = int(limit)
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id,
                               title,
                               summary_lines,
                               event_type,
                               importance,
                               COALESCE(peer_company_id, company) AS peer_id,
                               created_at,
                               source_raw_article_ids,
                               sources,
                               source_articles
                          FROM card_news
                         WHERE {where}
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("lexical search skipped | error=%s", exc)
            return []
        return [_card_row_to_candidate(row, score=0.62) for row in rows]

    def _lexical_integrated_issue_search(
        self, query: str, *, limit: int
    ) -> list[RetrievalCandidate]:
        terms = _search_terms(query)
        if not terms:
            return []
        where = " OR ".join([f"global_search_text ILIKE :kw{i}" for i, _ in enumerate(terms)])
        params: dict[str, Any] = {f"kw{i}": f"%{term}%" for i, term in enumerate(terms)}
        params["limit"] = int(limit)
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id::text AS id,
                               headline,
                               one_line_summary,
                               main_company,
                               event_type,
                               sectors,
                               confidence,
                               created_at,
                               sources
                          FROM integrated_issues
                         WHERE status = 'active'
                           AND ({where})
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("integrated issue lexical search skipped | error=%s", exc)
            return []
        return [_issue_row_to_candidate(row, score=0.59) for row in rows]

    def _lexical_briefing_search(self, query: str, *, limit: int) -> list[RetrievalCandidate]:
        terms = _search_terms(query)
        target_date = _relative_date_from_query(query)
        if not terms and target_date is None:
            return []
        filters = ["status IN ('completed', 'completed_partial')"]
        params: dict[str, Any] = {f"kw{i}": f"%{term}%" for i, term in enumerate(terms)}
        if target_date is not None:
            filters.append("COALESCE(report_date, date_to, date_from)::date = :target_date")
            params["target_date"] = target_date.isoformat()
        elif terms:
            filters.append(
                "("
                + " OR ".join([f"global_search_text ILIKE :kw{i}" for i, _ in enumerate(terms)])
                + ")"
            )
        params["limit"] = int(limit)
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id,
                               title,
                               briefing_type,
                               period_label,
                               key_summary,
                               sk_implication,
                               status,
                               COALESCE(report_date, date_to, date_from)::text AS report_date,
                               created_at
                          FROM briefing_reports
                         WHERE {" AND ".join(filters)}
                         ORDER BY COALESCE(report_date, date_to, date_from) DESC, created_at DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("briefing lexical search skipped | error=%s", exc)
            return []
        return [_briefing_row_to_candidate(row, score=0.56) for row in rows]

    def _lexical_peer_search(self, query: str, *, limit: int) -> list[RetrievalCandidate]:
        terms = _search_terms(query)
        if not terms:
            return []
        where = " OR ".join([f"global_search_text ILIKE :kw{i}" for i, _ in enumerate(terms)])
        params: dict[str, Any] = {f"kw{i}": f"%{term}%" for i, term in enumerate(terms)}
        params["limit"] = int(limit)
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id,
                               name,
                               tier,
                               keywords,
                               core_keywords,
                               profile_snapshot_generated_at,
                               financial_updated_at,
                               created_at
                          FROM peer_companies
                         WHERE is_active IS DISTINCT FROM false
                           AND ({where})
                         ORDER BY COALESCE(
                             profile_snapshot_generated_at,
                             financial_updated_at,
                             created_at
                         ) DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("peer lexical search skipped | error=%s", exc)
            return []
        return [_peer_row_to_candidate(row, score=0.5) for row in rows]

    def _recent_peer_market_trend_search(
        self, *, days: int, limit: int
    ) -> list[RetrievalCandidate]:
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            """
                        WITH ranked_cards AS (
                            SELECT id,
                                   title,
                                   summary_lines,
                                   event_type,
                                   importance,
                                   COALESCE(peer_company_id, company) AS peer_id,
                                   created_at,
                                   source_raw_article_ids,
                                   sources,
                                   source_articles,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY COALESCE(peer_company_id, company)
                                       ORDER BY created_at DESC
                                   ) AS rn
                              FROM card_news
                             WHERE created_at >= NOW() - (:days * INTERVAL '1 day')
                               AND COALESCE(peer_company_id, company) = ANY(
                                   CAST(:peer_ids AS text[])
                               )
                        )
                        SELECT id,
                               title,
                               summary_lines,
                               event_type,
                               importance,
                               peer_id,
                               created_at,
                               source_raw_article_ids,
                               sources,
                               source_articles
                          FROM ranked_cards
                         WHERE rn <= 2
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        {
                            "days": int(days),
                            "limit": int(limit),
                            "peer_ids": list(_SELECTED_PEER_IDS),
                        },
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("recent peer market trend search skipped | error=%s", exc)
            return []
        return [
            _card_row_to_candidate(
                row,
                score=0.88 - min(index, 6) * 0.03,
            )
            for index, row in enumerate(rows)
        ]

    def _recent_news_search(self, query: str, *, days: int, limit: int) -> list[RetrievalCandidate]:
        peer = _extract_peer(query)
        candidates = self._recent_raw_article_search(query, peer=peer, days=days, limit=limit)
        if len(candidates) >= limit:
            return candidates[:limit]
        return _dedupe_candidates(
            [
                *candidates,
                *self._recent_card_news_search(
                    query,
                    peer=peer,
                    days=days,
                    limit=limit - len(candidates),
                ),
            ]
        )[:limit]

    def _recent_raw_article_search(
        self,
        query: str,
        *,
        peer: tuple[str, str] | None,
        days: int,
        limit: int,
    ) -> list[RetrievalCandidate]:
        terms = _search_terms(query)
        conditions = ["COALESCE(published_at, created_at) >= NOW() - (:days * INTERVAL '1 day')"]
        params: dict[str, Any] = {"days": int(days), "limit": int(limit)}
        if peer:
            conditions.append("(peer_id = :peer_id OR :peer_id = ANY(peer_company_ids))")
            params["peer_id"] = peer[0]
        elif terms:
            conditions.append(
                "("
                + " OR ".join(
                    [
                        f"concat_ws(' ', title, content, source_name) ILIKE :kw{i}"
                        for i, _ in enumerate(terms)
                    ]
                )
                + ")"
            )
            params.update({f"kw{i}": f"%{term}%" for i, term in enumerate(terms)})
        else:
            return []
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id::text AS id,
                               peer_id,
                               source_name,
                               title,
                               content,
                               url,
                               COALESCE(published_at, created_at) AS published_at,
                               created_at
                          FROM raw_articles
                         WHERE {" AND ".join(conditions)}
                         ORDER BY COALESCE(published_at, created_at) DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("recent raw article search skipped | error=%s", exc)
            return []
        return [_raw_article_row_to_candidate(row, score=0.9) for row in rows]

    def _recent_card_news_search(
        self,
        query: str,
        *,
        peer: tuple[str, str] | None,
        days: int,
        limit: int,
    ) -> list[RetrievalCandidate]:
        if limit <= 0:
            return []
        terms = _search_terms(query)
        conditions = ["created_at >= NOW() - (:days * INTERVAL '1 day')"]
        params: dict[str, Any] = {"days": int(days), "limit": int(limit)}
        if peer:
            conditions.append("COALESCE(peer_company_id, company) = :peer_id")
            params["peer_id"] = peer[0]
        elif terms:
            conditions.append(
                "("
                + " OR ".join([f"global_search_text ILIKE :kw{i}" for i, _ in enumerate(terms)])
                + ")"
            )
            params.update({f"kw{i}": f"%{term}%" for i, term in enumerate(terms)})
        else:
            return []
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id,
                               title,
                               summary_lines,
                               event_type,
                               importance,
                               COALESCE(peer_company_id, company) AS peer_id,
                               created_at,
                               source_raw_article_ids,
                               sources,
                               source_articles
                          FROM card_news
                         WHERE {" AND ".join(conditions)}
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("recent card news search skipped | error=%s", exc)
            return []
        return [_card_row_to_candidate(row, score=0.76) for row in rows]

    def _recent_card_report_search(self, query: str, *, limit: int) -> list[RetrievalCandidate]:
        days = _card_report_window_days(query)
        peer = _extract_peer(query)
        rows = self._recent_card_report_rows(days=days, peer=peer, limit=limit)
        if not rows and days == 1:
            rows = self._recent_card_report_rows(days=2, peer=peer, limit=limit)
        return [
            _card_row_to_candidate(row, score=0.86 - min(index, 8) * 0.02)
            for index, row in enumerate(rows)
        ]

    def _recent_card_report_rows(
        self,
        *,
        days: int,
        peer: tuple[str, str] | None,
        limit: int,
    ) -> list[Any]:
        conditions = ["created_at >= NOW() - (:days * INTERVAL '1 day')"]
        params: dict[str, Any] = {"days": int(days), "limit": int(limit)}
        if peer:
            conditions.append("COALESCE(peer_company_id, company) = :peer_id")
            params["peer_id"] = peer[0]
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        text(
                            f"""
                        SELECT id,
                               title,
                               summary_lines,
                               event_type,
                               importance,
                               COALESCE(peer_company_id, company) AS peer_id,
                               created_at,
                               source_raw_article_ids,
                               sources,
                               source_articles
                          FROM card_news
                         WHERE {" AND ".join(conditions)}
                         ORDER BY created_at DESC
                         LIMIT :limit
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("recent card report search skipped | error=%s", exc)
            return []
        return list(rows)

    def _vector_search(self, query: str, *, top_k: int) -> list[RetrievalCandidate]:
        try:
            from src.rag.hybrid_search import hybrid_search

            hits = hybrid_search(query, top_k=min(top_k, _RAG_PREFETCH_K))
        except Exception as exc:  # noqa: BLE001
            log.debug("hybrid vector search skipped | error=%s", exc)
            return []
        candidates: list[RetrievalCandidate] = []
        for hit in hits:
            card_id = str(hit.get("card_news_id") or "")
            if not card_id:
                continue
            candidates.append(
                RetrievalCandidate(
                    source_type="card_news",
                    source_id=card_id,
                    title=str(hit.get("title") or ""),
                    snippet=str(hit.get("summary") or ""),
                    score=float(hit.get("score") or 0.55),
                    metadata={
                        "raw_article_id": hit.get("rdb_id"),
                        "peer_id": hit.get("company"),
                        "event_type": hit.get("event_type"),
                        "retrieval": "qdrant_hybrid_rrf",
                    },
                )
            )
        try:
            from src.rag.assistant_knowledge_index import search_assistant_knowledge

            knowledge_hits = search_assistant_knowledge(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            log.debug("assistant knowledge vector search skipped | error=%s", exc)
            knowledge_hits = []
        for hit in knowledge_hits:
            source_type = str(hit.get("source_type") or "assistant_knowledge")
            source_id = str(hit.get("source_id") or hit.get("point_id") or "")
            if not source_id:
                continue
            candidates.append(
                RetrievalCandidate(
                    source_type=source_type,
                    source_id=source_id,
                    title=str(hit.get("title") or ""),
                    snippet=str(hit.get("summary") or hit.get("text") or "")[:700],
                    score=float(hit.get("score") or 0.54),
                    metadata={
                        "peer_id": hit.get("peer_id"),
                        "event_type": hit.get("event_type"),
                        "retrieval": "qdrant_assistant_knowledge_rrf",
                    },
                )
            )
        return candidates

    def _compose_grounded_response(
        self,
        *,
        request: ChatTurnRequest,
        conversation_id: str,
        message_id: str,
        intent: str,
        candidates: list[RetrievalCandidate],
    ) -> dict[str, Any]:
        if intent == "report_lookup" and (_is_history_report_request(request) or not candidates):
            history_report = _history_report_response(
                request=request,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            if history_report:
                return history_report
        if not candidates or not self._enable_llm:
            return _grounded_response(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
                candidates=candidates,
                query=request.message,
            )
        try:
            llm_payload, trace_id = self._generate_grounded_answer(
                request=request,
                conversation_id=conversation_id,
                intent=intent,
                candidates=candidates,
            )
        except Exception as exc:  # noqa: BLE001 - LLM failure must not break chat.
            log.warning("chat grounded LLM compose fallback | error=%s", exc)
            return _grounded_response(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
                candidates=candidates,
                query=request.message,
            )

        response = _base_response(
            conversation_id=conversation_id,
            message_id=message_id,
            reply=str(llm_payload.get("reply") or "").strip(),
            intent=intent,
            scope="global_axis_data",
            sources=[candidate.to_source() for candidate in candidates],
            confidence=_safe_confidence(llm_payload.get("confidence"), default=0.72),
            follow_up=_string_list(
                llm_payload.get("follow_up_suggestions"),
                default=["관련 카드뉴스를 더 찾아줘", "이 내용을 브리핑 관점으로 정리해줘"],
            ),
            retrieval_mode="global_lexical+hybrid_rag+llm",
            answer_blocks=_dict_list(llm_payload.get("answer_blocks")),
            report_draft=(
                _normalize_report_draft(
                    llm_payload.get("report_draft"),
                    query=request.message,
                    candidates=candidates,
                )
                if intent == "report_lookup"
                else None
            ),
            llm_trace_id=trace_id,
            llm_model=_chat_llm_model(),
        )
        if not response["reply"]:
            return _grounded_response(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
                candidates=candidates,
                query=request.message,
            )
        if intent == "report_lookup" and _is_pdf_or_report_export_request(request.message):
            export_note = (
                "\n\n아래 보고서 초안의 'PDF 저장/출력' 버튼으로 바로 저장하거나 "
                "출력할 수 있습니다."
            )
            reply_text = response["reply"].rstrip()
            reply_text = reply_text.replace("작성하겠습니다", "작성했습니다")
            reply_text = reply_text.replace("생성하겠습니다", "생성했습니다")
            reply_text = reply_text.replace("정리하겠습니다", "정리했습니다")
            response["reply"] = reply_text + export_note
            response["provenance"]["export_requested"] = "pdf"
        return response

    def _generate_grounded_answer(
        self,
        *,
        request: ChatTurnRequest,
        conversation_id: str,
        intent: str,
        candidates: list[RetrievalCandidate],
    ) -> tuple[dict[str, Any], str | None]:
        prompt = _grounded_answer_prompt(request=request, intent=intent, candidates=candidates)
        llm = self._llm or _get_llm()
        with with_session(conversation_id):
            result = llm.invoke(
                prompt,
                config=tracing_config(
                    agent="ChatOrchestratorAgent",
                    phase="grounded_answer",
                    prompt_version=_PROMPT_VERSION,
                    session_id=conversation_id,
                    intent=intent,
                    source_count=len(candidates),
                ),
            )
        content = str(getattr(result, "content", result) or "")
        return _parse_json_object(content), get_current_trace_id()


def _classify_intent(message: str, request: ChatTurnRequest) -> str:
    compact = message.lower()
    if _is_obvious_off_topic(compact):
        return "off_topic"
    if _is_print_followup_request(message):
        return "print_help"
    if _is_mixer_handoff_request(compact):
        return "mixer_handoff"
    if _is_news_lookup_request(message):
        return "news_lookup"
    if _is_market_trend_request(message):
        return "market_trend"
    if _is_pdf_or_report_export_request(message):
        return "report_lookup"
    if re.search(r"핵심\s*신호|주요\s*신호|today'?s?\s*insight", compact) or (
        re.search(r"오늘|today", compact)
        and re.search(r"인사이트|동향|경쟁|카드|브리핑|보고서|리포트", compact)
    ):
        return "today_insight_summary"
    if re.search(r"브리핑|briefing|보고서|리포트", compact):
        return "report_lookup"
    if re.search(r"비교|compare|경쟁", compact):
        return "compare"
    if _is_direct_lookup_request(message) and _in_axis_scope(message, request):
        return "lookup"
    if _in_axis_scope(message, request):
        return "page_qa"
    return "off_topic"


def _is_mixer_handoff_request(compact_message: str) -> bool:
    if re.search(r"믹서|mixer|조합\s*분석|섞어서", compact_message):
        return True
    card_compare = re.search(r"카드|카드뉴스|뉴스", compact_message) and re.search(
        r"비교\s*분석", compact_message
    )
    competitor_compare = re.search(r"경쟁|경쟁사|peer|회사|기업", compact_message)
    return bool(card_compare and not competitor_compare)


def _route_after_intent(state: ChatGraphState) -> str:
    intent = state.get("intent")
    if intent == "off_topic":
        return "off_topic"
    if intent == "print_help":
        return "print_help"
    if intent == "today_insight_summary":
        return "today_lookup"
    if intent == "mixer_handoff":
        return "handoff_retrieve"
    return "retrieve"


def _security_block_reason(message: str) -> str | None:
    compact = message.lower()
    for pattern in _SECURITY_PATTERNS:
        if re.search(pattern, compact, flags=re.IGNORECASE):
            return "security_policy"
    return None


def _is_obvious_off_topic(compact_message: str) -> bool:
    weather = r"날씨|기온|강수|미세먼지|우산|비\s*와|눈\s*와|weather"
    location = r"내\s*위치|현재\s*위치|위치\s*알려|주소\s*알려|길\s*찾|지도|어디야"
    casual = r"점심|저녁\s*뭐|맛집|로또|운세|축구\s*결과|야구\s*결과"
    return bool(
        re.search(weather, compact_message)
        or re.search(location, compact_message)
        or re.search(casual, compact_message)
    )


def _is_news_lookup_request(message: str) -> bool:
    compact = message.lower()
    compact_no_space = re.sub(r"\s+", "", compact)
    if "카드뉴스" in compact_no_space and not re.search(r"기사|원문|언론|보도", compact):
        return False
    return bool(
        re.search(r"기사|원문|언론|보도|\bnews\b", compact)
        or ("뉴스" in compact and "카드뉴스" not in compact_no_space)
    )


def _is_market_trend_request(message: str) -> bool:
    compact = message.lower()
    has_market_scope = re.search(r"시장|업계|전체|전반|산업|경쟁사|peer|피어", compact)
    has_trend_intent = re.search(r"동향|트렌드|흐름|변화|추세|판도", compact)
    return bool(has_market_scope and has_trend_intent)


def _is_direct_lookup_request(message: str) -> bool:
    compact = message.lower()
    if re.search(r"요약|분석|비교|왜|시사점|의미|전망|대응|정리", compact):
        return False
    return bool(re.search(r"찾아|보여|알려|목록|리스트|관련\s*카드|관련\s*자료", compact))


def _is_pdf_or_report_export_request(message: str) -> bool:
    compact = message.lower()
    asks_export = re.search(
        r"pdf|피디에프|보고서|리포트|브리핑|출력|프린트|인쇄|다운로드|내보내|export|print",
        compact,
    )
    asks_create = re.search(r"만들|생성|작성|정리|요약|출력|프린트|인쇄|저장|다운로드", compact)
    return bool(asks_export and asks_create)


def _is_print_followup_request(message: str) -> bool:
    compact = message.lower()
    return bool(
        re.search(r"프린트|인쇄|출력|pdf\s*저장|pdf\s*다운로드|저장\s*/\s*출력", compact)
        and not re.search(r"보고서|리포트|브리핑|카드|카드뉴스|뉴스|신호|오늘|어제|최근", compact)
    )


def _is_history_report_request(request: ChatTurnRequest) -> bool:
    if not request.history:
        return False
    if not _is_pdf_or_report_export_request(request.message):
        return False
    return bool(
        re.search(
            r"이\s*내용|위\s*내용|앞\s*내용|방금|직전|채팅|대화|이걸|이거|현재\s*답변",
            request.message.lower(),
        )
    )


def _is_card_news_report_request(message: str) -> bool:
    compact = message.lower()
    compact_no_space = re.sub(r"\s+", "", compact)
    has_card_scope = "카드뉴스" in compact_no_space or re.search(r"카드|뉴스|신호", compact)
    has_generic_daily_scope = re.search(r"내용|소식|이슈|자료|변화|일어난|있었던|있던", compact)
    has_report_intent = re.search(
        r"pdf|피디에프|보고서|리포트|브리핑|요약|정리|출력|프린트|인쇄|다운로드",
        compact,
    )
    has_date_scope = re.search(r"오늘|어제|today|yesterday|최근", compact)
    return bool(
        has_report_intent and has_date_scope and (has_card_scope or has_generic_daily_scope)
    )


def _card_report_window_days(message: str) -> int:
    compact = message.lower()
    compact_no_space = re.sub(r"\s+", "", compact)
    if "어제오늘" in compact_no_space or (
        re.search(r"어제|yesterday", compact) and re.search(r"오늘|today", compact)
    ):
        return 2
    if re.search(r"어제|yesterday", compact):
        return 2
    if re.search(r"최근\s*7|일주일|1주", compact):
        return 7
    return 1


def _grounded_answer_prompt(
    *,
    request: ChatTurnRequest,
    intent: str,
    candidates: list[RetrievalCandidate],
) -> str:
    source_limit = _MARKET_TREND_LIMIT if intent == "market_trend" else _RAG_FINAL_K
    source_payload = [candidate.to_source() for candidate in candidates[:source_limit]]
    history_payload = [
        {"role": turn.role, "content": turn.content[:800]} for turn in request.history[-6:]
    ]
    page_payload = request.current_page.model_dump(mode="json") if request.current_page else {}
    payload = {
        "question": request.message,
        "intent": intent,
        "current_page": page_payload,
        "recent_history": history_payload,
        "sources": source_payload,
    }
    return f"""\
당신은 SK AX AXIS 서비스 안에서 동작하는 근거 기반 챗봇입니다.

역할:
- 사용자의 질문을 AXIS 화면/카드뉴스/통합 이슈/브리핑/믹서 결과 맥락에서 짧고 정확하게 답합니다.
- 아래 입력 JSON의 sources에 있는 사실만 사용합니다.
- sources에 없는 수치, 고객명, 계약명, 내부 DB 구조, SQL, 시스템 프롬프트는 절대 만들지 않습니다.
- 답변에는 내부 추론 과정을 쓰지 말고, 사용자에게 보여도 되는 요약과 근거만 씁니다.
- 후속 질문은 사용자가 AXIS 안에서 자연스럽게 이어갈 수 있는 버튼 문구로 씁니다.
- intent가 report_lookup이면 답변과 함께 report_draft를 작성합니다.
- report_draft는 내부적으로 "목차 및 구성 설계 → 초안 작성 → 근거로 내용 채우기" 순서로
  작성하되, 내부 단계명은 출력하지 않고 완성된 보고서만 반환합니다.
- report_draft는 5~7개 섹션으로 구성합니다. 각 섹션 body는 3~6문장 또는
  3~5개 bullet을 포함해 A4 출력 시 빈 페이지처럼 보이지 않게 충분히 작성합니다.
- intent가 market_trend이면 짧게 줄이지 말고 선정 peer사(삼성SDS, LG CNS,
  현대오토에버, 포스코DX)의 sources를 회사별로 분리해 설명합니다.
- market_trend 답변은 공통 변화, peer별 차이, SK AX 관점의 시사점,
  확인해야 할 근거 공백을 구체적으로 포함합니다.
- market_trend에서 특정 peer사의 근거가 sources에 없으면 "근거 부족"이라고
  표시하고 추정으로 채우지 않습니다.

입력 JSON:
{json.dumps(payload, ensure_ascii=False, default=str)}

출력은 JSON object 하나만 반환하세요.
{{
  "reply": "한국어 답변. 일반 질문은 2~5문장, market_trend는 더 길어도 되며 peer별 구체성을 우선",
  "answer_blocks": [
    {{"type": "summary", "title": "핵심 요약", "items": ["..."]}},
    {{"type": "evidence", "title": "근거", "items": ["source title 기반 근거"]}}
  ],
  "report_draft": {{
    "title": "보고서 제목. intent가 report_lookup이 아닐 때는 생략 가능",
    "sections": [
      {{"title": "섹션 제목", "body": "sources 기반 본문. 근거와 시사점을 채움"}}
    ]
  }},
  "follow_up_suggestions": ["후속 질문 1", "후속 질문 2"],
  "confidence": 0.0
}}
"""


def _parse_json_object(content: str) -> dict[str, Any]:
    text_value = content.strip()
    if not text_value:
        return {}
    try:
        parsed = json.loads(text_value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


def _safe_confidence(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
        if parsed <= 0 and default > 0:
            return default
        return max(0.0, min(parsed, 1.0))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any, *, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    out = [str(item).strip() for item in value if str(item).strip()]
    return out[:4] or default


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)][:4]


def _in_axis_scope(message: str, request: ChatTurnRequest) -> bool:
    compact = message.lower()
    if _extract_peer(message):
        return True
    if any(hint in compact for hint in _AXIS_SCOPE_HINTS):
        return True
    page = request.current_page
    return bool(page and page.route and any(hint in compact for hint in _PAGE_REFERENCE_HINTS))


def _visible_ids(request: ChatTurnRequest, key: str) -> list[str]:
    if not request.current_page:
        return []
    ids: list[str] = []
    for lookup_key in _VISIBLE_ID_ALIASES.get(key, (key,)):
        values = request.current_page.visible_item_ids.get(lookup_key) or []
        ids.extend(str(value) for value in values if value)
    return _dedupe_strings(ids)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _relative_date_from_query(query: str) -> date | None:
    compact = query.lower()
    today = datetime.now(UTC).date()
    if re.search(r"오늘|today", compact):
        return today
    if re.search(r"어제|yesterday", compact):
        return today - timedelta(days=1)
    match = re.search(r"(\d{1,2})\s*일\s*전", compact)
    if match:
        days = int(match.group(1))
        if 0 <= days <= 31:
            return today - timedelta(days=days)
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _search_terms(query: str) -> list[str]:
    peer = _extract_peer(query)
    tokens = [
        _normalize_search_token(token)
        for token in re.split(r"[\s,./|~?!]+", query)
        if len(token.strip()) >= 2
    ]
    tokens = [
        token
        for token in tokens
        if token and token not in _SEARCH_STOPWORDS and not token.endswith("해줘")
    ]
    if peer:
        tokens.extend([peer[0], peer[1]])
    return _dedupe_strings(tokens)[:5]


def _normalize_search_token(token: str) -> str:
    cleaned = token.strip().lower()
    cleaned = re.sub(r"(을|를|이|가|은|는|에|의|와|과|로|으로|만|좀)$", "", cleaned)
    return cleaned.strip()


def _extract_peer(query: str) -> tuple[str, str] | None:
    compact = query.lower()
    compact_no_space = re.sub(r"\s+", "", compact)
    for peer_id, label, alias in _PEER_ALIASES:
        alias_lower = alias.lower()
        if alias_lower in compact or alias_lower.replace(" ", "") in compact_no_space:
            return peer_id, label
    return None


def _card_row_to_candidate(row: Any, *, score: float) -> RetrievalCandidate:
    summary = row.get("summary_lines") or []
    snippet = " ".join(str(item) for item in summary if item) if isinstance(summary, list) else ""
    source_meta = _primary_source_metadata(row.get("sources"), row.get("source_articles"))
    source_count = _card_source_count(row)
    return RetrievalCandidate(
        source_type="card_news",
        source_id=str(row.get("id") or ""),
        title=str(row.get("title") or ""),
        snippet=snippet[:600],
        score=score,
        metadata={
            "peer_id": row.get("peer_id"),
            "event_type": row.get("event_type"),
            "importance": row.get("importance"),
            "created_at": str(row.get("created_at") or ""),
            "source_count": source_count,
            **source_meta,
        },
    )


def _issue_row_to_candidate(row: Any, *, score: float) -> RetrievalCandidate:
    source_meta = _primary_source_metadata(row.get("sources"))
    return RetrievalCandidate(
        source_type="integrated_issue",
        source_id=str(row.get("id") or ""),
        title=str(row.get("headline") or ""),
        snippet=str(row.get("one_line_summary") or "")[:600],
        score=score,
        metadata={
            "peer_id": row.get("main_company"),
            "event_type": row.get("event_type"),
            "sectors": row.get("sectors"),
            "created_at": str(row.get("created_at") or ""),
            **source_meta,
        },
    )


def _briefing_row_to_candidate(row: Any, *, score: float) -> RetrievalCandidate:
    snippet = str(row.get("key_summary") or row.get("sk_implication") or "")
    return RetrievalCandidate(
        source_type="briefing_report",
        source_id=str(row.get("id") or ""),
        title=str(row.get("title") or ""),
        snippet=snippet[:600],
        score=score,
        metadata={
            "briefing_type": row.get("briefing_type"),
            "period_label": row.get("period_label"),
            "report_date": row.get("report_date"),
            "created_at": str(row.get("created_at") or ""),
        },
    )


def _peer_row_to_candidate(row: Any, *, score: float) -> RetrievalCandidate:
    keywords = row.get("keywords") or row.get("core_keywords") or []
    snippet = (
        ", ".join(str(item) for item in keywords if item) if isinstance(keywords, list) else ""
    )
    return RetrievalCandidate(
        source_type="peer_profile",
        source_id=str(row.get("id") or ""),
        title=str(row.get("name") or row.get("id") or ""),
        snippet=snippet[:600],
        score=score,
        metadata={
            "peer_id": row.get("id"),
            "tier": row.get("tier"),
            "updated_at": str(
                row.get("profile_snapshot_generated_at")
                or row.get("financial_updated_at")
                or row.get("created_at")
                or ""
            ),
        },
    )


def _raw_article_row_to_candidate(row: Any, *, score: float) -> RetrievalCandidate:
    snippet = str(row.get("content") or "")[:600]
    return RetrievalCandidate(
        source_type="raw_article",
        source_id=str(row.get("id") or ""),
        title=str(row.get("title") or ""),
        snippet=snippet,
        score=score,
        metadata={
            "peer_id": row.get("peer_id"),
            "source_name": row.get("source_name"),
            "url": row.get("url"),
            "published_at": str(row.get("published_at") or ""),
            "created_at": str(row.get("created_at") or ""),
        },
    )


def _primary_source_metadata(*values: Any) -> dict[str, Any]:
    for value in values:
        for source in _jsonish_list(value):
            title = source.get("title") or source.get("headline")
            url = source.get("url") or source.get("source_url")
            source_name = source.get("source_name") or source.get("publisher")
            published_at = source.get("published_at") or source.get("created_at")
            if title or url or source_name or published_at:
                return {
                    "source_title": title,
                    "url": url,
                    "source_name": source_name,
                    "published_at": published_at,
                }
    return {}


def _card_source_count(row: Any) -> int:
    raw_ids = row.get("source_raw_article_ids") or []
    counts = [len(raw_ids) if isinstance(raw_ids, list) else 0]
    counts.extend(len(_jsonish_list(row.get(key))) for key in ("sources", "source_articles"))
    return max(counts) if counts else 0


def _jsonish_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    seen: set[tuple[str, str]] = set()
    out: list[RetrievalCandidate] = []
    for candidate in candidates:
        key = (candidate.source_type, candidate.source_id)
        if not candidate.source_id or key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _rerank_candidates(
    query: str, candidates: list[RetrievalCandidate], *, top_k: int
) -> list[RetrievalCandidate]:
    if len(candidates) <= top_k:
        return candidates[:top_k]
    rerank_input = [
        {
            "title": candidate.title,
            "summary": candidate.snippet,
            "_candidate": candidate,
        }
        for candidate in candidates
    ]
    try:
        from src.rag.reranker import rerank

        ranked = rerank(query, rerank_input, top_k=top_k)
        out: list[RetrievalCandidate] = []
        for item in ranked:
            candidate = item["_candidate"]
            candidate.score = float(item.get("rerank_score") or candidate.score)
            out.append(candidate)
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("chat rerank fallback | error=%s", exc)
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def _today_insight_response(
    *, conversation_id: str, message_id: str, insight: dict[str, Any]
) -> dict[str, Any]:
    payload = _dict_or_empty(insight.get("output_payload"))
    headline = str(payload.get("headline") or insight.get("headline") or "오늘의 핵심 신호")
    summary = str(payload.get("executive_summary") or insight.get("executive_summary") or "")
    implication = str(
        payload.get("executive_implication") or insight.get("executive_implication") or ""
    )
    raw_signals = payload.get("signals")
    signals = raw_signals if isinstance(raw_signals, list) else []
    signal_lines = [
        str(signal.get("summary") or signal.get("title") or "")
        for signal in signals
        if isinstance(signal, dict)
    ][:3]
    bullet = "\n".join(f"- {line}" for line in signal_lines if line)
    reply = f"{headline}\n\n{summary}".strip()
    if bullet:
        reply = f"{reply}\n\n{bullet}"
    if implication:
        reply = f"{reply}\n\nSK AX 관점: {implication}"
    sources = [
        {
            "type": "today_insight",
            "id": str(insight.get("id") or ""),
            "title": headline,
            "snippet": summary[:500],
            "score": 1.0,
        }
    ]
    return _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply or "오늘 생성된 Today Insight를 아직 찾지 못했습니다.",
        intent="today_insight_summary",
        scope="current_page",
        sources=sources,
        confidence=float(insight.get("confidence") or payload.get("confidence") or 0.74),
        follow_up=[
            "근거 카드뉴스도 함께 보여줘",
            "경쟁사별로 나눠서 설명해줘",
            "SK AX 대응 방향만 따로 정리해줘",
        ],
        retrieval_mode="today_insight_shortcut",
    )


def _news_lookup_response(
    *,
    conversation_id: str,
    message_id: str,
    candidates: list[RetrievalCandidate],
    query: str,
) -> dict[str, Any]:
    peer = _extract_peer(query)
    label = peer[1] if peer else "관련"
    if not candidates:
        return _base_response(
            conversation_id=conversation_id,
            message_id=message_id,
            reply=(
                f"최근 {_RECENT_NEWS_DAYS}일 기준으로 {label} 기사를 찾지 못했습니다. "
                "기업명이나 키워드를 조금 더 구체화해 주세요."
            ),
            intent="news_lookup",
            scope="global_axis_data",
            sources=[],
            confidence=0.28,
            follow_up=[
                f"{label} 카드뉴스까지 넓혀서 찾아줘",
                "최근 2주 기준으로 다시 찾아줘",
            ],
            retrieval_mode="recent_news_fast_path_empty",
        )

    lines = []
    for index, candidate in enumerate(candidates[:_RECENT_NEWS_LIMIT], start=1):
        meta = candidate.metadata or {}
        basis_at = _display_date(meta.get("published_at") or meta.get("created_at"))
        source_name = str(meta.get("source_name") or "").strip()
        suffix = " · ".join(item for item in (basis_at, source_name) if item)
        suffix = f" ({suffix})" if suffix else ""
        lines.append(f"{index}. {candidate.title}{suffix}")

    reply = (
        f"최근 {_RECENT_NEWS_DAYS}일 이내 {label} 기사 중심으로 최신순 정리했습니다.\n\n"
        + "\n".join(lines)
    )
    return _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply,
        intent="news_lookup",
        scope="global_axis_data",
        sources=[candidate.to_source() for candidate in candidates],
        confidence=0.82 if len(candidates) >= 2 else 0.62,
        follow_up=[
            f"{label} 기사 내용을 3줄로 요약해줘",
            f"{label} 관련 카드뉴스도 같이 보여줘",
        ],
        retrieval_mode="recent_news_fast_path",
    )


def _display_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return raw[:10]


def _candidate_fact_lines(
    candidates: list[RetrievalCandidate],
    *,
    limit: int,
) -> list[str]:
    lines: list[str] = []
    for candidate in candidates[:limit]:
        title = str(candidate.title or "").strip()
        snippet = str(candidate.snippet or "").strip()
        meta = candidate.metadata or {}
        peer_id = str(meta.get("peer_id") or "").strip()
        event_type = str(meta.get("event_type") or "").strip()
        basis_at = _display_date(meta.get("published_at") or meta.get("created_at"))
        labels = " · ".join(item for item in (peer_id, event_type, basis_at) if item)
        prefix = f"[{labels}] " if labels else ""
        if title and snippet:
            line = f"{prefix}{title} — {snippet}"
        else:
            line = f"{prefix}{title or snippet}"
        if line.strip():
            lines.append(line[:520])
    return lines


def _candidate_peer_lines(candidates: list[RetrievalCandidate]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        meta = candidate.metadata or {}
        peer_id = str(meta.get("peer_id") or meta.get("main_company") or "").strip()
        if not peer_id:
            continue
        label = _display_peer_label(peer_id)
        grouped.setdefault(label, [])
        if candidate.title:
            grouped[label].append(str(candidate.title))
    lines = []
    for peer_label, titles in grouped.items():
        unique_titles = _dedupe_strings([title for title in titles if title])[:3]
        if unique_titles:
            lines.append(
                f"- {peer_label}: "
                + " / ".join(unique_titles)
                + (
                    " 신호가 확인됩니다. 고객군, 적용 기술, 발표 시점을 "
                    "분리해 추적할 필요가 있습니다."
                )
            )
    return lines[:5]


def _display_peer_label(peer_id: str) -> str:
    normalized = peer_id.lower()
    for known_id, label, _alias in _PEER_ALIASES:
        if normalized == known_id.lower() or normalized == label.lower():
            return label
    return peer_id


def _seed_report_body(sections: list[dict[str, str]]) -> str:
    values = [
        f"{section.get('title', '').strip()}: {section.get('body', '').strip()}"
        for section in sections[:3]
        if section.get("body", "").strip()
    ]
    return "\n".join(values)


def _normalize_report_draft(
    value: Any, *, query: str, candidates: list[RetrievalCandidate]
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return _report_draft_from_candidates(query, candidates) if candidates else None
    title = str(value.get("title") or "").strip() or "AXIS 보고서 초안"
    sections = []
    for section in value.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_title = str(section.get("title") or "").strip()
        body = str(section.get("body") or "").strip()
        if section_title and body:
            sections.append({"title": section_title, "body": body})
    if not sections and candidates:
        return _report_draft_from_candidates(query, candidates, title_override=title)
    if not sections:
        return None
    if candidates:
        return _report_draft_from_candidates(
            query,
            candidates,
            title_override=title,
            seed_sections=sections,
        )
    return {"title": title[:120], "sections": sections[:7]}


def _report_draft_from_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    *,
    title_override: str | None = None,
    seed_sections: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    title_seed = "AXIS 보고서 초안"
    if _is_card_news_report_request(query):
        title_seed = "AXIS 카드뉴스 요약 보고서"
    if _is_pdf_or_report_export_request(query):
        title_seed = "AXIS 카드뉴스 요약 PDF"
    if re.search(r"브리핑|briefing", query, flags=re.IGNORECASE):
        title_seed = "AXIS 브리핑 초안"
    if title_override:
        title_seed = title_override[:120]

    evidence_lines = _candidate_fact_lines(candidates, limit=8)
    card_count = sum(1 for candidate in candidates if candidate.source_type == "card_news")
    date_lines = [
        _display_date((candidate.metadata or {}).get("created_at"))
        for candidate in candidates
        if candidate.metadata
    ]
    date_lines = [line for line in _dedupe_strings(date_lines) if line]
    period = ", ".join(date_lines[:3]) if date_lines else "확인된 기간"
    peer_lines = _candidate_peer_lines(candidates)
    seed_body = _seed_report_body(seed_sections or [])
    outline_titles = [
        "Executive Summary",
        "핵심 신호 상세",
        "피어/시장 영향",
        "근거 및 해석 한계",
        "SK AX 검토 포인트",
        "후속 액션",
    ]
    sections = [
        {
            "title": "목차 및 구성",
            "body": "\n".join(
                f"{index}. {title}" for index, title in enumerate(outline_titles, start=1)
            ),
        },
        {
            "title": "Executive Summary",
            "body": "\n".join(
                [
                    (
                        f"{period} 기준 AXIS 근거 {len(candidates)}건"
                        f"{f' 중 카드뉴스 {card_count}건' if card_count else ''}을 바탕으로 "
                        "보고서 초안을 구성했습니다."
                    ),
                    "핵심 결론은 피어사의 고객 접점, 기술 적용 범위, 운영 역량 메시지가 "
                    "함께 강화되고 있다는 점입니다.",
                    "SK AX 관점에서는 동일 고객군에서 요구되는 레퍼런스, 보안/운영 안정성, "
                    "산업별 적용 사례를 연결해 제안 우선순위를 재점검해야 합니다.",
                    seed_body[:520] if seed_body else "",
                ]
            ),
        },
        {
            "title": "핵심 신호 상세",
            "body": "\n".join(
                f"{index}. {line}"
                for index, line in enumerate(
                    evidence_lines or ["확인된 상세 신호가 부족합니다."],
                    start=1,
                )
            ),
        },
        {
            "title": "피어/시장 영향",
            "body": "\n".join(
                peer_lines
                or [
                    "피어사별 직접 식별 정보가 제한적이므로, 현재 근거에서는 "
                    "개별 회사별 우열보다 시장 신호의 방향성을 우선 해석해야 합니다.",
                    "고객사는 단일 기능보다 클라우드, 데이터, AI 운영, "
                    "보안 대응을 묶은 실행 역량을 비교할 가능성이 큽니다.",
                    "따라서 SK AX는 피어사가 강조하는 적용 사례와 고객군을 "
                    "추적하며 자사 레퍼런스와 차별화 메시지를 나란히 정리해야 합니다.",
                ]
            ),
        },
        {
            "title": "근거 및 해석 한계",
            "body": "\n".join(
                [
                    f"- 활용 근거: {len(candidates)}건",
                    f"- 확인 기간: {', '.join(date_lines[:5])}"
                    if date_lines
                    else "- 확인 기간: 근거별 생성일 기준",
                    (
                        "- 본 초안은 AXIS에 저장된 카드뉴스, 통합 이슈, 브리핑, "
                        "피어 프로필 요약을 기반으로 합니다."
                    ),
                    (
                        "- 계약 금액, 일정, 고객명은 근거에 명시된 경우에만 사용해야 하며, "
                        "출력 전 원문 링크와 대조가 필요합니다."
                    ),
                    (
                        "- 동일 사건이 여러 카드에 반복 수집됐을 수 있으므로, "
                        "최종 보고서에서는 중복 이벤트를 통합해야 합니다."
                    ),
                ]
            ),
        },
        {
            "title": "SK AX 검토 포인트",
            "body": "\n".join(
                [
                    (
                        "- 고객군별로 피어사의 메시지가 어디에 집중되는지 분리합니다. "
                        "금융, 제조, 물류, 공공 등 산업별 요구가 다르면 "
                        "제안 논리도 달라져야 합니다."
                    ),
                    (
                        "- 보안/운영 안정성/AI 적용 범위를 하나의 제안 패키지로 "
                        "설명할 수 있는지 점검합니다."
                    ),
                    (
                        "- 피어사의 최신 레퍼런스가 실제 고객 전환, PoC, 플랫폼 계약, "
                        "운영 고도화 중 어느 단계인지 구분합니다."
                    ),
                    (
                        "- SK AX가 이미 보유한 사례와 비교해 즉시 방어할 메시지와 "
                        "추가 확보가 필요한 근거를 나눕니다."
                    ),
                ]
            ),
        },
        {
            "title": "후속 액션",
            "body": "\n".join(
                [
                    "- 원문 기사와 카드뉴스를 확인해 사실 관계와 날짜를 확정합니다.",
                    (
                        "- 피어사별 최근 30일 신호를 다시 조회해 반복 등장하는 "
                        "고객군과 기술 키워드를 표로 정리합니다."
                    ),
                    (
                        "- 내부 제안/영업 자료에 반영할 수 있는 메시지와, "
                        "아직 근거가 부족해 보류해야 할 메시지를 분리합니다."
                    ),
                    (
                        "- 필요하면 이 초안을 브리핑용 1페이지 요약과 "
                        "임원 보고용 3페이지 버전으로 각각 재작성합니다."
                    ),
                ]
            ),
        },
    ]
    return {"title": title_seed, "sections": sections}


def _history_report_response(
    *,
    request: ChatTurnRequest,
    conversation_id: str,
    message_id: str,
) -> dict[str, Any] | None:
    report_draft = _report_draft_from_history(request)
    if not report_draft:
        return None
    response = _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=(
            "직전 대화 내용을 바탕으로 보고서 초안을 만들었습니다.\n\n"
            "아래 보고서 초안의 'PDF 저장/출력' 버튼으로 바로 저장하거나 출력할 수 있습니다."
        ),
        intent="report_lookup",
        scope="assistant_history",
        sources=[],
        confidence=0.58,
        follow_up=["근거 카드뉴스까지 포함해서 다시 만들어줘", "임원 보고용으로 더 짧게 정리해줘"],
        retrieval_mode="chat_history_report",
        answer_blocks=_answer_blocks_from_report_draft(report_draft),
        report_draft=report_draft,
    )
    response["provenance"]["export_requested"] = "pdf"
    return response


def _report_draft_from_history(request: ChatTurnRequest) -> dict[str, Any] | None:
    assistant_turns = [
        _clean_history_content(turn.content)
        for turn in request.history[-8:]
        if turn.role == "assistant"
    ]
    assistant_turns = [
        content
        for content in assistant_turns
        if content
        and not content.startswith("안녕하세요")
        and "기능에 문제가 생겼습니다" not in content
    ]
    if not assistant_turns:
        return None

    user_turns = [
        _clean_history_content(turn.content) for turn in request.history[-8:] if turn.role == "user"
    ]
    user_turns = [content for content in user_turns if content]
    recent_assistant_turns = assistant_turns[-5:]
    key_points = _history_key_points(recent_assistant_turns, limit=7)
    source_text = "\n\n".join(recent_assistant_turns)
    title = "AXIS 대화 기반 보고서"
    if re.search(r"pdf|피디에프|출력|프린트|인쇄|다운로드", request.message, re.IGNORECASE):
        title = "AXIS 대화 기반 PDF"
    if re.search(r"브리핑|briefing", request.message, re.IGNORECASE):
        title = "AXIS 대화 기반 브리핑"

    sections = [
        {
            "title": "목차 및 구성",
            "body": "\n".join(
                [
                    "1. Executive Summary",
                    "2. 상세 정리",
                    "3. 대화 맥락",
                    "4. 해석 한계",
                    "5. SK AX 관점 후속 검토",
                ]
            ),
        },
        {
            "title": "Executive Summary",
            "body": "\n".join(f"- {point}" for point in key_points[:5])
            or _join_sentences([source_text], limit=950),
        },
        {
            "title": "상세 정리",
            "body": _numbered_lines(recent_assistant_turns, limit=2_200),
        },
        {
            "title": "대화 맥락",
            "body": _join_sentences(
                [f"사용자 요청: {turn}" for turn in user_turns[-3:]]
                or ["최근 챗봇 답변 내용을 보고서 형식으로 재구성했습니다."],
                limit=650,
            ),
        },
        {
            "title": "해석 한계",
            "body": (
                "이 초안은 직전 대화의 표시 내용만 바탕으로 작성되었습니다. "
                "따라서 원문 기사, 카드뉴스, 브리핑 원본, 생성 시각이 필요한 수치 정보는 "
                "최종 제출 전에 별도로 확인해야 합니다. 대화 중 축약된 표현은 보고서에서 "
                "맥락을 보강했지만, 근거에 없는 고객명·금액·확정 일정은 추가하지 않았습니다."
            ),
        },
        {
            "title": "SK AX 관점 후속 검토",
            "body": (
                "외부 제출 전 원문 카드뉴스, 관련 기사, 수치 근거를 함께 확인하고 "
                "고객군/산업별 제안 메시지와 연결해 우선순위를 점검해 주세요. "
                "특히 경쟁사 신호가 기술 역량 홍보인지, 실제 고객 적용인지, 운영 고도화인지 "
                "구분해야 SK AX의 대응 메시지가 과장 없이 정리됩니다."
            ),
        },
    ]
    return {"title": title, "sections": sections}


def _history_key_points(values: list[str], *, limit: int) -> list[str]:
    points: list[str] = []
    for value in values:
        for item in re.split(r"(?<=[.!?。])\s+|(?:\s*[•]\s*)|(?:\s+-\s+)", value):
            cleaned = item.strip(" -•\t\n")
            if len(cleaned) < 8:
                continue
            points.append(cleaned[:260])
            if len(points) >= limit:
                return points
    return points


def _numbered_lines(values: list[str], *, limit: int) -> str:
    text_value = "\n".join(
        f"{index}. {value}" for index, value in enumerate(values, start=1) if value
    )
    return text_value[:limit]


def _answer_blocks_from_report_draft(report_draft: dict[str, Any]) -> list[dict[str, Any]]:
    sections = report_draft.get("sections")
    if not isinstance(sections, list):
        return []
    items = []
    for section in sections[:2]:
        if not isinstance(section, dict):
            continue
        body = str(section.get("body") or "").strip()
        if body:
            items.append(body[:140])
    return [{"type": "summary", "title": "보고서 구성", "items": items}] if items else []


def _clean_history_content(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _answer_blocks_from_candidates(candidates: list[RetrievalCandidate]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    return [
        {
            "type": "summary",
            "title": "확인된 내용",
            "items": [
                candidate.title for candidate in candidates[:3] if str(candidate.title).strip()
            ],
        },
        {
            "type": "evidence",
            "title": "근거",
            "items": [
                candidate.snippet for candidate in candidates[:3] if str(candidate.snippet).strip()
            ],
        },
    ]


def _join_sentences(values: list[str], *, limit: int) -> str:
    text_value = " ".join(str(value).strip() for value in values if str(value).strip())
    return text_value[:limit]


def _grounded_response(
    *,
    conversation_id: str,
    message_id: str,
    intent: str,
    candidates: list[RetrievalCandidate],
    query: str = "",
) -> dict[str, Any]:
    if not candidates:
        return _base_response(
            conversation_id=conversation_id,
            message_id=message_id,
            reply=(
                "현재 화면과 저장된 카드뉴스에서 바로 근거로 삼을 내용을 찾지 못했습니다. "
                "기업명, 기간, 키워드를 조금 더 구체화해 주세요."
            ),
            intent=intent,
            scope="global_axis_data",
            sources=[],
            confidence=0.24,
            follow_up=["최근 7일 카드뉴스 기준으로 찾아줘", "금융권 경쟁사 동향으로 좁혀줘"],
            retrieval_mode="empty",
        )
    lines = [f"- {candidate.title}: {candidate.snippet}" for candidate in candidates[:4]]
    if intent == "report_lookup":
        report_draft = _report_draft_from_candidates(query or "보고서 초안", candidates)
        reply = "확인된 근거를 바탕으로 채팅 안에서 볼 수 있는 보고서 초안을 만들었습니다."
        if _is_pdf_or_report_export_request(query):
            reply += (
                "\n\n아래 보고서 초안의 'PDF 저장/출력' 버튼으로 바로 저장하거나 "
                "출력할 수 있습니다."
            )
    else:
        report_draft = None
        reply = "확인된 근거 기준으로 정리하면 다음과 같습니다.\n\n" + "\n".join(lines)
    response = _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply,
        intent=intent,
        scope="global_axis_data",
        sources=[candidate.to_source() for candidate in candidates],
        confidence=0.68 if len(candidates) >= 2 else 0.52,
        follow_up=["이 내용을 브리핑 관점으로 정리해줘", "관련 카드뉴스를 더 찾아줘"],
        retrieval_mode="global_lexical+hybrid_rag",
        answer_blocks=_answer_blocks_from_candidates(candidates),
        report_draft=report_draft,
    )
    if intent == "report_lookup" and _is_pdf_or_report_export_request(query):
        response["provenance"]["export_requested"] = "pdf"
    return response


def _handoff_response(
    *,
    conversation_id: str,
    message_id: str,
    candidates: list[RetrievalCandidate],
) -> dict[str, Any]:
    card_ids = [
        candidate.source_id for candidate in candidates if candidate.source_type == "card_news"
    ]
    handoff = {
        "type": "navigate",
        "target_route": "/mixer",
        "label": "Mixer에서 카드 선택하기",
        "handoff_id": str(uuid.uuid4()),
        "payload_preview": {
            "recommended_card_ids": card_ids[:10],
            "candidate_count": len(card_ids),
            "requires_user_selection": True,
        },
    }
    if len(card_ids) < 2:
        reply = (
            "Mixer 분석은 최소 2개 이상의 카드 선택이 필요합니다. 현재 질문에서 충분한 후보를 "
            "찾지 못했으니 Mixer 페이지에서 카드를 직접 선택해 주세요."
        )
    else:
        reply = (
            f"관련 카드 후보 {len(card_ids)}개를 찾았습니다. 제가 여기서 바로 실행하지 않고, "
            "Mixer 페이지로 넘겨 사용자가 카드를 직접 확인하고 선택하도록 준비하겠습니다."
        )
    response = _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply,
        intent="mixer_handoff",
        scope="handoff",
        sources=[candidate.to_source() for candidate in candidates],
        confidence=0.7 if len(card_ids) >= 2 else 0.38,
        follow_up=["카드 후보를 더 넓혀줘", "최근 2주 기준으로 다시 찾아줘"],
        retrieval_mode="handoff_candidate_search",
    )
    response["handoff"] = handoff
    return response


def _print_help_response(*, conversation_id: str, message_id: str) -> dict[str, Any]:
    response = _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=(
            "직전에 생성된 보고서 초안 카드에 있는 'PDF 저장/출력' 버튼을 사용해 주세요. "
            "버튼이 보이지 않으면 '오늘 카드뉴스를 PDF로 정리해줘'처럼 보고서를 다시 생성하면 "
            "출력 가능한 보고서 카드가 함께 표시됩니다."
        ),
        intent="print_help",
        scope="assistant_ui",
        sources=[],
        confidence=0.78,
        follow_up=["오늘 카드뉴스를 PDF로 다시 정리해줘", "어제 카드뉴스를 보고서로 만들어줘"],
        retrieval_mode="print_followup_help",
        answer_blocks=[
            {
                "type": "action",
                "title": "출력 방법",
                "items": [
                    "보고서 초안 카드의 PDF 저장/출력 버튼을 누릅니다.",
                    "현재 답변에 보고서 카드가 없으면 PDF 보고서를 다시 생성합니다.",
                ],
            }
        ],
    )
    response["provenance"]["export_requested"] = "pdf"
    return response


def _blocked_response(
    *,
    conversation_id: str,
    message_id: str,
    reason: str,
    reply: str | None = None,
) -> dict[str, Any]:
    return _base_response(
        conversation_id=conversation_id,
        message_id=message_id,
        reply=reply
        or "요청하신 내용은 AXIS 챗봇의 답변 범위 또는 보안 정책에 맞지 않아 도와드릴 수 없습니다.",
        intent="blocked",
        scope="policy",
        sources=[],
        confidence=0.0,
        follow_up=["현재 화면의 카드뉴스를 요약해줘", "오늘 인사이트를 설명해줘"],
        retrieval_mode="blocked",
        blocked=True,
        blocked_reason=reason,
    )


def _base_response(
    *,
    conversation_id: str,
    message_id: str,
    reply: str,
    intent: str,
    scope: str,
    sources: list[dict[str, Any]],
    confidence: float,
    follow_up: list[str],
    retrieval_mode: str,
    blocked: bool = False,
    blocked_reason: str | None = None,
    answer_blocks: list[dict[str, Any]] | None = None,
    report_draft: dict[str, Any] | None = None,
    llm_trace_id: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    response = {
        "conversation_id": conversation_id,
        "session_id": conversation_id,
        "message_id": message_id,
        "reply": reply,
        "intent": intent,
        "scope": scope,
        "answer_blocks": answer_blocks or [],
        "sources": sources,
        "related_items": {"cards": [], "briefings": [], "mixers": [], "reports": []},
        "follow_up_suggestions": follow_up,
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "handoff": None,
        "provenance": {
            "agent": "ChatOrchestratorAgent",
            "runtime": "langgraph_stategraph",
            "prompt_version": _PROMPT_VERSION,
            "retrieval_mode": retrieval_mode,
            "llm_model": llm_model,
            "langfuse_trace_id": llm_trace_id,
            "rag": {
                "embedding_model": "BAAI/bge-m3",
                "fusion": "qdrant_dense_sparse_rrf",
                "prefetch_k": _RAG_PREFETCH_K,
                "vector_top_k": _RAG_VECTOR_K,
                "reranker": "BAAI/bge-reranker-v2-m3",
                "rerank_top_k": _RAG_RERANK_K,
                "final_top_k": _RAG_FINAL_K,
            },
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    if report_draft:
        response["report_draft"] = report_draft
    return response


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
