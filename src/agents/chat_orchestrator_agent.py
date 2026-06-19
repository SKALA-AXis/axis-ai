# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — ChatOrchestrator 프로토타입(intent·routing·compose), Supervisor 패턴
#   2026-05-19 박지원 — parser·agent 수정, 이후 카드뉴스 백필과 소스 카운트 추가
#   2026-06-08 박진 — 챗봇 에이전트·assistant RAG 추가, 포맷 정리, mixer chat·assistant 라우팅
"""Grounded chat orchestrator for the AXIS floating assistant.

The orchestrator is intentionally conservative:
* page context and exact DB lookups are preferred over broad retrieval,
* "run a tool" requests return page handoffs instead of executing agents inline,
* retrieved evidence is returned with the answer so the backend/frontend can store
  provenance in a compact assistant message row.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph
from sqlalchemy import text

from src.agents.chat._constants import (  # noqa: F401
    _AXIS_SCOPE_HINTS,
    _CHAT_GRAPH_MERMAID,
    _DEFAULT_LLM_MODEL,
    _MARKET_TREND_DAYS,
    _MARKET_TREND_LIMIT,
    _PAGE_REFERENCE_HINTS,
    _PEER_ALIASES,
    _PROMPT_VERSION,
    _RAG_FINAL_K,
    _RAG_PREFETCH_K,
    _RAG_RERANK_K,
    _RAG_VECTOR_K,
    _RECENT_NEWS_DAYS,
    _RECENT_NEWS_LIMIT,
    _SEARCH_STOPWORDS,
    _SECURITY_PATTERNS,
    _SELECTED_PEER_IDS,
    _VISIBLE_ID_ALIASES,
    ChatGraphState,
    RetrievalCandidate,
)
from src.agents.chat.intent_classifier import (  # noqa: F401
    _classify_intent,
    _is_history_report_request,
    _is_market_trend_request,
    _is_mixer_handoff_request,
    _is_print_followup_request,
    _route_after_intent,
)
from src.agents.chat.response_formatters import (  # noqa: F401
    _answer_blocks_from_candidates,
    _answer_blocks_from_report_draft,
    _base_response,
    _blocked_response,
    _candidate_fact_lines,
    _candidate_peer_lines,
    _card_report_window_days,
    _card_source_count,
    _chat_llm_model,
    _clean_history_content,
    _dedupe_strings,
    _dict_list,
    _dict_or_empty,
    _display_date,
    _display_peer_label,
    _extract_peer,
    _get_llm,
    _grounded_answer_prompt,
    _grounded_response,
    _handoff_response,
    _history_key_points,
    _history_report_response,
    _in_axis_scope,
    _is_card_news_report_request,
    _is_direct_lookup_request,
    _is_news_lookup_request,
    _is_obvious_off_topic,
    _is_pdf_or_report_export_request,
    _join_sentences,
    _jsonish_list,
    _llm,
    _normalize_report_draft,
    _numbered_lines,
    _parse_json_object,
    _primary_source_metadata,
    _print_help_response,
    _relative_date_from_query,
    _report_draft_from_candidates,
    _report_draft_from_history,
    _safe_confidence,
    _security_block_reason,
    _seed_report_body,
    _string_list,
    _today_insight_response,
    _visible_ids,
)
from src.agents.chat.retrieval_builders import (  # noqa: F401
    _briefing_row_to_candidate,
    _card_row_to_candidate,
    _dedupe_candidates,
    _issue_row_to_candidate,
    _news_lookup_response,
    _normalize_search_token,
    _peer_row_to_candidate,
    _raw_article_row_to_candidate,
    _rerank_candidates,
    _search_terms,
)
from src.contracts.chat_schemas import ChatTurnRequest
from src.db.postgres import SessionLocal
from src.observability.langfuse_client import (
    get_current_trace_id,
    tracing_config,
    with_session,
)

log = logging.getLogger(__name__)


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
