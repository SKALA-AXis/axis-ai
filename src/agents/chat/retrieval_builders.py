"""chat retrieval_builders — extracted from facade (move-only)."""

from __future__ import annotations

import logging
import re
from typing import Any

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

log = logging.getLogger(__name__)


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
    article_id = str(row.get("id") or "")
    snippet = ""
    retrieval_source = "rdb_row"
    if article_id:
        try:
            from src.rag.content_index import get_raw_article_body

            body = get_raw_article_body(article_id)
            snippet = body.text[:600]
            retrieval_source = f"content_{body.source}"
        except Exception as exc:  # noqa: BLE001
            log.debug("raw article VDB body lookup skipped | id=%s error=%s", article_id, exc)
    if not snippet:
        snippet = str(row.get("content") or "")[:600]
    return RetrievalCandidate(
        source_type="raw_article",
        source_id=article_id,
        title=str(row.get("title") or ""),
        snippet=snippet,
        score=score,
        metadata={
            "peer_id": row.get("peer_id"),
            "source_name": row.get("source_name"),
            "url": row.get("url"),
            "published_at": str(row.get("published_at") or ""),
            "created_at": str(row.get("created_at") or ""),
            "retrieval": retrieval_source,
        },
    )


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
