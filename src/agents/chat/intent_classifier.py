"""chat intent_classifier — extracted from facade (move-only)."""

from __future__ import annotations

import re

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
from src.contracts.chat_schemas import ChatTurnRequest


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


def _is_market_trend_request(message: str) -> bool:
    compact = message.lower()
    has_market_scope = re.search(r"시장|업계|전체|전반|산업|경쟁사|peer|피어", compact)
    has_trend_intent = re.search(r"동향|트렌드|흐름|변화|추세|판도", compact)
    return bool(has_market_scope and has_trend_intent)


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
