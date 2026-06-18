"""card_news text_utils1 — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any

from src.composers.card_news.schema import (  # noqa: F401
    _CARD_DETAIL_MAX,
    _CARD_PROMPT_VERSION,
    _DEFAULT_COVER_IMAGE_ALT,
    _DEFAULT_COVER_IMAGE_URL,
    _DISPLAY_ITEM_MAX,
    _DISPLAY_ITEM_MIN,
    _DISPLAY_ITEM_PREFERRED,
    _DISPLAY_TRUNCATED_MARKER_PATTERN,
    _DISPLAY_ZONE,
    _EVENT_TO_FACT_BASIS_TYPE,
    _FACT_BASIS_EVIDENCE_TYPES,
    _FRONTEND_EVENT_TYPES,
    _FRONTEND_SECTOR_IDS,
    _SUMMARY_ACTION_TOKENS,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _attach_card_news_schema_fields,
)
from src.composers.card_news.text_utils0 import (  # noqa: F401
    _accent_color,
    _action_semantic_role_for_index,
    _action_semantic_role_for_linkage_field,
    _article_content_sentences,
    _bounded_detail_items,
    _build_cluster_fetch_ids,
    _card_news_id,
    _card_signals,
    _card_text,
    _card_text_from_card,
    _clean_display_truncated_fragment,
    _clean_issue_term,
    _clean_source_headline,
    _compact_ascii,
    _compact_card_title,
    _compact_repeated_subject_in_phrase,
    _company_surface_replacements,
    _db_record,
    _dedupe_keep_order,
    _detail_line_candidates,
    _detail_line_key,
    _display_company_name,
    _display_token_pieces,
    _first_amount_like_term,
    _first_list_item,
    _first_non_empty,
    _first_sentence_is_too_thin,
    _first_text,
    _follow_up_action_from_statement,
    _format_numeric_text,
    _get_llm,
    _grounding_path_matches,
    _has_final_consonant,
    _has_non_money_quantity,
    _has_numeric_or_period_signal,
    _has_schedule_signal,
    _has_target_capacity_or_schedule,
    _insight_candidates_from_strategy,
    _is_financial_only_title,
    _is_internal_analysis_copy,
    _is_labeled_public_frontend_item,
    _is_market_reaction_summary_line,
    _is_near_duplicate_detail,
    _is_signature_noun_phrase,
    _is_untranslated_english_text,
    _issue_execution_detail_phrase,
    _issue_signature,
    _issue_terms,
    _join_context_terms,
    _json_like_text,
    _labeled_frontend_ready_items,
    _linkage_area_records,
    _linkage_record_terms,
    _linkage_text,
    _list_dicts,
    _list_string,
    _list_value,
    _llm,
    _looks_like_action,
    _looks_like_article_boilerplate,
    _looks_like_non_summary_line,
    _looks_like_sentence_title,
    _low_specificity_display_match_term,
    _metadata,
    _normalize_card_news_statement_style,
    _normalize_display_token,
    _normalize_event_type,
    _normalize_peer_id,
    _normalize_summary_similarity_token,
    _now_iso,
    _optional_float,
    _optional_int,
    _order_articles,
    _parse_datetime_for_display,
    _parse_json,
    _polish_frontend_ready_internal_terms,
    _profile_linkage_level,
    _profile_phrase_from_peer_copy,
    _public_copy_cleanup,
    _public_sentences,
    _published_datetime,
    _set_card_signals,
    _skax_business_phrase_from_implication,
    _split_public_conclusion_evidence,
    _statement_to_check_subject,
    _string_or_none,
    _strip_number_prefix,
    _strip_public_section_prefixes,
    _subject_company_names,
    _surface_alias_pattern,
    _unique_nonempty,
)
from src.config.companies import company_name_ko
from src.config.global_companies import global_company_name_ko


def _business_context_title(title: str, *, summary: dict[str, Any], event_type: str) -> str:
    value = _clean_card_editorial_text(title)
    if not _is_financial_only_title(value):
        return value
    if str(event_type or summary.get("cluster_event_type") or "").casefold() not in {
        "earnings",
        "analyst_report",
        "stock_market",
    }:
        return value
    focus = _business_focus_from_summary(summary)
    peer_name = _display_company_name(summary.get("main_company"))
    if focus:
        if "전망" in value or "목표" in value:
            return f"{peer_name}, {focus} 성장 전망" if peer_name else f"{focus} 성장 전망"
        return f"{peer_name}, {focus} 중심 실적 변화" if peer_name else f"{focus} 중심 실적 변화"
    return value


def _business_focus_from_summary(summary: dict[str, Any]) -> str:
    facts = _summary_fact_text(summary)
    focus_rules = (
        (r"클라우드|MSP|CSP|데이터센터", "클라우드·AI 인프라"),
        (r"AI\s*에이전트|생성형\s*AI|챗GPT|브리티|AX", "AI·AX 사업"),
        (r"IT\s*서비스|IT서비스|SI|ITO|시스템통합|아웃소싱", "IT서비스 사업"),
        (r"SDV|차량\s*SW|차량SW|모빌리티|내비게이션", "모빌리티 SW 사업"),
        (r"물류|첼로|Cello", "디지털 물류 사업"),
        (r"ERP|SCM|전환|구축|운영", "엔터프라이즈 IT 사업"),
    )
    matches = [label for pattern, label in focus_rules if re.search(pattern, facts, re.I)]
    return "·".join(dict.fromkeys(matches[:2]))


def _plain_summary_lines(summary: dict[str, Any], *, use_llm: bool = False) -> list[str]:
    # Summary is a factual section: preserve IntegrationAgent copy and avoid
    # display-time rewriting, ranking, or LLM re-summary.
    del use_llm
    return _literal_summary_lines(summary)


def _summary_candidate_lines(lines: list[str], summary: dict[str, Any]) -> list[str]:
    candidates = [str(line or "").strip() for line in lines if str(line or "").strip()]
    intelligence = summary.get("cluster_fact_intelligence") or {}
    if isinstance(intelligence, dict):
        for group_name in ("common_facts", "unique_facts"):
            for item in _list_dicts(intelligence.get(group_name)):
                fact_text = str(item.get("fact") or "").strip()
                if fact_text:
                    candidates.append(fact_text)
    for fact_item in _list_dicts(summary.get("consolidated_facts")):
        text = str(fact_item.get("fact") or "").strip()
        if text:
            candidates.append(text)
    return candidates


def _summary_lines_too_similar(left: str, right: str) -> bool:
    left_tokens = _summary_similarity_tokens(left)
    right_tokens = _summary_similarity_tokens(right)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False
    overlap = len(left_tokens & right_tokens)
    ratio = overlap / min(len(left_tokens), len(right_tokens))
    if ratio >= 0.72:
        return True
    shared_actions = (left_tokens & right_tokens) & _SUMMARY_ACTION_TOKENS
    if shared_actions and ratio >= 0.6:
        return True
    shared_event_anchors = {
        token
        for token in left_tokens & right_tokens
        if token in {"mou", "업무협약", "협약", "선정", "수주", "계약", "도입", "구축"}
        or token.endswith("협약")
    }
    return bool(shared_event_anchors) and overlap >= 4 and ratio >= 0.55


def _summary_similarity_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}", str(text or ""))
    stopwords = {
        "사업",
        "계약",
        "체결",
        "완료",
        "밝혔다",
        "위한",
        "관련",
        "통해",
        "규모",
        "계획",
        "예정",
    }
    normalized_stopwords = {_normalize_summary_similarity_token(token) for token in stopwords}
    normalized_tokens: set[str] = set()
    for token in tokens:
        pieces = [token, *re.split(r"[·/&+_-]+", token)]
        for piece in pieces:
            normalized = _normalize_summary_similarity_token(piece)
            if normalized and normalized not in normalized_stopwords:
                normalized_tokens.add(normalized)
    return normalized_tokens


def _summary_focus_tokens(summary: dict[str, Any]) -> set[str]:
    focus_parts = [
        summary.get("headline"),
        summary.get("main_event"),
        summary.get("main_issue"),
        summary.get("one_line_summary"),
    ]
    if not any(str(part or "").strip() for part in focus_parts):
        for line in _list_string(summary.get("fact_summary")):
            if not _is_market_reaction_summary_line(line):
                focus_parts.append(line)
                break
    return _summary_similarity_tokens(" ".join(str(part or "") for part in focus_parts))


def _contains_key_number_text(text: str, summary: dict[str, Any]) -> bool:
    value = str(text or "")
    for item in _list_dicts(summary.get("key_numbers")):
        label = str(item.get("metric_label") or item.get("metric_name") or "").strip()
        raw_value = str(item.get("value") or "").strip()
        unit = str(item.get("unit") or "").strip()
        if label and label in value:
            return True
        if raw_value and raw_value in value:
            return True
        if unit and raw_value and f"{raw_value}{unit}" in value:
            return True
    return False


def mark_near_duplicate_card_candidates(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 회사·핵심 제품/서비스명이 겹치는 단일 기사 카드를 병합 후보로 표시한다."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for card in cards:
        signals = _card_signals(card)
        signature = str(signals.get("issue_signature") or "").strip()
        if not signature:
            card_text = _card_text_from_card(card)
            terms = _issue_terms(card.get("news_summary") or {}, card_text)
            if terms:
                signature = _issue_signature(terms)
                signals["issue_signature"] = signature
                signals["near_duplicate_terms"] = terms
                _set_card_signals(card, signals)
        company = str(card.get("company") or card.get("peer_id") or "").strip()
        if signature and company:
            buckets.setdefault((company, signature), []).append(card)

    for grouped in buckets.values():
        if len(grouped) < 2:
            continue
        cluster_ids = [
            cluster_id
            for cluster_id in (_optional_int(card.get("cluster_id")) for card in grouped)
            if cluster_id is not None
        ]
        card_ids = [str(card.get("id")) for card in grouped if card.get("id")]
        for card in grouped:
            signals = _card_signals(card)
            signals["near_duplicate_card_candidate"] = True
            signals["near_duplicate_reason"] = (
                "같은 회사와 핵심 제품/서비스 키워드가 반복된 단일 기사 카드 후보"
            )
            signals["near_duplicate_card_ids"] = card_ids
            signals["near_duplicate_cluster_ids"] = cluster_ids
            _set_card_signals(card, signals)
    return cards


def _frontend_implication(analysis: dict[str, Any]) -> dict[str, Any]:
    key_implications = _bounded_detail_lines(
        [
            analysis.get("analysis_summary"),
            analysis.get("impact_reason"),
            *(_list_string(analysis.get("strategic_meaning"))),
        ]
    )
    return {
        "why_important": str(analysis.get("analysis_summary") or "").strip(),
        "potential_impact": str(analysis.get("impact_reason") or "").strip(),
        "follow_up_questions": [],
        "key_implications": key_implications,
        "suggested_actions": [],
        "confidence": _optional_float(analysis.get("confidence")),
    }


def _implication(analysis: dict[str, Any]) -> dict[str, Any]:
    return _frontend_implication(analysis)


def _frontend_implication_from_result(
    implication: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """v4.0 schema 인식 — peer_implication / skax_implication dict 의 핵심 필드 추출.

    W2-4: 기존에 peer_implication 전체를 str() 으로 변환하던 버그 정정. CardNewsComposer
    가 frontend 에 보내는 표면 schema 와 일치.
    """
    skax = implication.get("skax_implication") or {}
    peer = implication.get("peer_implication") or {}
    frontend_ready_items = _frontend_ready_display_items(implication)
    industry_metadata: dict[str, Any] = {}
    if not (frontend_ready_items.get("insight") or frontend_ready_items.get("action")):
        frontend_ready_items = _industry_frontend_ready_display_items(implication)
        industry_metadata = frontend_ready_items.get("metadata") or {}

    peer_implications = frontend_ready_items.get("insight") or []
    suggested_actions = frontend_ready_items.get("action") or []
    why_important = peer_implications[0] if peer_implications else ""
    potential_impact = peer_implications[0] if peer_implications else ""
    confidence = _optional_float(implication.get("confidence"))
    payload: dict[str, Any] = {
        "why_important": why_important,
        "potential_impact": potential_impact,
        "key_implications": peer_implications,
        "peer_implications": peer_implications,
        "skax_implications": _bounded_detail_items(why_important, potential_impact),
        "response_directions": suggested_actions,
        "follow_up_questions": [],
        "suggested_actions": suggested_actions,
        "confidence": confidence,
    }
    if industry_metadata:
        payload.update(industry_metadata)
    if isinstance(skax, dict):
        if skax.get("opportunities"):
            payload["opportunities"] = _list_string(skax.get("opportunities"))
        if skax.get("threats"):
            payload["threats"] = _list_string(skax.get("threats"))
        if skax.get("business_line_mapping"):
            payload["business_line_mapping"] = _list_string(skax.get("business_line_mapping"))
    if isinstance(peer, dict):
        if peer.get("company_id"):
            payload["peer_company_id"] = peer.get("company_id")
        if peer.get("company_name_ko"):
            payload["peer_company_name_ko"] = peer.get("company_name_ko")
        if peer.get("capability_change"):
            payload["peer_capability_change"] = peer.get("capability_change")
        if peer.get("precedent_link"):
            payload["precedent_link"] = peer.get("precedent_link")
    if implication.get("evidence_label"):
        payload["evidence_label"] = implication.get("evidence_label")
    return _with_structured_frontend_blocks(payload)


def _implication_from_result(
    implication: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
    frontend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """W2-4: CardNewsComposer 가 DB 에 저장할 implication JSONB 의 단일 출처.

    v2 schema 의 `peer_implication` / `skax_implication` / `follow_up_questions` /
    `watch_points` / `confidence` / `evidence_label` / `provenance` 를 모두 포함하고,
    frontend 호환 핵심 필드도 함께 평면화.
    """
    fallback = fallback or {}
    payload = dict(implication)
    # v2 schema 가 사용하는 필드를 모두 안전 default 로 채운다.
    payload.setdefault("implication_scope", "peer_and_skax")
    payload.setdefault("watch_points", payload.get("watch_points", []))
    # frontend 호환 - flatten.
    payload["frontend"] = frontend or _frontend_implication_from_result(
        implication,
        fallback=fallback,
    )
    return payload


def _frontend_ready_display_items(implication: Any) -> dict[str, Any]:
    if not isinstance(implication, dict):
        return {"insight": [], "action": [], "insight_blocks": [], "action_blocks": []}
    frontend_ready = implication.get("frontend_ready") or {}
    if not isinstance(frontend_ready, dict):
        return {"insight": [], "action": [], "insight_blocks": [], "action_blocks": []}
    source = str(frontend_ready.get("source") or "").strip()
    insight_blocks = _frontend_ready_section_blocks(
        frontend_ready.get("key_implication"),
        source=source,
        anchor_key="profile_anchor_terms",
    )
    action_blocks = _frontend_ready_section_blocks(
        frontend_ready.get("suggested_action"),
        source=source,
        anchor_key="skax_anchor_terms",
    )
    return {
        "insight": _labeled_frontend_ready_items(insight_blocks, heading="핵심 시사점"),
        "action": _labeled_frontend_ready_items(action_blocks, heading="핵심 대응"),
        "insight_blocks": insight_blocks,
        "action_blocks": action_blocks,
    }


def _industry_frontend_ready_display_items(implication: Any) -> dict[str, Any]:
    if not isinstance(implication, dict):
        return {"insight": [], "action": [], "metadata": {}}
    industry_ready = implication.get("industry_frontend_ready") or {}
    if not isinstance(industry_ready, dict):
        return {"insight": [], "action": [], "metadata": {}}
    if str(industry_ready.get("source") or "").strip() != "industry_signal_direct":
        return {"insight": [], "action": [], "metadata": {}}
    metadata = {
        "signal_scope": str(industry_ready.get("signal_scope") or "industry_signal"),
        "display_policy": str(industry_ready.get("display_policy") or "industry_only"),
    }
    insight_items: list[str] = []
    action_items: list[str] = []
    insight_blocks: list[dict[str, str]] = []
    action_blocks: list[dict[str, str]] = []
    for item in industry_ready.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_insight_blocks = _frontend_ready_section_blocks(
            item.get("key_implication"),
            source="industry_signal_direct",
            anchor_key="event_anchor_terms",
            display_sources={"industry_signal_direct"},
        )
        item_action_blocks = _frontend_ready_section_blocks(
            item.get("suggested_action"),
            source="industry_signal_direct",
            anchor_key="event_anchor_terms",
            display_sources={"industry_signal_direct"},
        )
        insight_blocks.extend(item_insight_blocks)
        action_blocks.extend(item_action_blocks)
        insight_items.extend(
            _labeled_frontend_ready_items(item_insight_blocks, heading="핵심 시사점")
        )
        action_items.extend(_labeled_frontend_ready_items(item_action_blocks, heading="핵심 대응"))
    return {
        "insight": insight_items[:2],
        "action": action_items[:2],
        "insight_blocks": insight_blocks[:2],
        "action_blocks": action_blocks[:2],
        "metadata": metadata if insight_items or action_items else {},
    }


def _frontend_ready_section_blocks(
    value: Any,
    *,
    source: str,
    anchor_key: str,
    display_sources: set[str] | None = None,
) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return []
    block_source = str(value.get("source") or source or "").strip()
    allowed_sources = display_sources or {"llm_direct", "frontend_repair_direct"}
    if block_source not in allowed_sources:
        return []
    evidence_mode = str(value.get("evidence_mode") or "").strip()
    if evidence_mode == "profile_based" and not _list_string(value.get(anchor_key)):
        return []
    sentence = _clean_frontend_ready_text(value.get("sentence"))
    evidence_sentence = _clean_frontend_ready_text(value.get("evidence_sentence"))
    if not sentence or not evidence_sentence:
        return []
    return [
        {
            "main": _ensure_card_sentence(sentence),
            "detail": _ensure_card_sentence(evidence_sentence),
        }
    ]


def _cleanup_public_frontend_implication(frontend: dict[str, Any]) -> dict[str, Any]:
    """Format frontend copy as conclusion + evidence without changing meaning."""

    cleaned = dict(frontend or {})
    for key in (
        "why_important",
        "potential_impact",
        "peer_capability_change",
    ):
        if key in cleaned:
            cleaned[key] = _public_copy_cleanup(cleaned.get(key))
    for key in (
        "key_implications",
        "peer_implications",
        "skax_implications",
        "suggested_actions",
        "response_directions",
        "skax_checkpoints",
    ):
        if key not in cleaned:
            continue
        section = (
            "action"
            if key in {"suggested_actions", "response_directions", "skax_checkpoints"}
            else "insight"
        )
        cleaned[key] = [
            formatted
            for item in _list_string(cleaned.get(key))
            if (
                formatted := _format_public_frontend_item(
                    item if _is_labeled_public_frontend_item(item) else _public_copy_cleanup(item),
                    section=section,
                )
            )
        ]
    if "follow_up_questions" in cleaned:
        cleaned["follow_up_questions"] = [
            _public_copy_cleanup(item) for item in _list_string(cleaned.get("follow_up_questions"))
        ]
    return _with_structured_frontend_blocks(cleaned)


def _with_structured_frontend_blocks(frontend: dict[str, Any]) -> dict[str, Any]:
    payload = dict(frontend or {})
    if not payload.get("key_implication_blocks"):
        payload["key_implication_blocks"] = _structured_blocks_from_labeled_lines(
            payload.get("key_implications") or payload.get("peer_implications")
        )
    if not payload.get("key_implication_items"):
        payload["key_implication_items"] = payload.get("key_implication_blocks") or []
    if not payload.get("response_direction_blocks"):
        payload["response_direction_blocks"] = _structured_blocks_from_labeled_lines(
            payload.get("response_directions")
            or payload.get("suggested_actions")
            or payload.get("skax_checkpoints")
        )
    if not payload.get("suggested_action_items"):
        payload["suggested_action_items"] = payload.get("response_direction_blocks") or []
    if not payload.get("skax_checkpoint_blocks"):
        payload["skax_checkpoint_blocks"] = payload.get("response_direction_blocks") or []
    for key in (
        "key_implication_blocks",
        "key_implication_items",
        "response_direction_blocks",
        "suggested_action_items",
        "skax_checkpoint_blocks",
    ):
        payload[key] = _clean_structured_blocks(payload.get(key))
    return payload


def _clean_structured_blocks(value: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for item in _list_value(value):
        if not isinstance(item, dict):
            continue
        main = _public_copy_cleanup(item.get("main"))
        detail = _public_copy_cleanup(item.get("detail"))
        if main:
            blocks.append({"main": main, "detail": detail})
    return blocks


def _structured_blocks_from_labeled_lines(lines: Any) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for line in _list_string(lines):
        block = _split_main_detail_block(line)
        if block["main"]:
            blocks.append(block)
    return blocks


def _split_main_detail_block(text: str) -> dict[str, str]:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return {"main": "", "detail": ""}

    value = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", value).strip()
    parts = re.split(r"\s*근거\s*/?\s*설명\s*:\s*", value, maxsplit=1)
    main = parts[0].strip() if parts else ""
    detail = parts[1].strip() if len(parts) == 2 else ""

    main = re.sub(r"^핵심\s*(?:시사점|대응)\s*:\s*", "", main).strip()
    detail = re.sub(r"^근거\s*/?\s*설명\s*:\s*", "", detail).strip()
    return {
        "main": _public_copy_cleanup(main),
        "detail": _public_copy_cleanup(detail),
    }


def _format_public_frontend_item(value: Any, *, section: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _is_labeled_public_frontend_item(text):
        return text
    conclusion, evidence = _split_public_conclusion_evidence(text)
    if not conclusion or not evidence:
        return text
    heading = "핵심 대응" if section == "action" else "핵심 시사점"
    return f"{heading}: {conclusion}\n근거/설명: {evidence}"


def _action_candidates_from_strategy(
    *,
    implication: dict[str, Any],
    strategic_root: dict[str, Any],
) -> list[dict[str, Any]]:
    skax = implication.get("skax_implication") or {}
    skax_linkage = strategic_root.get("skax_response_linkage") or {}
    candidates: list[dict[str, Any]] = []
    if isinstance(skax, dict):
        for index, item in enumerate(_list_string(skax.get("recommended_actions"))):
            candidates.append(
                {
                    "text": _clean_card_editorial_text(item),
                    "path": f"skax_implication.recommended_actions[{index}]",
                    "priority": 60 - (index * 5),
                    "semantic_role": _action_semantic_role_for_index(index),
                }
            )
    if isinstance(skax_linkage, dict):
        for field, priority, allow_watch in (
            ("internal_checkpoints", 90, False),
            ("recommended_focus", 86, False),
            ("monitoring_points", 84, True),
        ):
            for index, item in enumerate(_list_string(skax_linkage.get(field))):
                candidates.append(
                    {
                        "text": item,
                        "path": f"skax_response_linkage.{field}[{index}]",
                        "priority": priority - index,
                        "allow_watch_point": allow_watch,
                        "semantic_role": _action_semantic_role_for_linkage_field(field),
                    }
                )
    return candidates


def _literal_summary_lines(
    summary: dict[str, Any],
    *,
    fallback: list[str] | None = None,
) -> list[str]:
    """Return IntegratedIssue summary copy without display rewriting."""
    lines = _list_string(summary.get("display_fact_summary"))
    if not lines:
        lines = _list_string(summary.get("display_summary_lines"))
    if not lines:
        lines = _list_string(summary.get("fact_summary"))
    if not lines:
        lines = _list_string(summary.get("summary_lines"))
    if not lines:
        lines = [
            str(fact.get("fact") or "").strip()
            for fact in _list_dicts(summary.get("consolidated_facts"))
            if str(fact.get("fact") or "").strip()
        ]
    if not lines:
        lines = _list_string(summary.get("one_line_summary"))
    if not lines:
        lines = _list_string(summary.get("integrated_text"))
    if not lines:
        lines = fallback or []

    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = _clean_display_truncated_fragment(_strip_number_prefix(str(line or "")))
        if _looks_like_non_summary_line(text):
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if not text or key in seen:
            continue
        if not _has_numeric_or_period_signal(text) and any(
            _summary_lines_too_similar(text, existing) for existing in out
        ):
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= _SUMMARY_LINE_MAX:
            break
    return out


def _clean_subject_phrase(subject: str, summary: dict[str, Any]) -> str:
    value = re.sub(r"\s+", " ", str(subject or "")).strip(" .,")
    if not value:
        return "현재 이슈"
    company_names = _subject_company_names(summary)
    for company_name in company_names:
        escaped = re.escape(company_name)
        value = re.sub(
            rf"^{escaped}\s*(?:,|·|와|과|및|이|가|은|는)?\s*",
            "",
            value,
        ).strip(" ,.")
    value = re.sub(r"^(이번|해당)\s*", "", value).strip(" ,.")
    value = re.sub(r"^(?:,|·|와|과|및)\s*", "", value).strip(" ,.")
    value = re.sub(
        r"^(?:[가-힣A-Za-z0-9&._-]{2,30}(?:와|과|및)\s+){1,4}함께\s+",
        "",
        value,
    ).strip(" ,.")
    if not value:
        value = "현재 이슈"
    return value[:90] if len(value) > 90 else value


def _profile_phrase_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    return _linkage_priority_phrase(records)


def _profile_reason_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    reasons = _unique_nonempty(
        _first_text(record.get("why_relevant_to_issue"), record.get("reason")) for record in records
    )
    if reasons:
        return reasons[0]
    return ""


def _skax_business_phrase_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    return _linkage_priority_phrase(records, max_terms=2)


def _skax_reason_from_linkage(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    reasons = _unique_nonempty(
        _first_text(record.get("why_relevant_to_issue"), record.get("reason")) for record in records
    )
    if reasons:
        return reasons[0]
    return ""


def _has_specific_skax_linkage(strategic_root: dict[str, Any]) -> bool:
    linkage = strategic_root.get("skax_response_linkage") or {}
    if not isinstance(linkage, dict):
        return False
    if _linkage_area_records(linkage, field="matched_skax_areas"):
        return True
    mode = str(linkage.get("response_mode") or "").strip()
    return mode == "profile_based_action"


def _linkage_priority_phrase(records: list[dict[str, Any]], *, max_terms: int = 3) -> str:
    products = _unique_nonempty(
        term
        for record in records
        for term in _linkage_record_terms(
            record,
            "matched_products_or_services",
            "products_or_services",
            "product_or_service",
            "services",
        )
    )
    if products:
        return _join_context_terms(products[:max_terms])

    capabilities = _unique_nonempty(
        term
        for record in records
        for term in _linkage_record_terms(
            record,
            "matched_capabilities",
            "profile_capability",
            "capability",
            "capabilities",
            "core_capabilities",
        )
    )
    if capabilities:
        return _join_context_terms(capabilities[:max_terms])

    business_areas = _unique_nonempty(
        _first_text(
            record.get("business_area"),
            record.get("profile_area_name"),
            record.get("name"),
        )
        for record in records
        if _record_specificity_level(record) == "business_area"
        and _business_area_record_has_issue_overlap(record)
    )
    if business_areas:
        return _join_context_terms(business_areas[:max_terms])

    business_lines = _unique_nonempty(record.get("business_line") for record in records)
    return _join_context_terms(business_lines[:1])


def _record_specificity_level(record: dict[str, Any]) -> str:
    level = str(record.get("specificity_level") or "").strip().casefold()
    if level:
        return level
    if _linkage_record_terms(record, "matched_products_or_services", "products_or_services"):
        return "product_or_service"
    if _linkage_record_terms(record, "matched_capabilities", "profile_capability", "capability"):
        return "core_capability"
    if _first_text(
        record.get("business_area"),
        record.get("profile_area_name"),
        record.get("name"),
    ):
        return "business_area"
    if record.get("business_line"):
        return "business_line"
    return "profile_context"


def _business_area_record_has_issue_overlap(record: dict[str, Any]) -> bool:
    area = _first_text(
        record.get("business_area"),
        record.get("profile_area_name"),
        record.get("name"),
    )
    if not area:
        return False
    matched_terms = {
        term
        for raw_term in _linkage_record_terms(record, "matched_issue_terms", "matched_terms")
        for term in _display_token_pieces(raw_term)
        if not _low_specificity_display_match_term(term)
    }
    area_terms = {
        term
        for raw_term in re.findall(
            r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9&+·_-]{1,}",
            area,
        )
        for term in _display_token_pieces(raw_term)
        if not _low_specificity_display_match_term(term)
    }
    return bool(area_terms & matched_terms)


def _linkage_has_concrete_profile_detail(records: list[dict[str, Any]]) -> bool:
    for record in records:
        if _linkage_record_terms(
            record,
            "matched_products_or_services",
            "products_or_services",
            "product_or_service",
        ):
            return True
        if _linkage_record_terms(
            record,
            "matched_capabilities",
            "profile_capability",
            "capability",
            "capabilities",
        ):
            return True
    return False


def _profile_specificity_note(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("profile_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_profile_areas")
    if records and not _linkage_has_concrete_profile_detail(records):
        return (
            " 다만 현재 확인되는 사업 정보는 상위 사업명 수준이므로, 세부 "
            "서비스·역량 또는 수행 범위는 후속 단계에서 확인되어야 합니다."
        )
    return ""


def _skax_specificity_note(strategic_root: dict[str, Any]) -> str:
    linkage = strategic_root.get("skax_response_linkage") or {}
    records = _linkage_area_records(linkage, field="matched_skax_areas")
    if records and not _linkage_has_concrete_profile_detail(records):
        return (
            " 다만 현재 확인되는 SK AX의 관련 사업은 상위 사업명 수준이므로, "
            "세부 수행 역량은 별도로 확인해야 합니다."
        )
    return ""


def _summary_fact_text(summary: dict[str, Any]) -> str:
    return " ".join(
        [
            str(summary.get("headline") or ""),
            str(summary.get("main_event") or ""),
            str(summary.get("one_line_summary") or ""),
            " ".join(_list_string(summary.get("fact_summary"))),
            " ".join(_list_string(summary.get("summary_lines"))),
            " ".join(
                _first_text(item.get("fact"))
                for item in _list_value(summary.get("consolidated_facts"))
                if isinstance(item, dict)
            ),
        ]
    )


def _with_particle(text: str, consonant_particle: str, vowel_particle: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    return f"{value}{consonant_particle if _has_final_consonant(value[-1]) else vowel_particle}"


def _ensure_card_sentence(text: str) -> str:
    sentence = _clean_card_editorial_text(text)
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "다.", "요.", "임.")) else f"{sentence}."


def _clean_card_editorial_text(text: str) -> str:
    out = _clean_display_truncated_fragment(text)
    out = _normalize_company_surface_names(out)
    out = re.sub(r"[!！]+$", "", out).strip()
    out = re.sub(r"^함께\s+", "", out)
    out = re.sub(r"\s+함께\s+(?=\d+[조억만천]|\d+장|[A-Z0-9]+ 서비스)", " ", out)
    return out.strip()


def _clean_frontend_ready_text(text: Any) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    out = _normalize_company_surface_names(out)
    out = _strip_public_section_prefixes(out)
    out = _polish_frontend_ready_internal_terms(out)
    out = _normalize_card_news_statement_style(out)
    out = re.sub(r"[!！]+$", "", out).strip()
    return out


def _normalize_company_surface_names(text: str) -> str:
    normalized = text
    for pattern, replacement in _company_surface_replacements():
        normalized = re.sub(pattern, replacement, normalized, flags=re.I)
    return normalized


def _is_summary_repeat_insight(text: str, summary: dict[str, Any]) -> bool:
    line = re.sub(r"\s+", " ", str(text or "")).strip()
    if not line:
        return False
    first_sentence = re.split(r"(?<=[.!?。！？])\s+", line, maxsplit=1)[0]
    summary_lines = _literal_summary_lines(summary)
    for summary_line in summary_lines:
        if _summary_lines_too_similar(first_sentence, summary_line):
            return True
    return False


def _editorial_candidate_has_minimum_grounding(
    text: str,
    *,
    summary: dict[str, Any],
    strategic_root: dict[str, Any],
    section_type: str,
) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    issue_tokens = _grounding_tokens(_summary_fact_text(summary))
    issue_overlap = not issue_tokens or _has_token_overlap(value, issue_tokens)
    if section_type == "insight":
        profile_tokens = _grounding_tokens(_linkage_text(strategic_root.get("profile_linkage")))
        if not issue_overlap and not _has_token_overlap(value, profile_tokens):
            return False
        if profile_tokens and _has_token_overlap(value, profile_tokens):
            return True
        return bool(re.search(r"관찰\s*신호|변화\s*신호|후속|프로필|기존\s*사업", value))
    if section_type == "action":
        if not re.search(r"SK\s*AX|자사", value):
            return False
        skax_tokens = _grounding_tokens(_linkage_text(strategic_root.get("skax_response_linkage")))
        skax_overlap = bool(skax_tokens and _has_token_overlap(value, skax_tokens))
        if not issue_overlap and not skax_overlap:
            return False
        if skax_overlap:
            return True
        return bool(re.search(r"현재\s*입력|보완|모니터링|내부|점검|구분", value))
    return True


def _has_token_overlap(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return True
    return bool(_grounding_tokens(text) & tokens)


def _grounding_tokens(text: str) -> set[str]:
    tokens = _summary_similarity_tokens(str(text or ""))
    return {token for token in tokens if len(token) >= 2}


def _bounded_detail_lines(*values: Any) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in _detail_line_candidates(value):
            if _is_untranslated_english_text(text):
                continue
            key = _detail_line_key(text)
            if text and key and key not in seen and not _is_near_duplicate_detail(text, lines):
                lines.append(text)
                seen.add(key)
            if len(lines) >= _CARD_DETAIL_MAX:
                return lines
    return lines[:_CARD_DETAIL_MAX]


def _actionize_detail_lines(*values: Any) -> list[str]:
    actions: list[str] = []
    for text in _bounded_detail_lines(*values):
        stripped = text.rstrip(".。!?！？ ").strip()
        if not stripped:
            continue
        if _looks_like_action(stripped):
            action = stripped
        elif text.strip().endswith(("?", "？")):
            action = f"{stripped}를 확인합니다"
        else:
            action = _follow_up_action_from_statement(stripped)
        if not action.endswith((".", "。")):
            action += "."
        actions.append(action)
    return actions


def _published_date(articles: list[dict[str, Any]], created_at: str) -> str:
    return _published_datetime(articles, created_at)[:10]


def _subtitle(analysis: dict[str, Any], classification: dict[str, Any]) -> str:
    impact = str(analysis.get("impact_level") or "").strip()
    event_type = _normalize_event_type(classification.get("event_type"))
    return impact.upper() if impact else event_type


def _trust_score(articles: list[dict[str, Any]]) -> float | None:
    scores = [
        score
        for score in (_optional_float(article.get("credibility_score")) for article in articles)
        if score is not None
    ]
    return max(scores) if scores else None


def _validation_pass(summary: dict[str, Any], analysis: dict[str, Any]) -> bool:
    del analysis
    return not _fatal_summary_validation_issues(summary)


def _validation_missing(summary: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    missing: list[str] = _fatal_summary_validation_issues(summary)
    if not bool(summary.get("is_valid_summary", True)):
        missing.append("summary has non-fatal quality warnings")
    if bool(summary.get("fact_extraction_failed")):
        missing.append("fact extraction failed; rule-based candidates used")
    if not bool(analysis.get("is_valid_analysis", True)):
        missing.append("analysis is invalid")
    return missing


def _fatal_summary_validation_issues(summary: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing_indexes = _missing_fact_basis_line_indexes(summary)
    if missing_indexes:
        issues.append(
            "fact_basis missing for summary_line_index: "
            + ", ".join(str(index) for index in missing_indexes)
        )
    if not _summary_line_count_valid(summary):
        issues.append("summary_lines must contain 3~5 lines")
    if any(_looks_like_non_summary_line(line) for line in _literal_summary_lines(summary)):
        issues.append("summary_lines include article headline/question or boilerplate")
    if bool(summary.get("fact_extraction_failed")):
        issues.append("fact extraction failed")
    for warning in _list_string(summary.get("validation_warnings")):
        if (
            "fact_basis source_article_ids가 비어 있음" in warning
            or "수치 근거 부족" in warning
            or "한국어 조사/띄어쓰기 오류" in warning
        ):
            issues.append(warning)
    return list(dict.fromkeys(issues))


def _missing_fact_basis_line_indexes(summary: dict[str, Any]) -> list[int]:
    lines = _plain_summary_lines(summary)
    if not _SUMMARY_LINE_MIN <= len(lines) <= _SUMMARY_LINE_MAX:
        return []
    expected = set(range(1, len(lines) + 1))
    present: set[int] = set()
    for item in summary.get("fact_basis", []) or []:
        if not isinstance(item, dict):
            continue
        index = _optional_int(item.get("summary_line_index", item.get("summary_sentence_index")))
        if index in expected and _list_string(item.get("source_article_ids")):
            present.add(index)
    return sorted(expected - present)


def _summary_line_count_valid(summary: dict[str, Any]) -> bool:
    return _SUMMARY_LINE_MIN <= len(_plain_summary_lines(summary)) <= _SUMMARY_LINE_MAX


def _validation_sc_score(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    trust_score: float | None,
) -> float:
    for value in (
        analysis.get("validation_sc_score"),
        summary.get("validation_sc_score"),
        analysis.get("confidence"),
        summary.get("confidence"),
        trust_score,
    ):
        score = _optional_float(value)
        if score is not None:
            return score
    return 0.0


def _signals(
    *,
    classification: dict[str, Any],
    summary: dict[str, Any],
    source_article_ids: list[int],
    card_text: str,
) -> dict[str, Any]:
    raw = classification.get("signals")
    signals = dict(raw) if isinstance(raw, dict) else {}
    cluster_size = _optional_int(signals.get("cluster_size")) or len(source_article_ids)
    signals["cluster_size"] = cluster_size
    signals["summary_scope"] = "single_article_summary" if cluster_size <= 1 else "cluster_summary"
    if cluster_size <= 1:
        signals["single_article_summary"] = True
    terms = _issue_terms(summary, card_text)
    if terms:
        signals["issue_signature"] = _issue_signature(terms)
        signals["near_duplicate_terms"] = terms
        if cluster_size <= 1:
            signals["near_duplicate_card_candidate"] = True
            signals["near_duplicate_reason"] = (
                "단일 기사 클러스터이며 핵심 제품/서비스 키워드가 있어 유사 카드 병합 검토 필요"
            )
    if summary.get("representative_id"):
        signals["representative_id"] = _optional_int(summary.get("representative_id"))
    return signals


def peer_company_label(company_id: str | None) -> str:
    if not company_id:
        return ""
    local_name = company_name_ko(company_id)
    return local_name if local_name != company_id else global_company_name_ko(company_id)
