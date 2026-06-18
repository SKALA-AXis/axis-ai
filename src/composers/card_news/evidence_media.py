"""card_news evidence_media — extracted from facade (move-only)."""

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
from src.composers.card_news.text_utils1 import (  # noqa: F401
    _action_candidates_from_strategy,
    _actionize_detail_lines,
    _bounded_detail_lines,
    _business_area_record_has_issue_overlap,
    _business_context_title,
    _business_focus_from_summary,
    _clean_card_editorial_text,
    _clean_frontend_ready_text,
    _clean_structured_blocks,
    _clean_subject_phrase,
    _cleanup_public_frontend_implication,
    _contains_key_number_text,
    _editorial_candidate_has_minimum_grounding,
    _ensure_card_sentence,
    _fatal_summary_validation_issues,
    _format_public_frontend_item,
    _frontend_implication,
    _frontend_implication_from_result,
    _frontend_ready_display_items,
    _frontend_ready_section_blocks,
    _grounding_tokens,
    _has_specific_skax_linkage,
    _has_token_overlap,
    _implication,
    _implication_from_result,
    _industry_frontend_ready_display_items,
    _is_summary_repeat_insight,
    _linkage_has_concrete_profile_detail,
    _linkage_priority_phrase,
    _literal_summary_lines,
    _missing_fact_basis_line_indexes,
    _normalize_company_surface_names,
    _plain_summary_lines,
    _profile_phrase_from_linkage,
    _profile_reason_from_linkage,
    _profile_specificity_note,
    _published_date,
    _record_specificity_level,
    _signals,
    _skax_business_phrase_from_linkage,
    _skax_reason_from_linkage,
    _skax_specificity_note,
    _split_main_detail_block,
    _structured_blocks_from_labeled_lines,
    _subtitle,
    _summary_candidate_lines,
    _summary_fact_text,
    _summary_focus_tokens,
    _summary_line_count_valid,
    _summary_lines_too_similar,
    _summary_similarity_tokens,
    _trust_score,
    _validation_missing,
    _validation_pass,
    _validation_sc_score,
    _with_particle,
    _with_structured_frontend_blocks,
    mark_near_duplicate_card_candidates,
    peer_company_label,
)
from src.config.sectors import SECTOR_KEYWORDS
from src.db.article_store import get_articles_by_ids


def _default_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": i + 1,
            "title": a["title"],
            "source_name": a["source_name"],
            "url": a["url"],
            "credibility_score": a.get("credibility_score"),
        }
        for i, a in enumerate(articles)
    ]


def _load_source_articles(summary: dict[str, Any]) -> list[dict[str, Any]]:
    article_ids = [
        article_id
        for article_id in (
            _optional_int(raw_id) for raw_id in _list_string(summary.get("source_article_ids"))
        )
        if article_id is not None
    ]
    return get_articles_by_ids(article_ids) if article_ids else []


def _rich_sources(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, article in enumerate(articles, start=1):
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        if not title and not url:
            continue
        source = {
            "index": index,
            "article_id": _optional_int(article.get("id")),
            "title": title,
            "url": url,
            "archive_url": article.get("archive_url"),
            "source_name": str(article.get("source_name") or article.get("publisher") or ""),
            "published_at": _string_or_none(article.get("published_at")),
            "link_status": "ok",
        }
        image_urls = _article_image_urls(article)
        if image_urls:
            source["image_urls"] = image_urls
        sources.append(source)
    return sources


def _media_assets(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for article_index, article in enumerate(articles):
        article_id = _optional_int(article.get("id"))
        title = str(article.get("title") or "").strip()
        for image_index, url in enumerate(_article_image_urls(article)):
            if url in seen:
                continue
            seen.add(url)
            asset = {
                "id": f"img-{article_id or len(candidates) + 1}-{len(candidates) + 1}",
                "type": "image",
                "url": url,
                "alt": title or "뉴스 본문 이미지",
            }
            score = _image_asset_score(url=url, article=article, image_index=image_index)
            candidates.append((score, -article_index, asset))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [asset for _, _, asset in candidates]


def _image_asset_score(*, url: str, article: dict[str, Any], image_index: int) -> float:
    text = " ".join(
        [
            str(article.get("title") or ""),
            str(article.get("content") or "")[:300],
            url,
        ]
    )
    compact = text.casefold()
    score = 10.0 - image_index * 0.25
    if image_index == 0:
        score += 1.0
    if re.search(r"현장|행사|간담회|협약|mou|체결|센터|데이터센터|공장|회의|대표|부장", text, re.I):
        score += 3.0
    if re.search(r"ai|ax|클라우드|데이터센터|보안|솔루션|플랫폼|로봇|공장", text, re.I):
        score += 1.5
    if re.search(r"주가|차트|목표가|목표주가|거래량|실적표|종목|증권", text):
        score -= 4.0
    if re.search(r"1x1|spacer|blank|placeholder|transparent|pixel", compact):
        score -= 8.0
    if re.search(r"cdn-cgi/image/fit=cover/?$", compact):
        score -= 6.0
    if re.search(r"[?&](?:w|width)=8[0-9]\\b|[?&](?:h|height)=5[0-9]\\b", compact):
        score -= 2.0
    return score


def _media_assets_for_cluster(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility shim for older callers/tests that name cluster-level media."""
    return _media_assets(articles)


def _article_image_urls(article: dict[str, Any]) -> list[str]:
    metadata = _metadata(article)
    candidates: list[str] = []
    for key in (
        "image_url",
        "thumbnail_url",
        "thumbnail",
        "og_image",
        "main_image",
        "image",
    ):
        value = article.get(key) or metadata.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for key in ("image_urls", "images", "media_assets", "visual_images"):
        candidates.extend(_image_urls_from_value(article.get(key)))
        candidates.extend(_image_urls_from_value(metadata.get(key)))
    return _dedupe_keep_order([url for url in candidates if _is_image_reference(url)])


def _image_urls_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                urls.extend(
                    _list_string(
                        item.get("url")
                        or item.get("image_url")
                        or item.get("image_path")
                        or item.get("path")
                    )
                )
        return urls
    if isinstance(value, dict):
        return _list_string(
            value.get("url")
            or value.get("image_url")
            or value.get("image_path")
            or value.get("path")
        )
    return []


def _evidence_chain(
    summary: dict[str, Any],
    analysis: dict[str, Any],
    sources: list[dict[str, Any]],
    source_article_ids: list[int],
    cluster_id: int | None,
    created_at: str,
) -> dict[str, Any]:
    missing = _validation_missing(summary, analysis)
    return {
        "source_links": [
            {
                "article_id": source.get("article_id"),
                "title": source.get("title"),
                "source_name": source.get("source_name"),
                "url": source.get("url"),
            }
            for source in sources
        ],
        "provenance": {
            "raw_article_ids": source_article_ids,
            "cluster_id": cluster_id,
            "llm_model": analysis.get("model") or summary.get("model"),
            "prompt_version": _CARD_PROMPT_VERSION,
            "evidence_version": "v1.0",
            "run_at": created_at,
        },
        "financial_refs": [],
        "mbb_refs": [],
        "evidence_version": "v1.0",
        "fact_basis": _fact_basis(summary),
        "pass": _validation_pass(summary, analysis),
        "missing": missing,
    }


def _source_indexes(sources: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index in (_optional_int(source.get("index")) for source in sources)
        if index is not None
    ]


def _source_article_ids(summary: dict[str, Any], articles: list[dict[str, Any]]) -> list[int]:
    raw_ids = _list_string(summary.get("source_article_ids"))
    article_ids = [
        article_id
        for article_id in (_optional_int(raw_id) for raw_id in raw_ids)
        if article_id is not None
    ]
    if article_ids:
        return article_ids
    return [
        article_id
        for article_id in (_optional_int(article.get("id")) for article in articles)
        if article_id is not None
    ]


def _sector_has_direct_evidence(sector: str, card_text: str) -> bool:
    config = SECTOR_KEYWORDS.get(sector if sector != "biz_area" else "deal")
    if not config:
        return False
    lowered = str(card_text or "").lower()
    return any(str(keyword or "").lower() in lowered for keyword in config.get("keywords", []))


def _fact_basis(summary: dict[str, Any]) -> list[dict[str, Any]]:
    basis: list[dict[str, Any]] = []
    for item in summary.get("fact_basis", []) or []:
        if not isinstance(item, dict):
            continue
        raw_source_ids = _list_string(item.get("source_article_ids"))
        source_ids = [
            article_id
            for article_id in (_optional_int(value) for value in raw_source_ids)
            if article_id is not None
        ]
        evidence_texts = _list_string(item.get("evidence_texts"))
        basis_item: dict[str, Any] = {
            "summary_line_index": _optional_int(
                item.get("summary_line_index", item.get("summary_sentence_index"))
            ),
            "source_article_ids": source_ids,
            "evidence_text": (evidence_texts[0] if evidence_texts else str(item.get("fact") or "")),
            "evidence_type": _normalize_fact_basis_evidence_type(item.get("evidence_type")),
        }
        fact_ids = _list_string(item.get("fact_ids"))
        if fact_ids:
            basis_item["fact_ids"] = fact_ids
        basis.append(basis_item)
    return _dedupe_card_fact_basis(basis)


def _dedupe_card_fact_basis(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for item in items:
        line_index = _optional_int(item.get("summary_line_index"))
        evidence_text = str(item.get("evidence_text") or "").strip()
        source_ids = _list_string(item.get("source_article_ids"))
        if not line_index or not evidence_text:
            continue
        duplicate = False
        for existing in deduped:
            if _optional_int(existing.get("summary_line_index")) != line_index:
                continue
            if _list_string(existing.get("source_article_ids")) != source_ids:
                continue
            if str(existing.get("evidence_text") or "").strip() == evidence_text:
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    return deduped


def _normalize_fact_basis_evidence_type(value: Any) -> str:
    evidence_type = str(value or "").strip()
    if evidence_type in _FACT_BASIS_EVIDENCE_TYPES:
        return evidence_type
    if evidence_type in _EVENT_TO_FACT_BASIS_TYPE:
        return _EVENT_TO_FACT_BASIS_TYPE[evidence_type]
    return "reported_fact"


def _is_image_reference(value: str) -> bool:
    lower = value.lower()
    if lower.startswith(("http://", "https://", "/", "file://")):
        return True
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".avif"))
