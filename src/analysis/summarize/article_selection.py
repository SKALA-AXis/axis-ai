"""summarize article_selection — extracted from facade (move-only)."""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any

from src.analysis.summarize.config import (  # noqa: F401
    _ALLOWED_EVIDENCE_TYPES,
    _ARTICLE_CONTENT_CHARS,
    _ARTICLE_FACT_EXTRACTION_PROMPT,
    _ARTICLE_UI_BOILERPLATE_MARKERS,
    _COMPACT_ARTICLE_CONTENT_CHARS,
    _EVENT_TYPE_VALUES,
    _EVENT_TYPES,
    _FACT_EXTRACTION_BATCH_SIZE,
    _FACT_EXTRACTION_MAX_TOKENS,
    _FACT_EXTRACTION_MODE,
    _FACT_ID_SUMMARY_PROMPT,
    _FACT_TYPES,
    _FULL_TEXT_ARTICLE_LIMIT,
    _INDUSTRY_TREND_ALIASES,
    _INDUSTRY_TREND_COMPANY_ID,
    _KNOWN_COMPANY_ALIASES,
    _LLM_MODEL,
    _MAJORITY_THRESHOLD,
    _MAX_ANALYZED_ARTICLES,
    _MIN_ANALYZED_ARTICLES,
    _MIXED_THRESHOLD,
    _NEAR_DUPLICATE_SIMILARITY,
    _NUMBER_TOKEN_PATTERN,
    _PEER_ALIASES,
    _PROMPT_VERSION,
    _SNIPPET_CANDIDATE_SENTENCES,
    _SNIPPET_DEDUP_SIMILARITY,
    _SNIPPETS_PER_ARTICLE,
    _SUMMARY_LINE_MAX,
    _SUMMARY_LINE_MIN,
    _SUMMARY_MAX_TOKENS,
    _SUMMARY_ROLES,
    _SUPPORTING_ARTICLE_CONTENT_CHARS,
    _USE_FACT_EXTRACTION_LLM,
    _VALIDATION_MAX_TOKENS,
    _env_bool,
    _env_float,
    _env_int,
    _get_llm,
    _llm,
)
from src.analysis.summarize.rule_based_facts import (  # noqa: F401
    _contract_candidate_sentences,
    _contract_detail_facts_from_article,
    _contract_entities,
    _contract_fact_sentence,
    _dedupe_contract_facts,
    _first_sentence_matching,
    _is_article_context_detail_snippet,
    _is_article_relevant_snippet,
    _rule_based_article_fact_notes,
    _rule_based_entities,
    _rule_based_event_type,
    _rule_based_fact_notes_need_llm,
    _rule_based_fact_type_and_role,
    _scope_fact_sentence,
    _select_rule_based_sentences,
    _snippet_score,
)
from src.analysis.summarize.text_utils import (  # noqa: F401
    _append_reason,
    _article_company_alias_mentioned,
    _article_ids,
    _article_numeric_id,
    _article_similarity_tokens,
    _article_target_company_alias_mentioned,
    _article_topic_tokens,
    _articles_text,
    _as_int_list,
    _as_list,
    _body_peer_companies,
    _candidate_peer_companies,
    _chunked,
    _clamp_float,
    _clean_domain_term,
    _clean_json_response,
    _compact,
    _company_display_name,
    _company_list,
    _coverage_info,
    _date_tokens,
    _dedupe_ints,
    _dedupe_keep_order,
    _dedupe_similar_texts,
    _detect_conflict_notes,
    _empty_summary,
    _ensure_sentence,
    _escape_json_string_newlines,
    _event_verbs_in_text,
    _extract_json_object_text,
    _fact_is_off_topic_for_article,
    _fact_key,
    _has_bad_korean_join,
    _has_business_scope_terms,
    _has_detail_preservation_terms,
    _has_uncertain_fact_marker,
    _has_unique_fact_importance,
    _is_article_ui_boilerplate,
    _is_company_neutral_context_detail,
    _is_industry_trend_cluster,
    _is_peer_comparison_issue,
    _join_warnings,
    _matched_companies,
    _metadata,
    _normalize_content,
    _normalize_event_type,
    _normalize_number_token,
    _normalize_string_list,
    _number_token_covered,
    _number_tokens,
    _parse_json,
    _render_prompt,
    _repair_json_text,
    _safe_int,
    _safe_json_loads,
    _split_evidence_sentences,
    _strip_article_ui_boilerplate,
    _summary_mentions_company,
    _summary_metadata,
    _target_company_aliases,
    _text_similarity,
    normalize_korean_spacing,
)


def _select_analysis_articles(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
) -> dict[str, Any]:
    total = len(articles)
    limit = _analysis_article_limit(total)
    if total <= limit:
        return {
            "status": "full_cluster",
            "articles": articles,
            "selected_article_ids": _article_ids(articles),
            "excluded_article_ids": [],
            "majority_article_ids": _article_ids(articles),
            "outlier_article_ids": [],
            "majority_ratio": 1.0 if total else 0.0,
            "analysis_article_limit": limit,
            "warning": "",
        }

    groups = _same_event_title_groups(articles)
    majority = max(groups, key=len) if groups else articles
    majority_ids = set(_article_ids(majority))
    majority_ratio = len(majority) / total if total else 0.0
    outlier_ids = [
        article_id for article_id in _article_ids(articles) if article_id not in majority_ids
    ]
    if majority_ratio < _MIXED_THRESHOLD:
        return {
            "status": "mixed_cluster_no_majority",
            "articles": [],
            "selected_article_ids": [],
            "excluded_article_ids": _article_ids(articles),
            "majority_article_ids": _article_ids(majority),
            "outlier_article_ids": outlier_ids,
            "majority_ratio": round(majority_ratio, 4),
            "analysis_article_limit": limit,
            "warning": (
                "mixed_cluster_no_majority: "
                f"majority_ratio={majority_ratio:.2f} threshold={_MIXED_THRESHOLD:.2f}"
            ),
        }

    selected = _diverse_articles_from_same_event_group(
        articles=majority,
        representative_id=representative_id,
        limit=limit,
    )
    selected_ids = set(_article_ids(selected))
    excluded_ids = [
        article_id for article_id in _article_ids(articles) if article_id not in selected_ids
    ]
    status = "sampled_majority_group"
    warning = (
        f"large_cluster_sampled: analyzed={len(selected)} total={total} "
        f"majority_ratio={majority_ratio:.2f}"
    )
    if majority_ratio < _MAJORITY_THRESHOLD:
        status = "sampled_mixed_majority_group"
        warning = (
            f"mixed_cluster_warning: majority_ratio={majority_ratio:.2f} "
            f"threshold={_MAJORITY_THRESHOLD:.2f}; {warning}"
        )
    return {
        "status": status,
        "articles": selected,
        "selected_article_ids": _article_ids(selected),
        "excluded_article_ids": excluded_ids,
        "majority_article_ids": _article_ids(majority),
        "outlier_article_ids": outlier_ids,
        "majority_ratio": round(majority_ratio, 4),
        "analysis_article_limit": limit,
        "warning": warning,
    }


def _analysis_article_limit(total: int) -> int:
    if total <= 0:
        return 0
    max_limit = max(1, _MAX_ANALYZED_ARTICLES)
    min_limit = min(max_limit, max(1, _MIN_ANALYZED_ARTICLES))
    dynamic = int(math.ceil(math.sqrt(total) * 2.5))
    return min(total, max_limit, max(min_limit, dynamic))


def _same_event_title_groups(articles: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(articles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    features = [_title_group_features(article) for article in articles]
    for left_index in range(len(articles)):
        for right_index in range(left_index + 1, len(articles)):
            if _same_title_event(features[left_index], features[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, article in enumerate(articles):
        groups.setdefault(find(index), []).append(article)
    return sorted(groups.values(), key=len, reverse=True)


def _title_group_features(article: dict[str, Any]) -> dict[str, Any]:
    title = normalize_korean_spacing(article.get("title") or "")
    companies = set(_company_list(article)) | set(_matched_companies(article))
    tokens = _title_event_tokens(title, companies=companies)
    return {
        "tokens": tokens,
        "companies": companies,
        "event_type": _rule_based_event_type([title]),
    }


def _same_title_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["event_type"] != right["event_type"]:
        return False
    left_companies = left["companies"]
    right_companies = right["companies"]
    if left_companies and right_companies and not left_companies & right_companies:
        return False
    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shared = left_tokens & right_tokens
    if len(shared) < 2:
        return False
    coverage = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    return coverage >= 0.45 or (len(shared) >= 3 and jaccard >= 0.22)


def _title_event_tokens(title: str, *, companies: set[str]) -> set[str]:
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9]+", str(title or "").lower())
    company_tokens = _company_alias_title_tokens(companies)
    return {
        token
        for token in (_normalize_title_event_token(token) for token in raw_tokens)
        if _useful_title_event_token(token) and token not in company_tokens
    }


def _normalize_title_event_token(token: str) -> str:
    value = re.sub(r"[^가-힣a-z0-9]", "", str(token or "").lower())
    if re.search(r"[가-힣]", value):
        value = re.sub(r"(으로|로|과|와|은|는|이|가|을|를|에|의)$", "", value)
    return value


def _company_alias_title_tokens(companies: set[str]) -> set[str]:
    tokens: set[str] = set()
    for company_id in companies:
        for alias in _PEER_ALIASES.get(company_id, [company_id]):
            tokens.update(
                _normalize_title_event_token(token)
                for token in re.findall(r"[가-힣A-Za-z0-9]+", str(alias or "").lower())
            )
    return {token for token in tokens if token}


def _useful_title_event_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if token in {
        "및",
        "로",
        "으로",
        "에서",
        "기반",
        "사업",
        "기업",
        "그룹",
        "전사",
        "확대",
        "가속",
        "추진",
    }:
        return False
    return True


def _diverse_articles_from_same_event_group(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        articles,
        key=lambda article: _analysis_article_score(article, representative_id=representative_id),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_titles: list[str] = []
    for article in ranked:
        title = str(article.get("title") or "")
        is_representative = _article_numeric_id(article) == representative_id or bool(
            article.get("is_representative")
        )
        if (
            not is_representative
            and title
            and any(_text_similarity(title, existing) >= 0.82 for existing in selected_titles)
        ):
            continue
        selected.append(article)
        if title:
            selected_titles.append(title)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(ranked)):
        selected_ids = set(_article_ids(selected))
        for article in ranked:
            if _article_numeric_id(article) in selected_ids:
                continue
            selected.append(article)
            if len(selected) >= limit:
                break
    selected_ids_order = set(_article_ids(selected))
    return [article for article in articles if _article_numeric_id(article) in selected_ids_order]


def _analysis_article_score(article: dict[str, Any], *, representative_id: int) -> float:
    title = str(article.get("title") or "")
    score = _article_evidence_score(article, representative_id=representative_id)
    score += min(2.0, 0.5 * len(_number_tokens(title)))
    score += min(1.0, 0.5 * len(_date_tokens(title)))
    score += min(1.5, 0.5 * len(_rule_based_entities([title])))
    if _rule_based_event_type([title]) != "general_update":
        score += 1.0
    return score


def _build_fetch_ids(
    representative_id: int,
    cluster_article_ids: list[int] | None,
    max_cluster_articles: int | None,
) -> list[int]:
    if not cluster_article_ids:
        return [representative_id]

    others = [article_id for article_id in cluster_article_ids if article_id != representative_id]
    ids = _dedupe_ints([representative_id, *others])
    if max_cluster_articles and max_cluster_articles > 0:
        return ids[:max_cluster_articles]
    return ids


def _format_articles(
    articles: list[dict[str, Any]],
    target_companies: list[str],
    representative_id: int,
    compact: bool = False,
) -> str:
    del representative_id, compact
    seen_snippets: list[str] = []
    lines = [
        f"cluster_target_peer_companies: {json.dumps(target_companies, ensure_ascii=False)}",
        "cluster_target_peer_aliases: "
        f"{json.dumps(_target_company_aliases(target_companies), ensure_ascii=False)}",
        "content_policy: 원문 전체 content는 LLM에 넣지 않습니다. "
        "각 기사에서 rule-based로 추출한 evidence_snippets만 사용하고, "
        "중복 문장은 LLM 호출 전에 제거합니다.",
    ]

    for index, article in enumerate(articles, start=1):
        article_id = _article_numeric_id(article)
        metadata = _metadata(article)
        snippets = _article_prompt_snippets(
            article=article,
            target_companies=target_companies,
            seen_snippets=seen_snippets,
        )
        seen_snippets.extend(snippets)
        lines.append(
            "\n".join(
                [
                    f"[{index}] article_id: {article_id}",
                    "article_role: evidence_snippets",
                    f"title: {article.get('title') or ''}",
                    f"source_name: {article.get('source_name') or ''}",
                    f"publisher: {article.get('publisher') or ''}",
                    f"published_at: {article.get('published_at') or ''}",
                    f"company: {json.dumps(_company_list(article), ensure_ascii=False)}",
                    "matched_companies: "
                    f"{json.dumps(_matched_companies(article), ensure_ascii=False)}",
                    f"metadata: {json.dumps(_summary_metadata(metadata), ensure_ascii=False)}",
                    f"evidence_snippets: {json.dumps(snippets, ensure_ascii=False)}",
                ]
            )
        )

    return "\n\n".join(lines)


def _article_prompt_snippets(
    *,
    article: dict[str, Any],
    target_companies: list[str],
    seen_snippets: list[str],
) -> list[str]:
    title = normalize_korean_spacing(article.get("title") or "")
    sentences = _dedupe_keep_order(
        [
            title,
            *_split_evidence_sentences(
                article.get("content") or "",
                limit=_SNIPPET_CANDIDATE_SENTENCES,
            ),
        ]
    )
    sentences = [sentence for sentence in sentences if not _is_article_ui_boilerplate(sentence)]
    relevant_sentences: list[str] = []
    previous_was_relevant = False
    for sentence in sentences:
        is_relevant = _is_article_relevant_snippet(
            sentence,
            article=article,
            target_companies=target_companies,
            title=title,
        )
        if is_relevant or (
            previous_was_relevant
            and _is_article_context_detail_snippet(
                sentence,
                article=article,
                target_companies=target_companies,
            )
        ):
            relevant_sentences.append(sentence)
        previous_was_relevant = is_relevant
    sentences = relevant_sentences
    scored = sorted(
        (
            (_snippet_score(sentence, article=article, target_companies=target_companies), sentence)
            for sentence in sentences
            if sentence
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected: list[str] = []
    local_seen: list[str] = []
    for score, sentence in scored:
        if score <= 0 and selected:
            continue
        if _is_near_duplicate_snippet(sentence, [*seen_snippets, *local_seen]):
            continue
        selected.append(sentence)
        local_seen.append(sentence)
        if len(selected) >= _SNIPPETS_PER_ARTICLE:
            break
    if not selected and title and not _is_near_duplicate_snippet(title, seen_snippets):
        selected.append(title)
    return selected


def _is_near_duplicate_snippet(text: str, selected_texts: list[str]) -> bool:
    if not text or not selected_texts:
        return False
    return any(
        _text_similarity(text, selected) >= _SNIPPET_DEDUP_SIMILARITY for selected in selected_texts
    )


def _full_text_article_ids(
    *,
    articles: list[dict[str, Any]],
    representative_id: int,
) -> set[int]:
    """Select a small evidence set for expensive content analysis.

    News clusters can contain many long articles. The summarizer should still
    know the whole cluster membership, but only a few high-signal articles
    should contribute full body text to the LLM prompt.
    """
    if _FULL_TEXT_ARTICLE_LIMIT <= 0:
        return set()
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    for index, article in enumerate(articles):
        article_id = _article_numeric_id(article)
        if article_id <= 0:
            continue
        score = _article_evidence_score(article, representative_id=representative_id)
        ranked.append((score, -index, article_id, article))
    ranked.sort(reverse=True)
    selected: list[int] = []
    selected_texts: list[str] = []
    for _, _, article_id, article in ranked:
        dedupe_text = _article_dedupe_text(article)
        is_representative = article_id == representative_id or bool(
            article.get("is_representative")
        )
        if (
            not is_representative
            and dedupe_text
            and _is_near_duplicate_article(dedupe_text, selected_texts)
        ):
            continue
        selected.append(article_id)
        if dedupe_text:
            selected_texts.append(dedupe_text)
        if len(selected) >= _FULL_TEXT_ARTICLE_LIMIT:
            break
    return set(selected[:_FULL_TEXT_ARTICLE_LIMIT])


def _article_evidence_score(article: dict[str, Any], *, representative_id: int) -> float:
    article_id = _article_numeric_id(article)
    score = 0.0
    if article_id == representative_id:
        score += 10.0
    if article.get("is_representative"):
        score += 8.0
    score += _clamp_float(article.get("importance_score"), default=0.0) * 3.0
    score += _clamp_float(article.get("relevance_score"), default=0.0) * 2.0
    if _matched_companies(article):
        score += 1.0
    if article.get("content"):
        score += 0.5
    return score


def _article_dedupe_text(article: dict[str, Any]) -> str:
    text = " ".join(
        part
        for part in (
            str(article.get("title") or ""),
            _normalize_content(article.get("content") or "")[:1600],
        )
        if part
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_near_duplicate_article(text: str, selected_texts: list[str]) -> bool:
    if not text or not selected_texts:
        return False
    return any(
        SequenceMatcher(None, text, selected).ratio() >= _NEAR_DUPLICATE_SIMILARITY
        for selected in selected_texts
    )
