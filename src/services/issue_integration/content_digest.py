"""Source-body digest builders for IntegratedIssue.

This module produces extractive content summaries from raw_articles.content and
parser_result sections. It does not create strategic interpretation; it preserves
what the source says in compressed, provenance-linked form.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration.lexicon import FACT_TYPE_INDICATORS
from src.services.issue_integration.policy import DEFAULT_POLICY, IntegrationPolicy

_NUMBER_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)*\s*(?:%|원|조|억|만|천만|백만|달러|USD|KRW|usd|krw|건|명|개|분기|년|월|일)?"
)
_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[가-힣A-Za-z0-9\"'“‘])"
)
_SOURCE_ATTRIBUTION_PATTERN = re.compile(
    r"\s*출처\s*:\s*[^()\n]*(?:\([^)\n]*\))?\s*$",
    re.IGNORECASE,
)
_NEWS_BYLINE_PREFIX_PATTERN = re.compile(r"^\[[^\]]{2,50}(?:기자|=)[^\]]*\]\s*")


def build_content_payload(
    input_bundle: AnalysisInputBundle,
    *,
    policy: IntegrationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Build integrated and per-source content digests."""

    source_by_id = {
        _source_id(source): source
        for source in input_bundle.sources or []
        if _source_id(source) > 0
    }
    source_digests_internal = [
        _source_digest(item, source_by_id=source_by_id, policy=policy)
        for item in input_bundle.items
    ]
    source_digests_internal = [
        digest for digest in source_digests_internal if digest["_has_content"]
    ]
    for source_index, digest in enumerate(source_digests_internal, start=1):
        digest["_source_index"] = source_index
        for point in digest.get("_key_point_payloads", []):
            point["source_index"] = source_index
        for extract in digest.get("_body_extract_payloads", []):
            extract["source_index"] = source_index
    content_digest = _integrated_digest(source_digests_internal, policy=policy)
    return {
        "content_digest": _public_content_digest(content_digest),
        "content_digest_storage": build_content_digest_storage(content_digest),
    }


def _source_digest(
    item: dict[str, Any],
    *,
    source_by_id: dict[int, dict[str, Any]],
    policy: IntegrationPolicy,
) -> dict[str, Any]:
    raw_article_id = _raw_item_id(item)
    source = source_by_id.get(raw_article_id, {})
    units = _content_units(item)
    ranked_units = _rank_units(units, title=str(item.get("title") or ""), policy=policy)
    key_points = [
        _key_point_payload(unit, raw_article_id=raw_article_id, position=index)
        for index, unit in enumerate(
            ranked_units[: policy.source_content_key_point_limit],
            start=1,
        )
    ]
    body_extracts = [
        _body_extract_payload(unit, raw_article_id=raw_article_id, position=index)
        for index, unit in enumerate(ranked_units[: policy.source_body_extract_limit], start=1)
    ]
    title = str(item.get("title") or "").strip()
    summary = _compose_summary(
        [title, *[point["point"] for point in key_points]],
        limit=policy.source_content_digest_chars,
    )
    source_name = item.get("source_name") or source.get("source_name") or item.get("source_type")
    publisher = item.get("publisher")
    full_text = " ".join(unit["text"] for unit in units)
    return {
        "id": raw_article_id,
        "title": title,
        "source": source_name,
        "source_name": source_name,
        "summary": summary,
        "key_points": [point["point"] for point in key_points],
        "body_extracts": _public_body_extracts(body_extracts),
        "url": item.get("url"),
        "published_at": item.get("published_at"),
        "publisher": publisher,
        "relevance_label": source.get("relevance_label") or item.get("relevance_label"),
        "relevance_score": source.get("relevance_score", item.get("relevance_score")),
        "_is_analysis_eligible": bool(source.get("is_analysis_eligible", True)),
        "_has_content": bool(full_text.strip() or title),
        "_content_length_chars": len(full_text),
        "_key_point_payloads": key_points,
        "_body_extract_payloads": body_extracts,
        "_numbers_and_dates": _dedupe_strings(_NUMBER_PATTERN.findall(full_text)),
        "_source_type": item.get("source_type"),
        "_source": source_name,
        "_publisher": publisher,
    }


def _integrated_digest(
    source_digests: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
) -> dict[str, Any]:
    eligible = [digest for digest in source_digests if digest.get("_is_analysis_eligible")]
    basis = eligible or source_digests
    key_points = _dedupe_key_points(
        [
            point
            for digest in basis
            for point in digest.get("_key_point_payloads", [])
            if isinstance(point, dict)
        ]
    )[: policy.integrated_content_key_point_limit]
    body_extracts = _dedupe_body_extracts(
        [
            extract
            for digest in basis
            for extract in digest.get("_body_extract_payloads", [])
            if isinstance(extract, dict)
        ]
    )[: policy.integrated_body_extract_limit]
    summary = _compose_summary(
        [
            *[str(digest.get("title") or "") for digest in basis if digest.get("title")],
            *[point["point"] for point in key_points],
        ],
        limit=policy.integrated_content_digest_chars,
    )
    sections = _build_sections(key_points, body_extracts, policy=policy)
    return {
        "summary": summary,
        "detailed_explanation": _integrated_detailed_explanation(
            sections,
            body_extracts,
            limit=policy.integrated_content_digest_chars,
        ),
        "key_points": key_points,
        "body_extracts": _public_body_extracts(body_extracts),
        "sections": sections,
        "basis_scope": "analysis_eligible_sources" if eligible else "all_sources_for_review",
        "raw_article_ids": [digest["id"] for digest in source_digests],
        "basis_raw_article_ids": [digest["id"] for digest in basis],
        "eligible_raw_article_ids": [digest["id"] for digest in eligible],
        "source_digest_count": len(source_digests),
        "section_count": len(sections),
        "has_content": bool(summary or key_points),
        "compression_method": "extractive_semantic_rank_v1",
    }


def build_content_digest_storage(content_digest: dict[str, Any]) -> dict[str, Any]:
    """Project content_digest into DB-friendly column/value groups."""

    return {
        "content_summary": content_digest.get("summary", ""),
        "content_detailed_explanation": content_digest.get("detailed_explanation", ""),
        "content_has_content": bool(content_digest.get("has_content")),
        "content_compression_method": content_digest.get("compression_method", ""),
        "content_basis_scope": content_digest.get("basis_scope", ""),
        "content_source_count": int(content_digest.get("source_digest_count") or 0),
        "content_section_count": int(content_digest.get("section_count") or 0),
        "content_raw_article_ids": content_digest.get("raw_article_ids", []),
        "content_basis_raw_article_ids": content_digest.get("basis_raw_article_ids", []),
        "content_eligible_raw_article_ids": content_digest.get("eligible_raw_article_ids", []),
        "content_sections": content_digest.get("sections", []),
        "content_body_extracts": content_digest.get("body_extracts", []),
        "content_key_points": content_digest.get("key_points", []),
    }


def _content_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    title = str(item.get("title") or "").strip()
    if title:
        units.append(
            {
                "text": title,
                "role": "title",
                "section_key": "",
                "section_title": "",
                "position": 0,
                "from_parser": False,
            }
        )
    parser_units = _parser_units(item)
    if parser_units:
        units.extend(parser_units)
    content = _clean_source_text(str(item.get("content") or ""))
    for position, sentence in enumerate(_split_sentences(content), start=len(units)):
        units.append(
            {
                "text": sentence,
                "role": "content_sentence",
                "section_key": "",
                "section_title": "",
                "position": position,
                "from_parser": False,
            }
        )
    return _dedupe_units(units)


def _parser_units(item: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _parser_result(item)
    if not parsed:
        return []
    units: list[dict[str, Any]] = []
    for position, chunk in enumerate(_as_dict_list(parsed.get("document_chunks")), start=1):
        text = " ".join(str(chunk.get("text") or "").split())
        if not text:
            continue
        units.append(
            {
                "text": text,
                "role": "parser_chunk",
                "section_key": str(chunk.get("section_key") or ""),
                "section_title": str(chunk.get("section_title") or ""),
                "position": position,
                "from_parser": True,
                "source_chunk_uid": chunk.get("chunk_id"),
            }
        )
    sections = parsed.get("sections")
    if isinstance(sections, dict):
        for position, (section_key, section_value) in enumerate(sections.items(), start=1):
            text = " ".join(str(section_value or "").split())
            if not text:
                continue
            units.append(
                {
                    "text": text,
                    "role": "parser_section",
                    "section_key": str(section_key),
                    "section_title": str(section_key),
                    "position": position,
                    "from_parser": True,
                }
            )
    return units


def _rank_units(
    units: list[dict[str, Any]],
    *,
    title: str,
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    if not units:
        return []
    title_tokens = _tokens(title)
    ranked: list[dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("text") or "")
        normalized = text.lower()
        weights = policy.content_sentence_score_weights
        score = 0.0
        score += weights["materiality_terms"] * sum(
            1 for term in policy.materiality_terms if term.lower() in normalized
        )
        score += weights["numeric_density"] * len(_NUMBER_PATTERN.findall(text))
        score += weights["title_overlap"] * _title_overlap(title_tokens, text)
        if unit.get("from_parser"):
            score += weights["parser_section"]
        if _safe_int(unit.get("position")) <= 2:
            score += weights["position_signal"]
        if any(marker.lower() in normalized for marker in policy.uncertainty_markers):
            score += weights["uncertainty_marker"]
        ranked.append({**unit, "content_rank_score": round(score, 4)})
    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("content_rank_score") or 0.0),
            -_safe_int(item.get("position")),
            len(str(item.get("text") or "")),
        ),
        reverse=True,
    )


def _key_point_payload(
    unit: dict[str, Any],
    *,
    raw_article_id: int,
    position: int,
) -> dict[str, Any]:
    text = str(unit.get("text") or "").strip()
    section = _section_for_unit(unit)
    return {
        "id": _content_ref_id("kp", raw_article_id, position),
        "point": text,
        "raw_article_ids": [raw_article_id] if raw_article_id > 0 else [],
        "section": section,
        "keywords": _keywords_for_text(text, policy_terms=()),
        "numbers_and_dates": _NUMBER_PATTERN.findall(text),
    }


def _body_extract_payload(
    unit: dict[str, Any],
    *,
    raw_article_id: int,
    position: int,
) -> dict[str, Any]:
    text = str(unit.get("text") or "").strip()
    section = _section_for_unit(unit)
    return {
        "id": _content_ref_id("extract", raw_article_id, position),
        "text": text,
        "raw_article_id": raw_article_id,
        "raw_article_ids": [raw_article_id] if raw_article_id > 0 else [],
        "section": section,
        "section_title": unit.get("section_title"),
        "content_role": unit.get("role"),
        "source_chunk_uid": unit.get("source_chunk_uid"),
        "numbers_and_dates": _NUMBER_PATTERN.findall(text),
    }


def _public_body_extracts(extracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": extract.get("id"),
            "text": extract.get("text"),
            "section": extract.get("section"),
            "section_title": extract.get("section_title"),
            "content_role": extract.get("content_role"),
            "numbers_and_dates": extract.get("numbers_and_dates", []),
        }
        for extract in extracts
    ]


def _public_content_digest(content_digest: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in content_digest.items()
        if key not in {"raw_article_ids", "basis_raw_article_ids", "eligible_raw_article_ids"}
    }
    public["key_points"] = [
        {
            key: value
            for key, value in point.items()
            if key not in {"source_index", "raw_article_ids"}
        }
        for point in _as_dict_list(content_digest.get("key_points"))
    ]
    public["body_extracts"] = _public_body_extracts(
        _as_dict_list(content_digest.get("body_extracts"))
    )
    public["sections"] = [
        {
            key: value
            for key, value in section.items()
            if key not in {"key_point_ids", "body_extract_ids", "source_indexes", "raw_article_ids"}
        }
        for section in _as_dict_list(content_digest.get("sections"))
    ]
    return public


def _build_sections(
    key_points: list[dict[str, Any]],
    body_extracts: list[dict[str, Any]],
    *,
    policy: IntegrationPolicy,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in key_points:
        section = str(point.get("section") or "general_fact")
        grouped.setdefault(section, []).append(point)
    extracts_by_section: dict[str, list[dict[str, Any]]] = {}
    for extract in body_extracts:
        section = str(extract.get("section") or "general_fact")
        extracts_by_section.setdefault(section, []).append(extract)
        grouped.setdefault(section, [])
    sections: list[dict[str, Any]] = []
    for section, points in grouped.items():
        extracts = extracts_by_section.get(section, [])
        texts = _dedupe_strings(
            [
                *[str(point.get("point") or "") for point in points],
                *[str(extract.get("text") or "") for extract in extracts],
            ]
        )
        raw_article_ids = _dedupe_ints(
            [
                *[
                    article_id
                    for extract in extracts
                    for article_id in extract.get("raw_article_ids", [])
                ],
                *[
                    article_id
                    for point in points
                    for article_id in point.get("raw_article_ids", [])
                ],
            ]
        )
        sections.append(
            {
                "id": section,
                "section": section,
                "title": _section_title(section),
                "summary": _compose_summary(texts, limit=policy.source_content_digest_chars),
                "detailed_explanation": _section_detailed_explanation(
                    section=section,
                    points=points,
                    extracts=extracts,
                    limit=policy.source_content_digest_chars,
                ),
                "keywords": _keywords_for_texts(
                    texts,
                    policy=policy,
                    limit=policy.content_section_keyword_limit,
                ),
                "key_point_ids": _dedupe_strings(point.get("id") for point in points),
                "body_extract_ids": _dedupe_strings(extract.get("id") for extract in extracts),
                "source_indexes": _dedupe_ints(
                    [
                        *[point.get("source_index") for point in points],
                        *[extract.get("source_index") for extract in extracts],
                    ]
                ),
                "numbers_and_dates": _dedupe_strings(
                    [
                        *[
                            number
                            for extract in extracts
                            for number in extract.get("numbers_and_dates", [])
                        ],
                        *[
                            number
                            for point in points
                            for number in point.get("numbers_and_dates", [])
                        ],
                    ]
                ),
                "raw_article_ids": raw_article_ids,
                "source_count": len(raw_article_ids),
            }
        )
    return sorted(
        sections,
        key=lambda item: (len(item.get("key_point_ids", [])), str(item.get("section") or "")),
        reverse=True,
    )[: policy.content_section_limit]


def _section_detailed_explanation(
    *,
    section: str,
    points: list[dict[str, Any]],
    extracts: list[dict[str, Any]],
    limit: int,
) -> str:
    title = _section_title(section)
    texts = _dedupe_strings(
        [
            *[str(point.get("point") or "") for point in points],
            *[str(extract.get("text") or "") for extract in extracts],
        ]
    )
    explanation = _compose_summary(texts, limit=limit, keep_more=True)
    if not explanation:
        return ""
    return f"{title}: {explanation}"


def _integrated_detailed_explanation(
    sections: list[dict[str, Any]],
    body_extracts: list[dict[str, Any]],
    *,
    limit: int,
) -> str:
    parts = [
        str(section.get("detailed_explanation") or section.get("summary") or "")
        for section in sections
    ]
    if not parts:
        parts = [str(extract.get("text") or "") for extract in body_extracts]
    return _compose_summary(parts, limit=limit, keep_more=True)


def _section_title(section: str) -> str:
    titles = {
        "financial_metric": "수치/재무",
        "risk_fact": "리스크/불확실성",
        "business_signal": "사업 변화",
        "market_fact": "시장/산업 흐름",
        "governance_fact": "조직/지배구조",
        "general_fact": "주요 내용",
        "business": "사업 내용",
        "industry_trend": "산업 동향",
    }
    return titles.get(section, section.replace("_", " ").strip() or "주요 내용")


def _section_for_unit(unit: dict[str, Any]) -> str:
    section_key = str(unit.get("section_key") or "").strip()
    if section_key:
        return section_key
    text = str(unit.get("text") or "")
    scores = {
        fact_type: sum(1 for token in tokens if token.lower() in text.lower())
        for fact_type, tokens in FACT_TYPE_INDICATORS.items()
    }
    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        return "general_fact"
    winners = sorted(fact_type for fact_type, score in scores.items() if score == best_score)
    return winners[0]


def _compose_summary(
    values: list[str],
    *,
    limit: int,
    keep_more: bool = False,
) -> str:
    parts: list[str] = []
    used = 0
    for value in values:
        text = " ".join(str(value or "").split())
        if not text or text in parts:
            continue
        projected = used + len(text) + (1 if parts else 0)
        if parts and projected > limit:
            if keep_more:
                break
            continue
        parts.append(text)
        used = projected
        if not keep_more and len(parts) >= 3:
            break
    summary = " ".join(parts).strip()
    if len(summary) <= limit:
        return summary
    boundary = summary.rfind(" ", 0, limit)
    return summary[: boundary if boundary > 0 else limit].rstrip()


def _split_sentences(content: str) -> list[str]:
    if not content:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(content) if part.strip()]
    return sentences or [content]


def _clean_source_text(content: str) -> str:
    text = " ".join(str(content or "").split())
    if not text:
        return ""
    text = _NEWS_BYLINE_PREFIX_PATTERN.sub("", text)
    text = _SOURCE_ATTRIBUTION_PATTERN.sub("", text)
    return text.strip()


def _content_ref_id(prefix: str, raw_article_id: int, position: int) -> str:
    article_part = str(raw_article_id) if raw_article_id > 0 else "unknown"
    return f"{prefix}:{article_part}:{position}"


def _dedupe_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for unit in units:
        key = re.sub(r"\s+", " ", str(unit.get("text") or "").strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(unit)
    return out


def _dedupe_key_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for point in points:
        text = str(point.get("point") or "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not key:
            continue
        if key in by_key:
            existing = by_key[key]
            existing["raw_article_ids"] = _dedupe_ints(
                [
                    *existing.get("raw_article_ids", []),
                    *point.get("raw_article_ids", []),
                ]
            )
            continue
        by_key[key] = point
        out.append(point)
    return out


def _dedupe_body_extracts(extracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for extract in extracts:
        text = str(extract.get("text") or "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not key:
            continue
        if key in by_key:
            existing = by_key[key]
            existing["raw_article_ids"] = _dedupe_ints(
                [
                    *existing.get("raw_article_ids", []),
                    *extract.get("raw_article_ids", []),
                ]
            )
            continue
        by_key[key] = extract
        out.append(extract)
    return out


def _keywords_for_texts(
    texts: list[str],
    *,
    policy: IntegrationPolicy,
    limit: int,
) -> list[str]:
    corpus = " ".join(texts)
    policy_keywords = [term for term in policy.materiality_terms if term.lower() in corpus.lower()]
    token_counts: dict[str, int] = {}
    for token in _tokens(corpus):
        token_counts[token] = token_counts.get(token, 0) + corpus.lower().count(token)
    frequent_tokens = [
        token for token, _ in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return _dedupe_strings([*policy_keywords, *frequent_tokens])[:limit]


def _keywords_for_text(text: str, *, policy_terms: tuple[str, ...]) -> list[str]:
    policy_keywords = [term for term in policy_terms if term.lower() in text.lower()]
    return _dedupe_strings([*policy_keywords, *sorted(_tokens(text))])[:5]


def _parser_result(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("parser_result")
    if isinstance(value, dict):
        return value
    metadata = _metadata(item)
    if isinstance(metadata.get("parser_result"), dict):
        return metadata["parser_result"]
    return {}


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _title_overlap(title_tokens: set[str], text: str) -> float:
    if not title_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    return len(title_tokens & text_tokens) / len(title_tokens)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", value) if token.strip()}


def _source_id(source: dict[str, Any]) -> int:
    return _safe_int(source.get("raw_article_id") or source.get("article_id"))


def _raw_item_id(item: dict[str, Any]) -> int:
    return _safe_int(
        item.get("raw_article_id")
        or item.get("id")
        or item.get("preprocess_id")
        or item.get("article_id")
    )


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _dedupe_ints(values: Any) -> list[int]:
    out: list[int] = []
    for value in values:
        number = _safe_int(value)
        if number > 0 and number not in out:
            out.append(number)
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_content_digest_storage", "build_content_payload"]
