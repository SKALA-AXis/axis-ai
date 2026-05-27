"""Source map builder for citation-ready IntegratedIssue outputs."""

from __future__ import annotations

import json
from typing import Any

from src.analysis.models import AnalysisInputBundle


def build_source_map(
    input_bundle: AnalysisInputBundle,
    *,
    content_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable raw_article_id -> source_index map.

    The map follows input item order first so it aligns with content_digest.sources.
    Any source rows not present in items are appended afterward.
    """

    content_digest = content_digest or {}
    item_by_id = {_raw_id(item): item for item in input_bundle.items or [] if _raw_id(item) > 0}
    source_by_id = {
        _source_id(source): source
        for source in input_bundle.sources or []
        if _source_id(source) > 0
    }
    ordered_ids: list[int] = []
    for item in input_bundle.items or []:
        raw_id = _raw_id(item)
        if raw_id > 0 and raw_id not in ordered_ids:
            ordered_ids.append(raw_id)
    for source in input_bundle.sources or []:
        raw_id = _source_id(source)
        if raw_id > 0 and raw_id not in ordered_ids:
            ordered_ids.append(raw_id)

    sources: list[dict[str, Any]] = []
    for index, raw_id in enumerate(ordered_ids, start=1):
        item = item_by_id.get(raw_id, {})
        source = source_by_id.get(raw_id, {})
        sources.append(_source_entry(index=index, raw_article_id=raw_id, item=item, source=source))

    if not sources and input_bundle.items:
        for index, item in enumerate(input_bundle.items, start=1):
            raw_id = _raw_id(item)
            sources.append(_source_entry(index=index, raw_article_id=raw_id, item=item, source={}))

    return {
        "map_version": "source_map_v1",
        "sources": sources,
        "raw_article_id_to_source_index": {
            str(source["raw_article_id"]): source["source_index"]
            for source in sources
            if _safe_int(source.get("raw_article_id")) > 0
        },
        "source_index_to_raw_article_id": {
            str(source["source_index"]): source["raw_article_id"]
            for source in sources
            if _safe_int(source.get("raw_article_id")) > 0
        },
        "raw_article_ids": [source["raw_article_id"] for source in sources],
        "basis_raw_article_ids": content_digest.get("basis_raw_article_ids", []),
        "eligible_raw_article_ids": content_digest.get("eligible_raw_article_ids", []),
        "source_count": len(sources),
        "eligible_source_count": sum(1 for source in sources if source.get("is_analysis_eligible")),
    }


def _source_entry(
    *,
    index: int,
    raw_article_id: int,
    item: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    metadata = _metadata(item)
    link_check = _link_check(item=item, source=source)
    source_name = _first_non_empty(
        source.get("source_name"),
        item.get("source_name"),
        item.get("publisher"),
        item.get("source_type"),
    )
    publisher = _first_non_empty(item.get("publisher"), source.get("publisher"))
    return {
        "source_index": index,
        "raw_article_id": raw_article_id,
        "article_id": _safe_int(source.get("article_id") or raw_article_id),
        "title": _first_non_empty(source.get("title"), item.get("title")),
        "url": _first_non_empty(source.get("url"), item.get("url")),
        "source_name": source_name,
        "publisher": publisher,
        "published_at": _first_non_empty(source.get("published_at"), item.get("published_at")),
        "collected_at": _first_non_empty(source.get("collected_at"), item.get("collected_at")),
        "source_type": _first_non_empty(source.get("source_type"), item.get("source_type")),
        "content_type": _first_non_empty(source.get("content_type"), item.get("content_type")),
        "processing_status": _first_non_empty(
            source.get("processing_status"), item.get("processing_status")
        ),
        "crawl_status": _first_non_empty(source.get("crawl_status"), item.get("crawl_status")),
        "error_message": _first_non_empty(source.get("error_message"), item.get("error_message")),
        "relevance_label": _first_non_empty(
            source.get("relevance_label"), item.get("relevance_label")
        ),
        "relevance_score": source.get("relevance_score", item.get("relevance_score")),
        "importance_score": source.get("importance_score", item.get("importance_score")),
        "is_analysis_eligible": bool(source.get("is_analysis_eligible", True)),
        "link_status": _first_non_empty(link_check.get("status"), "unknown"),
        "final_url": _first_non_empty(link_check.get("final_url"), metadata.get("final_url")),
    }


def _link_check(*, item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    value = source.get("link_check")
    if isinstance(value, dict):
        return value
    value = item.get("link_check")
    if isinstance(value, dict):
        return value
    metadata = _metadata(item)
    value = metadata.get("link_check")
    return value if isinstance(value, dict) else {}


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _source_id(source: dict[str, Any]) -> int:
    return _safe_int(source.get("raw_article_id") or source.get("article_id"))


def _raw_id(item: dict[str, Any]) -> int:
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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_source_map"]
