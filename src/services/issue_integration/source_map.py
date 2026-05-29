"""Source map builder for citation-ready IntegratedIssue outputs."""

from __future__ import annotations

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

    source_entries: list[tuple[int, int, dict[str, Any], bool]] = []
    for index, raw_id in enumerate(ordered_ids, start=1):
        item = item_by_id.get(raw_id, {})
        source = source_by_id.get(raw_id, {})
        source_entries.append(
            (
                index,
                raw_id,
                _source_entry(raw_article_id=raw_id, item=item, source=source),
                bool(source.get("is_analysis_eligible", True)),
            )
        )

    if not source_entries and input_bundle.items:
        for index, item in enumerate(input_bundle.items, start=1):
            raw_id = _raw_id(item)
            source_entries.append(
                (
                    index,
                    raw_id,
                    _source_entry(raw_article_id=raw_id, item=item, source={}),
                    True,
                )
            )
    sources = [entry for _, _, entry, _ in source_entries]

    raw_article_ids = [raw_id for _, raw_id, _, _ in source_entries]
    eligible_raw_article_ids = [
        raw_id for _, raw_id, _, eligible in source_entries if raw_id > 0 and eligible
    ]
    basis_raw_article_ids = (
        content_digest.get("basis_raw_article_ids") or eligible_raw_article_ids or raw_article_ids
    )
    return {
        "map_version": "source_map_v1",
        "sources": sources,
        "raw_article_id_to_source_index": {
            str(raw_id): index for index, raw_id, _, _ in source_entries if raw_id > 0
        },
        "source_index_to_raw_article_id": {
            str(index): raw_id for index, raw_id, _, _ in source_entries if raw_id > 0
        },
        "raw_article_ids": raw_article_ids,
        "basis_raw_article_ids": basis_raw_article_ids,
        "eligible_raw_article_ids": (
            content_digest.get("eligible_raw_article_ids") or eligible_raw_article_ids
        ),
        "source_count": len(sources),
        "eligible_source_count": sum(1 for *_, eligible in source_entries if eligible),
    }


def _source_entry(
    *,
    raw_article_id: int,
    item: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_name = _first_non_empty(
        source.get("source_name"),
        item.get("source_name"),
        item.get("publisher"),
        item.get("source_type"),
    )
    publisher = _first_non_empty(item.get("publisher"), source.get("publisher"))
    return {
        "id": raw_article_id,
        "title": _first_non_empty(source.get("title"), item.get("title")),
        "source_name": source_name,
        "source_type": _first_non_empty(source.get("source_type"), item.get("source_type")),
        "publisher": publisher,
        "published_at": _first_non_empty(source.get("published_at"), item.get("published_at")),
        "url": _first_non_empty(source.get("url"), item.get("url")),
        "relevance_label": _first_non_empty(
            source.get("relevance_label"), item.get("relevance_label")
        ),
        "relevance_score": source.get("relevance_score", item.get("relevance_score")),
    }


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
