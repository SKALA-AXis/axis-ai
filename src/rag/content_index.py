"""Long-form content index for LLM context retrieval.

This module keeps RDB as the system of record, but mirrors expensive long text
fields into Qdrant with enough payload metadata for exact id-based lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from qdrant_client import models as qm
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)
from sqlalchemy import text

from src.db.postgres import SessionLocal
from src.db.qdrant_client import COLLECTION_DOCUMENTS, ensure_collections, get_qdrant_client
from src.rag.embedder import embed_text

log = logging.getLogger(__name__)

CONTENT_INDEX_VERSION = os.getenv("CONTENT_VDB_VERSION", "content-vdb-v1")

KIND_RAW_ARTICLE_BODY = "raw_article_body"
KIND_INTEGRATED_ISSUE = "integrated_issue"
KIND_ANALYSIS_RESULT = "analysis_result"
KIND_IMPLICATION_RESULT = "implication_result"
KIND_RESPONSE_DIRECTION = "response_direction"

TABLE_RAW_ARTICLES = "raw_articles"
TABLE_INTEGRATED_ISSUES = "integrated_issues"
TABLE_CARD_NEWS = "card_news"

CARD_CONTENT_KINDS = (
    KIND_ANALYSIS_RESULT,
    KIND_IMPLICATION_RESULT,
    KIND_RESPONSE_DIRECTION,
)
_KIND_ORDER = {
    KIND_RAW_ARTICLE_BODY: 0,
    KIND_INTEGRATED_ISSUE: 1,
    KIND_ANALYSIS_RESULT: 2,
    KIND_IMPLICATION_RESULT: 3,
    KIND_RESPONSE_DIRECTION: 4,
}
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000c0de")
_TOP_K_PREFETCH = 50
_DEFAULT_CHUNK_CHARS = max(500, int(os.getenv("CONTENT_VDB_CHUNK_CHARS", "1800")))
_DEFAULT_BATCH_SIZE = max(1, int(os.getenv("CONTENT_VDB_UPSERT_BATCH_SIZE", "8")))
_MAX_SCROLL_LIMIT = max(1, int(os.getenv("CONTENT_VDB_SCROLL_LIMIT", "512")))


@dataclass(slots=True)
class ContentChunk:
    source_table: str
    source_id: str
    content_kind: str
    text: str
    chunk_index: int
    chunk_count: int
    char_start: int
    char_end: int
    title: str = ""
    raw_article_id: int | None = None
    integrated_issue_id: str | None = None
    card_news_id: str | None = None
    peer_id: str | None = None
    source_type: str | None = None
    source_name: str | None = None
    event_type: str | None = None
    url: str | None = None
    published_at: Any = None
    created_at: Any = None
    updated_at: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def chunk_uid(self) -> str:
        return (
            f"{CONTENT_INDEX_VERSION}:{self.source_table}:{self.source_id}:"
            f"{self.content_kind}:{self.chunk_index}:{self.text_hash}"
        )

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, self.chunk_uid))

    def payload(self) -> dict[str, Any]:
        payload = {
            "content_index_version": CONTENT_INDEX_VERSION,
            "source_type": self.source_type,
            "source_table": self.source_table,
            "source_id": self.source_id,
            "content_kind": self.content_kind,
            "title": self.title[:500],
            "raw_article_id": self.raw_article_id,
            "integrated_issue_id": self.integrated_issue_id,
            "card_news_id": self.card_news_id,
            "peer_id": self.peer_id,
            "company": self.peer_id,
            "source_name": self.source_name,
            "event_type": self.event_type,
            "url": self.url,
            "published_at": _json_scalar(self.published_at),
            "created_at": _json_scalar(self.created_at),
            "updated_at": _json_scalar(self.updated_at),
            "chunk_id": self.chunk_uid,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text_hash": self.text_hash,
            "metadata": _json_compatible(self.metadata),
            "text": self.text,
        }
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


@dataclass(slots=True)
class ContentFetchResult:
    text: str
    chunks: list[dict[str, Any]]
    source: str
    content_kinds: list[str] = field(default_factory=list)
    error: str | None = None

    def to_context(self, *, max_chars: int = 2400) -> dict[str, Any]:
        text_value = self.text.strip()
        if max_chars > 0 and len(text_value) > max_chars:
            text_value = text_value[:max_chars] + "\n...TRUNCATED..."
        return {
            "source": self.source,
            "content_kinds": list(self.content_kinds),
            "chunk_count": len(self.chunks),
            "text": text_value,
        }


def index_content_chunks(
    chunks: Sequence[ContentChunk],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Embed and upsert prepared chunks into ``axis_documents``."""

    prepared = [chunk for chunk in chunks if str(chunk.text or "").strip()]
    if not prepared:
        return {"collection": COLLECTION_DOCUMENTS, "indexed": 0, "skipped": len(chunks)}

    client = get_qdrant_client()
    ensure_collections(client)

    if replace_existing:
        seen: set[tuple[str, str, str]] = set()
        for chunk in prepared:
            key = (chunk.source_table, chunk.source_id, chunk.content_kind)
            if key in seen:
                continue
            seen.add(key)
            _delete_existing_chunks(*key)

    indexed = 0
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        vector_batch = _embed_text_batch([chunk.text for chunk in batch])
        points: list[qm.PointStruct] = []
        for chunk, vectors in zip(batch, vector_batch, strict=True):
            sparse = vectors.get("sparse") or {"indices": [], "values": []}
            points.append(
                qm.PointStruct(
                    id=chunk.point_id,
                    vector={
                        "dense": vectors.get("dense") or [],
                        "sparse": qm.SparseVector(
                            indices=[int(index) for index in sparse.get("indices", [])],
                            values=[float(value) for value in sparse.get("values", [])],
                        ),
                    },
                    payload=chunk.payload(),
                )
            )
        indexed += len(points)
        client.upsert(collection_name=COLLECTION_DOCUMENTS, points=points)

    return {
        "collection": COLLECTION_DOCUMENTS,
        "content_index_version": CONTENT_INDEX_VERSION,
        "indexed": indexed,
        "skipped": len(chunks) - indexed,
    }


def _embed_text_batch(texts: Sequence[str]) -> list[dict[str, Any]]:
    if not texts:
        return []
    if getattr(embed_text, "__module__", "") != "src.rag.embedder":
        return [embed_text(text_value, mode="both") for text_value in texts]
    try:
        from src.rag.embedder import EMBED_MAX_LENGTH, get_embedder

        model = get_embedder()
        result = model.encode(
            list(texts),
            return_dense=True,
            return_sparse=True,
            max_length=EMBED_MAX_LENGTH,
        )
        dense_vecs = result["dense_vecs"]
        sparse_vecs = result["lexical_weights"]
        vectors: list[dict[str, Any]] = []
        for index, dense in enumerate(dense_vecs):
            sparse = sparse_vecs[index]
            vectors.append(
                {
                    "dense": dense.tolist() if hasattr(dense, "tolist") else list(dense),
                    "sparse": {
                        "indices": list(sparse.keys()),
                        "values": list(sparse.values()),
                    },
                }
            )
        return vectors
    except Exception as exc:  # noqa: BLE001
        log.debug("content VDB batch embedding fallback | error=%s", exc)
        return [embed_text(text_value, mode="both") for text_value in texts]


def index_raw_article_record(article: Mapping[str, Any]) -> dict[str, Any]:
    chunks = raw_article_chunks(article)
    return index_content_chunks(chunks)


def index_integrated_issue_payload(
    *,
    integrated_issue_id: str | None,
    integrated_issue: Mapping[str, Any],
    input_bundle: Any | None = None,
) -> dict[str, Any]:
    chunks = integrated_issue_chunks(
        integrated_issue_id=integrated_issue_id,
        integrated_issue=integrated_issue,
        input_bundle=input_bundle,
    )
    return index_content_chunks(chunks)


def index_card_analysis_payload(
    *,
    card_news_id: str | None,
    card: Mapping[str, Any],
) -> dict[str, Any]:
    chunks = card_analysis_chunks(card_news_id=card_news_id, card=card)
    return index_content_chunks(chunks)


def index_raw_articles_from_rdb(
    *,
    article_ids: Sequence[int | str] | None = None,
    source_types: Sequence[str] | None = None,
    limit: int = 500,
    offset: int = 0,
    replace_existing: bool = False,
) -> dict[str, Any]:
    rows = _fetch_raw_article_rows(
        article_ids=article_ids,
        source_types=source_types,
        limit=limit,
        offset=offset,
    )
    chunks = [chunk for row in rows for chunk in raw_article_chunks(row)]
    result = (
        index_content_chunks(chunks, replace_existing=replace_existing)
        if chunks
        else _empty_index_result()
    )
    result.update({"source_table": TABLE_RAW_ARTICLES, "rows": len(rows)})
    return result


def index_integrated_issues_from_rdb(
    *,
    integrated_issue_ids: Sequence[str] | None = None,
    limit: int = 500,
    offset: int = 0,
    replace_existing: bool = False,
) -> dict[str, Any]:
    rows = _fetch_integrated_issue_rows(
        integrated_issue_ids=integrated_issue_ids,
        limit=limit,
        offset=offset,
    )
    chunks = [
        chunk
        for row in rows
        for chunk in integrated_issue_chunks(
            integrated_issue_id=str(row.get("id") or ""),
            integrated_issue=row,
        )
    ]
    result = (
        index_content_chunks(chunks, replace_existing=replace_existing)
        if chunks
        else _empty_index_result()
    )
    result.update({"source_table": TABLE_INTEGRATED_ISSUES, "rows": len(rows)})
    return result


def index_card_analysis_from_rdb(
    *,
    card_news_ids: Sequence[str] | None = None,
    limit: int = 500,
    offset: int = 0,
    replace_existing: bool = False,
) -> dict[str, Any]:
    rows = _fetch_card_news_rows(card_news_ids=card_news_ids, limit=limit, offset=offset)
    chunks = [
        chunk
        for row in rows
        for chunk in card_analysis_chunks(card_news_id=str(row.get("id") or ""), card=row)
    ]
    result = (
        index_content_chunks(chunks, replace_existing=replace_existing)
        if chunks
        else _empty_index_result()
    )
    result.update({"source_table": TABLE_CARD_NEWS, "rows": len(rows)})
    return result


def raw_article_chunks(article: Mapping[str, Any]) -> list[ContentChunk]:
    raw_id = _optional_int(article.get("id") or article.get("raw_article_id"))
    text_value = _clean_text(article.get("content"))
    if raw_id is None or not text_value:
        return []
    return _chunks_for_text(
        source_table=TABLE_RAW_ARTICLES,
        source_id=str(raw_id),
        content_kind=KIND_RAW_ARTICLE_BODY,
        text_value=text_value,
        title=_clean_text(article.get("title")),
        raw_article_id=raw_id,
        peer_id=_peer_id(article),
        source_type=_clean_text(article.get("source_type")),
        source_name=_clean_text(article.get("source_name") or article.get("publisher")),
        url=_clean_text(article.get("url")),
        published_at=article.get("published_at"),
        created_at=article.get("created_at") or article.get("collected_at"),
        updated_at=article.get("updated_at"),
        metadata={
            "language": article.get("language"),
            "content_type": article.get("content_type"),
            "processing_status": article.get("processing_status"),
            "crawl_status": article.get("crawl_status"),
            "relevance_label": article.get("relevance_label"),
        },
    )


def integrated_issue_chunks(
    *,
    integrated_issue_id: str | None,
    integrated_issue: Mapping[str, Any],
    input_bundle: Any | None = None,
) -> list[ContentChunk]:
    issue_id = _clean_text(
        integrated_issue_id
        or integrated_issue.get("id")
        or integrated_issue.get("integrated_issue_id")
        or integrated_issue.get("issue_key")
    )
    if not issue_id:
        return []

    payload = _issue_payload(integrated_issue)
    text_value = _join_sections(
        [
            ("headline", payload.get("headline")),
            ("one_line_summary", payload.get("one_line_summary")),
            ("content_summary", payload.get("content_summary")),
            ("content_detailed_explanation", payload.get("content_detailed_explanation")),
            ("integrated_text", payload.get("integrated_text")),
            ("fact_summary", payload.get("fact_summary")),
            ("content_digest", payload.get("content_digest")),
            ("issue_frame", payload.get("issue_frame")),
            ("evidence", payload.get("evidence")),
        ]
    )
    if not text_value:
        return []

    raw_ids = _int_list(
        payload.get("source_article_ids")
        or payload.get("analyzed_source_ids")
        or payload.get("source_ids")
    )
    peer_id = _clean_text(payload.get("main_company"))
    if not peer_id and input_bundle is not None:
        companies = getattr(input_bundle, "companies", None) or []
        peer_id = _clean_text(companies[0] if companies else "")
    return _chunks_for_text(
        source_table=TABLE_INTEGRATED_ISSUES,
        source_id=issue_id,
        content_kind=KIND_INTEGRATED_ISSUE,
        text_value=text_value,
        title=_clean_text(payload.get("headline") or payload.get("one_line_summary")),
        integrated_issue_id=issue_id,
        peer_id=peer_id,
        source_type=_clean_text(payload.get("source_family") or payload.get("issue_source_type")),
        event_type=_clean_text(payload.get("event_type") or payload.get("cluster_event_type")),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        metadata={
            "source_raw_article_ids": raw_ids,
            "cluster_id": payload.get("cluster_id"),
            "confidence": payload.get("confidence"),
        },
    )


def card_analysis_chunks(
    *,
    card_news_id: str | None,
    card: Mapping[str, Any],
) -> list[ContentChunk]:
    card_id = _clean_text(card_news_id or card.get("id") or card.get("card_id"))
    if not card_id:
        return []

    evidence_payload = _json_dict(card.get("evidence_payload"))
    package = _json_dict(evidence_payload.get("analysis_package"))
    implication = (
        _json_dict(package.get("implication"))
        or _json_dict(evidence_payload.get("implication"))
        or _json_dict(card.get("implication"))
    )
    analysis = _json_dict(package.get("analysis")) or _json_dict(evidence_payload.get("analysis"))
    response_direction = _response_direction_payload(implication, evidence_payload, card)
    integrated_issue_id = _clean_text(
        card.get("integrated_issue_id")
        or evidence_payload.get("integrated_issue_id")
        or package.get("integrated_issue_id")
    )
    title = _clean_text(card.get("title"))
    peer_id = _clean_text(card.get("peer_id") or card.get("peer_company_id") or card.get("company"))
    event_type = _clean_text(card.get("event_type"))
    created_at = card.get("created_at")
    updated_at = card.get("updated_at")
    metadata: dict[str, Any] = {
        "source_raw_article_ids": _int_list(card.get("source_raw_article_ids")),
        "importance": card.get("importance"),
        "importance_score": card.get("importance_score"),
    }
    chunks: list[ContentChunk] = []
    sections = [
        (
            KIND_ANALYSIS_RESULT,
            _join_sections(
                [
                    ("title", card.get("title")),
                    ("summary_lines", card.get("summary_lines")),
                    ("analysis", analysis),
                ]
            ),
        ),
        (KIND_IMPLICATION_RESULT, _join_sections([("implication", implication)])),
        (KIND_RESPONSE_DIRECTION, _join_sections([("response_direction", response_direction)])),
    ]
    for content_kind, text_value in sections:
        if not text_value:
            continue
        chunks.extend(
            _chunks_for_text(
                source_table=TABLE_CARD_NEWS,
                source_id=card_id,
                content_kind=content_kind,
                text_value=text_value,
                title=title,
                card_news_id=card_id,
                integrated_issue_id=integrated_issue_id or None,
                peer_id=peer_id,
                source_type="card_news",
                event_type=event_type,
                created_at=created_at,
                updated_at=updated_at,
                metadata=metadata,
            )
        )
    return chunks


def fetch_content_chunks(
    *,
    source_table: str | None = None,
    source_id: str | None = None,
    raw_article_id: int | str | None = None,
    integrated_issue_id: str | None = None,
    card_news_id: str | None = None,
    content_kinds: Sequence[str] | None = None,
    limit: int = _MAX_SCROLL_LIMIT,
) -> list[dict[str, Any]]:
    conditions: list[Any] = [
        FieldCondition(
            key="content_index_version",
            match=MatchValue(value=CONTENT_INDEX_VERSION),
        )
    ]
    if source_table:
        conditions.append(FieldCondition(key="source_table", match=MatchValue(value=source_table)))
    if source_id:
        conditions.append(FieldCondition(key="source_id", match=MatchValue(value=str(source_id))))
    if raw_article_id is not None:
        raw_id = _optional_int(raw_article_id)
        if raw_id is not None:
            conditions.append(FieldCondition(key="raw_article_id", match=MatchValue(value=raw_id)))
    if integrated_issue_id:
        conditions.append(
            FieldCondition(
                key="integrated_issue_id",
                match=MatchValue(value=str(integrated_issue_id)),
            )
        )
    if card_news_id:
        conditions.append(
            FieldCondition(key="card_news_id", match=MatchValue(value=str(card_news_id)))
        )

    content_kind_values = [str(kind) for kind in content_kinds or [] if str(kind or "").strip()]
    if len(content_kind_values) == 1:
        conditions.append(
            FieldCondition(key="content_kind", match=MatchValue(value=content_kind_values[0]))
        )
    elif len(content_kind_values) > 1:
        # Qdrant's MatchAny has had client-version differences, so keep lookup
        # compatible by merging exact per-kind scrolls.
        out: list[dict[str, Any]] = []
        for kind in content_kind_values:
            out.extend(
                fetch_content_chunks(
                    source_table=source_table,
                    source_id=source_id,
                    raw_article_id=raw_article_id,
                    integrated_issue_id=integrated_issue_id,
                    card_news_id=card_news_id,
                    content_kinds=[kind],
                    limit=limit,
                )
            )
        return _sort_chunks(out)[:limit]

    try:
        client = get_qdrant_client()
        hits: list[dict[str, Any]] = []
        offset = None
        page_limit = min(max(1, int(limit)), 128)
        while len(hits) < limit:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_DOCUMENTS,
                scroll_filter=Filter(must=conditions),
                limit=min(page_limit, limit - len(hits)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            payloads = [_point_payload(point) for point in points]
            hits.extend([payload for payload in payloads if payload])
            if not next_offset:
                break
            offset = next_offset
        return _sort_chunks(hits)
    except Exception as exc:  # noqa: BLE001
        log.debug("content VDB exact lookup skipped | error=%s", exc)
        return []


def get_raw_article_body(
    raw_article_id: int | str,
    *,
    fallback_to_rdb: bool = True,
) -> ContentFetchResult:
    chunks = fetch_content_chunks(
        source_table=TABLE_RAW_ARTICLES,
        raw_article_id=raw_article_id,
        content_kinds=[KIND_RAW_ARTICLE_BODY],
    )
    if chunks:
        return ContentFetchResult(
            text=_join_same_kind_chunks(chunks),
            chunks=chunks,
            source="vdb",
            content_kinds=[KIND_RAW_ARTICLE_BODY],
        )
    if not fallback_to_rdb:
        return ContentFetchResult("", [], "missing", [KIND_RAW_ARTICLE_BODY])
    row = _fetch_raw_article_by_id(raw_article_id)
    if not row:
        return ContentFetchResult("", [], "missing", [KIND_RAW_ARTICLE_BODY])
    return ContentFetchResult(
        text=_clean_text(row.get("content")),
        chunks=[],
        source="rdb_fallback",
        content_kinds=[KIND_RAW_ARTICLE_BODY],
    )


def get_integrated_issue_context(
    integrated_issue_id: str,
    *,
    fallback_to_rdb: bool = True,
) -> ContentFetchResult:
    chunks = fetch_content_chunks(
        source_table=TABLE_INTEGRATED_ISSUES,
        integrated_issue_id=integrated_issue_id,
        content_kinds=[KIND_INTEGRATED_ISSUE],
    )
    if chunks:
        return ContentFetchResult(
            text=_join_same_kind_chunks(chunks),
            chunks=chunks,
            source="vdb",
            content_kinds=[KIND_INTEGRATED_ISSUE],
        )
    if not fallback_to_rdb:
        return ContentFetchResult("", [], "missing", [KIND_INTEGRATED_ISSUE])
    row = _fetch_integrated_issue_by_id(integrated_issue_id)
    if not row:
        return ContentFetchResult("", [], "missing", [KIND_INTEGRATED_ISSUE])
    return ContentFetchResult(
        text=_integrated_issue_text_from_row(row),
        chunks=[],
        source="rdb_fallback",
        content_kinds=[KIND_INTEGRATED_ISSUE],
    )


def get_card_analysis_context(
    card_news_id: str,
    *,
    content_kinds: Sequence[str] | None = None,
    fallback_to_rdb: bool = True,
) -> ContentFetchResult:
    kinds = list(content_kinds or CARD_CONTENT_KINDS)
    chunks = fetch_content_chunks(
        source_table=TABLE_CARD_NEWS,
        card_news_id=card_news_id,
        content_kinds=kinds,
    )
    if chunks:
        return ContentFetchResult(
            text=_join_grouped_chunks(chunks),
            chunks=chunks,
            source="vdb",
            content_kinds=_dedupe([str(chunk.get("content_kind")) for chunk in chunks]),
        )
    if not fallback_to_rdb:
        return ContentFetchResult("", [], "missing", kinds)
    row = _fetch_card_news_by_id(card_news_id)
    if not row:
        return ContentFetchResult("", [], "missing", kinds)
    return ContentFetchResult(
        text=_card_text_from_row(row),
        chunks=[],
        source="rdb_fallback",
        content_kinds=kinds,
    )


def search_content_chunks(
    query: str,
    *,
    top_k: int = 8,
    source_table: str | None = None,
    content_kind: str | None = None,
    peer_id: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over content chunks indexed by this module."""

    query_text = str(query or "").strip()
    if not query_text:
        return []
    try:
        vectors = embed_text(query_text, mode="both")
    except Exception as exc:  # noqa: BLE001
        log.debug("content VDB search embedding skipped | error=%s", exc)
        return []

    conditions: list[Any] = [
        FieldCondition(
            key="content_index_version",
            match=MatchValue(value=CONTENT_INDEX_VERSION),
        )
    ]
    if source_table:
        conditions.append(FieldCondition(key="source_table", match=MatchValue(value=source_table)))
    if content_kind:
        conditions.append(FieldCondition(key="content_kind", match=MatchValue(value=content_kind)))
    if peer_id:
        conditions.append(FieldCondition(key="peer_id", match=MatchValue(value=peer_id)))
    query_filter = Filter(must=conditions)

    client = get_qdrant_client()
    try:
        results = client.query_points(
            collection_name=COLLECTION_DOCUMENTS,
            prefetch=[
                Prefetch(
                    query=vectors["dense"],
                    using="dense",
                    limit=_TOP_K_PREFETCH,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=[int(index) for index in vectors["sparse"]["indices"]],
                        values=[float(value) for value in vectors["sparse"]["values"]],
                    ),
                    using="sparse",
                    limit=_TOP_K_PREFETCH,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return [
            {
                **(point.payload or {}),
                "score": point.score,
                "point_id": str(point.id),
            }
            for point in results.points
            if point.payload
        ]
    except Exception as exc:  # noqa: BLE001
        log.debug("content VDB query_points search fallback | error=%s", exc)

    try:
        from src.rag.qdrant_compat import legacy_rrf_search

        hits = legacy_rrf_search(
            collection_name=COLLECTION_DOCUMENTS,
            dense_vector=vectors["dense"],
            sparse_vector=vectors["sparse"],
            limit=top_k,
            prefetch_limit=_TOP_K_PREFETCH,
            query_filter=query_filter,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("content VDB legacy search skipped | error=%s", exc)
        return []
    return [
        {
            **(hit.get("payload") or {}),
            "score": hit.get("score"),
            "point_id": str(hit.get("id") or ""),
        }
        for hit in hits
        if hit.get("payload")
    ]


def attach_vdb_context_to_row(row: dict[str, Any], *, max_chars: int = 2400) -> dict[str, Any]:
    """Attach VDB context to an analysis/card row without removing RDB fields."""

    evidence_payload = _json_dict(row.get("evidence_payload"))
    context: dict[str, Any] = _json_dict(evidence_payload.get("vdb_context"))

    issue_id = _clean_text(row.get("integrated_issue_id") or row.get("ii_id"))
    if issue_id and "integrated_issue" not in context:
        issue_result = get_integrated_issue_context(issue_id, fallback_to_rdb=False)
        if issue_result.text:
            context["integrated_issue"] = issue_result.to_context(max_chars=max_chars)

    card_id = _clean_text(row.get("card_id") or row.get("id"))
    if card_id and "card_analysis" not in context:
        card_result = get_card_analysis_context(card_id, fallback_to_rdb=False)
        if card_result.text:
            context["card_analysis"] = card_result.to_context(max_chars=max_chars)

    if context:
        evidence_payload["vdb_context"] = context
        row["evidence_payload"] = evidence_payload
    return row


def attach_vdb_contexts_to_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_chars: int = 2400,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(attach_vdb_context_to_row(row, max_chars=max_chars))
        except Exception as exc:  # noqa: BLE001
            log.debug("content VDB row hydration skipped | error=%s", exc)
            out.append(row)
    return out


def _chunks_for_text(
    *,
    source_table: str,
    source_id: str,
    content_kind: str,
    text_value: str,
    title: str = "",
    raw_article_id: int | None = None,
    integrated_issue_id: str | None = None,
    card_news_id: str | None = None,
    peer_id: str | None = None,
    source_type: str | None = None,
    source_name: str | None = None,
    event_type: str | None = None,
    url: str | None = None,
    published_at: Any = None,
    created_at: Any = None,
    updated_at: Any = None,
    metadata: dict[str, Any] | None = None,
) -> list[ContentChunk]:
    spans = _chunk_text(text_value)
    chunks: list[ContentChunk] = []
    for index, (chunk_text, start, end) in enumerate(spans):
        if not chunk_text.strip():
            continue
        chunks.append(
            ContentChunk(
                source_table=source_table,
                source_id=str(source_id),
                content_kind=content_kind,
                text=chunk_text,
                chunk_index=index,
                chunk_count=len(spans),
                char_start=start,
                char_end=end,
                title=title,
                raw_article_id=raw_article_id,
                integrated_issue_id=integrated_issue_id,
                card_news_id=card_news_id,
                peer_id=peer_id,
                source_type=source_type,
                source_name=source_name,
                event_type=event_type,
                url=url,
                published_at=published_at,
                created_at=created_at,
                updated_at=updated_at,
                metadata=metadata or {},
            )
        )
    return chunks


def _chunk_text(
    text_value: str,
    *,
    max_chars: int = _DEFAULT_CHUNK_CHARS,
) -> list[tuple[str, int, int]]:
    text_value = _clean_text(text_value)
    if not text_value:
        return []
    if len(text_value) <= max_chars:
        return [(text_value, 0, len(text_value))]

    spans: list[tuple[str, int, int]] = []
    start = 0
    min_break = int(max_chars * 0.55)
    while start < len(text_value):
        end = min(start + max_chars, len(text_value))
        if end < len(text_value):
            window = text_value[start:end]
            break_candidates = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("다. "),
            ]
            break_at = max(break_candidates)
            if break_at >= min_break:
                end = start + break_at + 1
        if end <= start:
            end = min(start + max_chars, len(text_value))
        spans.append((text_value[start:end], start, end))
        start = end
    return spans


def _delete_existing_chunks(source_table: str, source_id: str, content_kind: str) -> None:
    try:
        from src.rag.qdrant_compat import delete_by_filter

        delete_by_filter(
            collection_name=COLLECTION_DOCUMENTS,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_index_version",
                        match=MatchValue(value=CONTENT_INDEX_VERSION),
                    ),
                    FieldCondition(key="source_table", match=MatchValue(value=source_table)),
                    FieldCondition(key="source_id", match=MatchValue(value=str(source_id))),
                    FieldCondition(key="content_kind", match=MatchValue(value=content_kind)),
                ]
            ),
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "content VDB stale chunk delete skipped | table=%s id=%s kind=%s error=%s",
            source_table,
            source_id,
            content_kind,
            exc,
        )


def _fetch_raw_article_rows(
    *,
    article_ids: Sequence[int | str] | None,
    source_types: Sequence[str] | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    conditions = [
        "content IS NOT NULL",
        "length(content) > 0",
        "COALESCE(UPPER(processing_status), '') NOT IN ('SKIPPED', 'FILTERED', 'FAILED')",
        "COALESCE(LOWER(crawl_status), '') <> 'failed'",
        "COALESCE(LOWER(relevance_label), '') <> 'irrelevant'",
    ]
    params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    id_values = [_optional_int(value) for value in article_ids or []]
    id_values = [value for value in id_values if value is not None]
    if id_values:
        placeholders = ",".join(f":id_{index}" for index in range(len(id_values)))
        conditions.append(f"id IN ({placeholders})")
        params.update({f"id_{index}": value for index, value in enumerate(id_values)})
    source_type_values = [_clean_text(value) for value in source_types or [] if _clean_text(value)]
    if source_type_values:
        placeholders = ",".join(f":source_type_{index}" for index in range(len(source_type_values)))
        conditions.append(f"source_type IN ({placeholders})")
        params.update(
            {f"source_type_{index}": value for index, value in enumerate(source_type_values)}
        )
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                    SELECT id, source_type, source_name, publisher, title, content, url,
                           published_at, collected_at, created_at, created_at AS updated_at,
                           company, language, content_type, crawl_status,
                           processing_status, relevance_label, metadata
                      FROM raw_articles
                     WHERE {" AND ".join(conditions)}
                     ORDER BY id
                     LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _fetch_integrated_issue_rows(
    *,
    integrated_issue_ids: Sequence[str] | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    conditions = ["TRUE"]
    params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    id_values = [_clean_text(value) for value in integrated_issue_ids or [] if _clean_text(value)]
    if id_values:
        placeholders = ",".join(f"CAST(:id_{index} AS uuid)" for index in range(len(id_values)))
        conditions.append(f"id IN ({placeholders})")
        params.update({f"id_{index}": value for index, value in enumerate(id_values)})
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                    SELECT id::text AS id, headline, one_line_summary, main_company,
                           event_type, source_family, confidence, analyzed_source_ids,
                           source_ids, content_summary, content_detailed_explanation,
                           content_digest, issue_frame, evidence, sources, payload,
                           created_at, updated_at
                      FROM integrated_issues
                     WHERE {" AND ".join(conditions)}
                     ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                     LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _fetch_card_news_rows(
    *,
    card_news_ids: Sequence[str] | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    conditions = ["TRUE"]
    params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    id_values = [_clean_text(value) for value in card_news_ids or [] if _clean_text(value)]
    if id_values:
        placeholders = ",".join(f":id_{index}" for index in range(len(id_values)))
        conditions.append(f"id IN ({placeholders})")
        params.update({f"id_{index}": value for index, value in enumerate(id_values)})
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                    SELECT id, title, summary_lines, event_type, importance,
                           importance_score, implication, evidence_payload,
                           source_raw_article_ids, sources, integrated_issue_id::text,
                           COALESCE(peer_company_id, company) AS peer_id,
                           created_at, created_at AS updated_at
                      FROM card_news
                     WHERE {" AND ".join(conditions)}
                     ORDER BY created_at DESC NULLS LAST
                     LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _fetch_raw_article_by_id(raw_article_id: int | str) -> dict[str, Any] | None:
    raw_id = _optional_int(raw_article_id)
    if raw_id is None:
        return None
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text("SELECT id, title, content FROM raw_articles WHERE id = :id"),
                    {"id": raw_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        log.debug("raw article RDB fallback lookup failed | id=%s error=%s", raw_article_id, exc)
        return None


def _fetch_integrated_issue_by_id(integrated_issue_id: str) -> dict[str, Any] | None:
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                        SELECT id::text AS id, headline, one_line_summary,
                               content_summary, content_detailed_explanation,
                               content_digest, issue_frame, evidence, sources, payload
                          FROM integrated_issues
                         WHERE id = CAST(:id AS uuid)
                        """
                    ),
                    {"id": integrated_issue_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "integrated issue RDB fallback lookup failed | id=%s error=%s",
            integrated_issue_id,
            exc,
        )
        return None


def _fetch_card_news_by_id(card_news_id: str) -> dict[str, Any] | None:
    try:
        with SessionLocal() as db:
            row = (
                db.execute(
                    text(
                        """
                        SELECT id, title, summary_lines, event_type, implication,
                               evidence_payload, sources, integrated_issue_id::text,
                               source_raw_article_ids, created_at
                          FROM card_news
                         WHERE id = :id
                        """
                    ),
                    {"id": card_news_id},
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        log.debug("card_news RDB fallback lookup failed | id=%s error=%s", card_news_id, exc)
        return None


def _issue_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_dict(value.get("payload"))
    if payload:
        merged = dict(payload)
        for key, item in value.items():
            if key not in merged and item not in (None, "", [], {}):
                merged[key] = item
        return merged
    return dict(value)


def _integrated_issue_text_from_row(row: Mapping[str, Any]) -> str:
    payload = _issue_payload(row)
    return _join_sections(
        [
            ("headline", payload.get("headline")),
            ("one_line_summary", payload.get("one_line_summary")),
            ("content_summary", payload.get("content_summary")),
            ("content_detailed_explanation", payload.get("content_detailed_explanation")),
            ("content_digest", payload.get("content_digest")),
            ("issue_frame", payload.get("issue_frame")),
            ("evidence", payload.get("evidence")),
        ]
    )


def _card_text_from_row(row: Mapping[str, Any]) -> str:
    evidence_payload = _json_dict(row.get("evidence_payload"))
    package = _json_dict(evidence_payload.get("analysis_package"))
    implication = (
        _json_dict(package.get("implication"))
        or _json_dict(evidence_payload.get("implication"))
        or _json_dict(row.get("implication"))
    )
    return _join_sections(
        [
            ("title", row.get("title")),
            ("summary_lines", row.get("summary_lines")),
            ("analysis", package.get("analysis") or evidence_payload.get("analysis")),
            ("implication", implication),
            ("response_direction", _response_direction_payload(implication, evidence_payload, row)),
        ]
    )


def _response_direction_payload(
    implication: Mapping[str, Any],
    evidence_payload: Mapping[str, Any],
    card: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for source in (implication, evidence_payload, card):
        for key in (
            "frontend",
            "frontend_ready",
            "industry_frontend_ready",
            "recommended_actions",
            "response_directions",
            "suggested_actions",
            "skax_checkpoints",
            "response_direction_blocks",
        ):
            value = source.get(key) if isinstance(source, Mapping) else None
            if value not in (None, "", [], {}):
                candidates[key] = value
    return candidates


def _join_sections(sections: Sequence[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for label, value in sections:
        text_value = _section_text(value)
        if text_value:
            parts.append(f"{label}:\n{text_value}")
    return "\n\n".join(parts).strip()


def _section_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(_json_compatible(value), ensure_ascii=False, default=str)
    except TypeError:
        return str(value).strip()


def _join_same_kind_chunks(chunks: Sequence[Mapping[str, Any]]) -> str:
    return "".join(str(chunk.get("text") or "") for chunk in _sort_chunks(chunks))


def _join_grouped_chunks(chunks: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.get("content_kind") or ""), []).append(chunk)
    parts: list[str] = []
    for kind in sorted(grouped, key=lambda value: _KIND_ORDER.get(value, 999)):
        text_value = _join_same_kind_chunks(grouped[kind]).strip()
        if text_value:
            parts.append(text_value)
    return "\n\n".join(parts)


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, Mapping):
        payload = point.get("payload")
    else:
        payload = getattr(point, "payload", None)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sort_chunks(chunks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(chunk) for chunk in chunks],
        key=lambda chunk: (
            _KIND_ORDER.get(str(chunk.get("content_kind") or ""), 999),
            _safe_int(chunk.get("chunk_index")),
            _safe_int(chunk.get("char_start")),
        ),
    )


def _empty_index_result() -> dict[str, Any]:
    return {
        "collection": COLLECTION_DOCUMENTS,
        "content_index_version": CONTENT_INDEX_VERSION,
        "indexed": 0,
        "skipped": 0,
    }


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _peer_id(article: Mapping[str, Any]) -> str | None:
    for key in ("peer_id", "peer_company_id"):
        value = _clean_text(article.get(key))
        if value:
            return value
    company = article.get("company")
    if isinstance(company, list) and company:
        return _clean_text(company[0]) or None
    if isinstance(company, str) and company.strip().startswith("["):
        try:
            parsed = json.loads(company)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and parsed:
            return _clean_text(parsed[0]) or None
    value = _clean_text(company)
    return value or None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    out: list[int] = []
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None:
            out.append(parsed)
    return _dedupe(out)


def _dedupe(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_int(value: Any) -> int:
    parsed = _optional_int(value)
    return parsed if parsed is not None else 0


__all__ = [
    "CARD_CONTENT_KINDS",
    "CONTENT_INDEX_VERSION",
    "KIND_ANALYSIS_RESULT",
    "KIND_IMPLICATION_RESULT",
    "KIND_INTEGRATED_ISSUE",
    "KIND_RAW_ARTICLE_BODY",
    "KIND_RESPONSE_DIRECTION",
    "attach_vdb_context_to_row",
    "attach_vdb_contexts_to_rows",
    "card_analysis_chunks",
    "fetch_content_chunks",
    "get_card_analysis_context",
    "get_integrated_issue_context",
    "get_raw_article_body",
    "index_card_analysis_from_rdb",
    "index_card_analysis_payload",
    "index_content_chunks",
    "index_integrated_issue_payload",
    "index_integrated_issues_from_rdb",
    "index_raw_article_record",
    "index_raw_articles_from_rdb",
    "integrated_issue_chunks",
    "raw_article_chunks",
    "search_content_chunks",
]
