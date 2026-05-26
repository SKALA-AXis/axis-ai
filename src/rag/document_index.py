"""DART document chunks for agent retrieval in Qdrant."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import models as qm
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    SparseVector,
)

from src.db.qdrant_client import COLLECTION_DOCUMENTS, ensure_collections, get_qdrant_client
from src.rag.embedder import embed_text

log = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000da47")
_DART_VECTOR_SECTION_KEYS = {"company_overview", "business"}
_TOP_K_PREFETCH = 50


def index_dart_chunks(
    *,
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[str]:
    """Upsert DART I/II chunks into Qdrant for later agent retrieval."""
    chunks = [
        chunk
        for chunk in parser_result.get("document_chunks") or []
        if isinstance(chunk, dict)
        and chunk.get("section_key") in _DART_VECTOR_SECTION_KEYS
        and str(chunk.get("text") or "").strip()
    ]
    if not chunks:
        return []

    client = get_qdrant_client()
    ensure_collections(client)

    point_ids: list[str] = []
    points: list[qm.PointStruct] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        point_id = _point_id(article, parser_result, chunk)
        vectors = embed_text(text, mode="both")
        sparse = vectors["sparse"]
        sparse_vec = qm.SparseVector(
            indices=[int(index) for index in sparse["indices"]],
            values=[float(value) for value in sparse["values"]],
        )
        payload = {
            "source_type": "dart",
            "raw_article_id": int(article["id"]) if article.get("id") is not None else None,
            "peer_id": _peer_id(article, parser_result),
            "company": _peer_id(article, parser_result),
            "rcept_no": parser_result.get("rcept_no") or _extra(article).get("rcept_no"),
            "corp_code": parser_result.get("corp_code") or _extra(article).get("corp_code"),
            "corp_name": parser_result.get("corp_name") or _extra(article).get("corp_name"),
            "report_name": parser_result.get("report_name") or article.get("title"),
            "period": parser_result.get("period") or _extra(article).get("period"),
            "period_year": parser_result.get("period_year") or _extra(article).get("period_year"),
            "period_quarter": parser_result.get("period_quarter")
            or _extra(article).get("period_quarter"),
            "published_at": _timestamp(article.get("published_at")),
            "section_key": chunk.get("section_key"),
            "section_title": chunk.get("section_title"),
            "chunk_id": _chunk_uid(article, parser_result, chunk),
            "chunk_index": chunk.get("chunk_index"),
            "topics": chunk.get("topics") or [],
            "matched_keywords": chunk.get("matched_keywords") or [],
            "url": article.get("url") or parser_result.get("url"),
            "text": text,
        }
        points.append(
            qm.PointStruct(
                id=point_id,
                vector={"dense": vectors["dense"], "sparse": sparse_vec},
                payload=payload,
            )
        )
        point_ids.append(point_id)

    client.upsert(collection_name=COLLECTION_DOCUMENTS, points=points)
    log.info("DART chunk vector index 완료 | article=%s chunks=%d", article.get("id"), len(points))
    return point_ids


def search_dart_chunks(
    query: str,
    *,
    top_k: int = 8,
    peer_id: str | None = None,
    period: str | None = None,
    section_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search DART chunks from Qdrant for agent context."""
    try:
        vectors = embed_text(query, mode="both")
    except Exception as exc:
        log.warning("DART chunk 검색 임베딩 실패 | error=%s", exc)
        return []

    conditions: list[qm.Condition] = []
    if peer_id:
        conditions.append(FieldCondition(key="peer_id", match=MatchValue(value=peer_id)))
    if period:
        conditions.append(FieldCondition(key="period", match=MatchValue(value=period)))
    if section_key:
        conditions.append(FieldCondition(key="section_key", match=MatchValue(value=section_key)))
    query_filter = Filter(must=conditions) if conditions else None

    client = get_qdrant_client()
    try:
        results = client.query_points(
            collection_name=COLLECTION_DOCUMENTS,
            prefetch=[
                Prefetch(
                    query=NamedVector(name="dense", vector=vectors["dense"]),  # type: ignore[arg-type]
                    limit=_TOP_K_PREFETCH,
                    filter=query_filter,
                ),
                Prefetch(
                    query=NamedSparseVector(  # type: ignore[arg-type]
                        name="sparse",
                        vector=SparseVector(
                            indices=[int(index) for index in vectors["sparse"]["indices"]],
                            values=[float(value) for value in vectors["sparse"]["values"]],
                        ),
                    ),
                    limit=_TOP_K_PREFETCH,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
    except Exception as exc:
        log.warning("DART chunk 검색 실패 | error=%s", exc)
        return []

    return [
        {
            **(point.payload or {}),
            "score": point.score,
            "point_id": str(point.id),
        }
        for point in results.points
        if point.payload
    ]


def _point_id(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    chunk: dict[str, Any],
) -> str:
    return str(uuid.uuid5(_NAMESPACE, _chunk_uid(article, parser_result, chunk)))


def _chunk_uid(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    chunk: dict[str, Any],
) -> str:
    rcept_no = parser_result.get("rcept_no") or _extra(article).get("rcept_no") or article.get("id")
    section_key = chunk.get("section_key") or "section"
    chunk_index = chunk.get("chunk_index") or chunk.get("chunk_id") or 0
    return f"dart:{rcept_no}:{section_key}:{chunk_index}"


def _peer_id(article: dict[str, Any], parser_result: dict[str, Any]) -> str | None:
    if parser_result.get("peer_id"):
        return str(parser_result.get("peer_id"))
    company = article.get("company")
    if isinstance(company, list) and company:
        return str(company[0])
    return None


def _extra(article: dict[str, Any]) -> dict[str, Any]:
    value = article.get("extra") or article.get("metadata") or {}
    return value if isinstance(value, dict) else {}


def _timestamp(value: Any) -> int:
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp())
    return 0


__all__ = ["index_dart_chunks", "search_dart_chunks"]
