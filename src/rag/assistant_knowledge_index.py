# 작성일: 2026-06-08
# 작성자: 박진
# 변경이력:
#   2026-06-08 박진 — 챗봇 어시스턴트 RAG용 지식 인덱스 신규 작성 및 포맷 정리
"""Assistant knowledge indexing for grounded chat retrieval.

This index complements ``axis_main`` card vectors. It stores compact, user-safe
retrieval chunks from integrated issues, card analysis payloads, and peer
profile snapshots in ``axis_documents``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

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

ASSISTANT_KNOWLEDGE_VERSION = "assistant-knowledge-v1"
ASSISTANT_KNOWLEDGE_SOURCE_TYPES = (
    "integrated_issue",
    "card_news_analysis",
    "peer_profile",
)
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000a551")
_TOP_K_PREFETCH = 50
_DEFAULT_LIMIT_PER_SOURCE = 500
_DEFAULT_BATCH_SIZE = 8


@dataclass(slots=True)
class AssistantKnowledgeRecord:
    source_type: str
    source_id: str
    title: str
    text: str
    summary: str = ""
    peer_id: str | None = None
    event_type: str | None = None
    updated_at: str | None = None

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"{self.source_type}:{self.source_id}"))

    def payload(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title[:500],
            "summary": (self.summary or self.text)[:1000],
            "text": self.text[:3000],
            "peer_id": self.peer_id,
            "company": self.peer_id,
            "event_type": self.event_type,
            "updated_at": self.updated_at,
            "knowledge_version": ASSISTANT_KNOWLEDGE_VERSION,
        }


def index_assistant_knowledge(
    *,
    limit_per_source: int = _DEFAULT_LIMIT_PER_SOURCE,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    source_types: Iterable[str] | None = None,
    exclude_source_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected_source_types = _resolve_source_types(
        source_types=source_types,
        exclude_source_types=exclude_source_types,
    )
    records = [
        record
        for record in iter_assistant_knowledge_records(limit_per_source=limit_per_source)
        if record.source_type in selected_source_types
    ]
    client = get_qdrant_client()
    ensure_collections(client)

    points: list[qm.PointStruct] = []
    indexed = 0
    skipped = 0
    by_source: dict[str, int] = {}
    for record in records:
        text_value = record.text.strip()
        if not text_value:
            skipped += 1
            continue
        vectors = embed_text(text_value, mode="both")
        sparse = vectors["sparse"]
        points.append(
            qm.PointStruct(
                id=record.point_id,
                vector={
                    "dense": vectors["dense"],
                    "sparse": qm.SparseVector(
                        indices=[int(index) for index in sparse["indices"]],
                        values=[float(value) for value in sparse["values"]],
                    ),
                },
                payload=record.payload(),
            )
        )
        indexed += 1
        by_source[record.source_type] = by_source.get(record.source_type, 0) + 1

        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_DOCUMENTS, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION_DOCUMENTS, points=points)

    return {
        "collection": COLLECTION_DOCUMENTS,
        "knowledge_version": ASSISTANT_KNOWLEDGE_VERSION,
        "source_types": sorted(selected_source_types),
        "indexed": indexed,
        "skipped": skipped,
        "by_source": by_source,
    }


def delete_assistant_knowledge(
    *,
    source_types: Iterable[str] | None,
    exclude_source_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Delete assistant-owned Qdrant points for selected source types.

    This intentionally deletes only points stamped with ``ASSISTANT_KNOWLEDGE_VERSION``.
    It is meant for cases like removing temporary card-news mock vectors without
    touching DART chunks or other collections.
    """
    selected_source_types = _resolve_source_types(
        source_types=source_types,
        exclude_source_types=exclude_source_types,
    )
    client = get_qdrant_client()
    existing = {collection.name for collection in client.get_collections().collections}
    if COLLECTION_DOCUMENTS not in existing:
        return {
            "collection": COLLECTION_DOCUMENTS,
            "knowledge_version": ASSISTANT_KNOWLEDGE_VERSION,
            "source_types": sorted(selected_source_types),
            "deleted": {},
            "skipped": "collection_missing",
        }

    from src.rag.qdrant_compat import delete_by_filter

    deleted: dict[str, Any] = {}
    for source_type in sorted(selected_source_types):
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="knowledge_version",
                    match=MatchValue(value=ASSISTANT_KNOWLEDGE_VERSION),
                ),
                FieldCondition(key="source_type", match=MatchValue(value=source_type)),
            ]
        )
        deleted[source_type] = delete_by_filter(
            collection_name=COLLECTION_DOCUMENTS,
            query_filter=query_filter,
            wait=True,
        )
    return {
        "collection": COLLECTION_DOCUMENTS,
        "knowledge_version": ASSISTANT_KNOWLEDGE_VERSION,
        "source_types": sorted(selected_source_types),
        "deleted": deleted,
    }


def search_assistant_knowledge(query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
    vectors = embed_text(query, mode="both")
    client = get_qdrant_client()
    query_filter = Filter(
        must=[
            FieldCondition(
                key="knowledge_version",
                match=MatchValue(value=ASSISTANT_KNOWLEDGE_VERSION),
            )
        ]
    )
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
    except Exception as exc:  # noqa: BLE001 - cluster Qdrant 1.9 needs REST fallback.
        log.warning(
            "assistant knowledge query_points 검색 실패. REST fallback 시도 | error=%s",
            exc,
        )

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
        log.warning("assistant knowledge REST fallback 검색 실패 | error=%s", exc)
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


def iter_assistant_knowledge_records(
    *, limit_per_source: int = _DEFAULT_LIMIT_PER_SOURCE
) -> Iterable[AssistantKnowledgeRecord]:
    yield from _integrated_issue_records(limit_per_source)
    yield from _card_analysis_records(limit_per_source)
    yield from _peer_profile_records(limit_per_source)


def _resolve_source_types(
    *,
    source_types: Iterable[str] | None,
    exclude_source_types: Iterable[str] | None,
) -> set[str]:
    valid = set(ASSISTANT_KNOWLEDGE_SOURCE_TYPES)
    selected = _normalize_source_types(source_types) if source_types else set(valid)
    excluded = _normalize_source_types(exclude_source_types) if exclude_source_types else set()
    resolved = selected - excluded
    if not resolved:
        raise ValueError("assistant knowledge source type selection is empty")
    return resolved


def _normalize_source_types(values: Iterable[str] | None) -> set[str]:
    valid = set(ASSISTANT_KNOWLEDGE_SOURCE_TYPES)
    normalized = {
        item.strip() for value in values or [] for item in str(value).split(",") if item.strip()
    }
    unknown = normalized - valid
    if unknown:
        raise ValueError(
            "unknown assistant knowledge source types: "
            f"{', '.join(sorted(unknown))}; valid={', '.join(ASSISTANT_KNOWLEDGE_SOURCE_TYPES)}"
        )
    return normalized


def _integrated_issue_records(limit: int) -> Iterable[AssistantKnowledgeRecord]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT id::text AS id,
                           headline,
                           one_line_summary,
                           payload,
                           main_company,
                           event_type,
                           updated_at
                      FROM integrated_issues
                     WHERE status = 'active'
                     ORDER BY updated_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001 - optional source in local/dev DBs.
        log.debug("assistant integrated issue records skipped | error=%s", exc)
        return []

    return [
        AssistantKnowledgeRecord(
            source_type="integrated_issue",
            source_id=str(row.get("id") or ""),
            title=str(row.get("headline") or ""),
            summary=str(row.get("one_line_summary") or ""),
            text=_join_text(
                row.get("headline"),
                row.get("one_line_summary"),
                _compact_json(row.get("payload"), max_chars=1800),
            ),
            peer_id=str(row.get("main_company") or "") or None,
            event_type=str(row.get("event_type") or "") or None,
            updated_at=str(row.get("updated_at") or ""),
        )
        for row in rows
    ]


def _card_analysis_records(limit: int) -> Iterable[AssistantKnowledgeRecord]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT id,
                           title,
                           summary_lines,
                           evidence_payload,
                           COALESCE(peer_company_id, company) AS peer_id,
                           event_type,
                           created_at
                      FROM card_news
                     ORDER BY created_at DESC
                     LIMIT :limit
                    """
                    ),
                    {"limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("assistant card analysis records skipped | error=%s", exc)
        return []

    records: list[AssistantKnowledgeRecord] = []
    for row in rows:
        summary = _summary_lines(row.get("summary_lines"))
        evidence = _compact_json(row.get("evidence_payload"), max_chars=2000)
        records.append(
            AssistantKnowledgeRecord(
                source_type="card_news_analysis",
                source_id=str(row.get("id") or ""),
                title=str(row.get("title") or ""),
                summary=summary,
                text=_join_text(row.get("title"), summary, evidence),
                peer_id=str(row.get("peer_id") or "") or None,
                event_type=str(row.get("event_type") or "") or None,
                updated_at=str(row.get("created_at") or ""),
            )
        )
    return records


def _peer_profile_records(limit: int) -> Iterable[AssistantKnowledgeRecord]:
    try:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    text(
                        """
                    SELECT id,
                           name,
                           tier,
                           keywords,
                           core_keywords,
                           profile_snapshot,
                           peer_plus_payload,
                           legacy_payload,
                           COALESCE(
                               profile_snapshot_generated_at,
                               financial_updated_at,
                               created_at
                           ) AS updated_at
                      FROM peer_companies
                     ORDER BY COALESCE(
                         profile_snapshot_generated_at,
                         financial_updated_at,
                         created_at
                     ) DESC
                     LIMIT :limit
                    """
                    ),
                    {"limit": int(limit)},
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("assistant peer profile records skipped | error=%s", exc)
        return []

    records: list[AssistantKnowledgeRecord] = []
    for row in rows:
        profile_text = _join_text(
            row.get("name"),
            _compact_json(row.get("keywords"), max_chars=400),
            _compact_json(row.get("core_keywords"), max_chars=400),
            _compact_json(row.get("profile_snapshot"), max_chars=2000),
            _compact_json(row.get("peer_plus_payload"), max_chars=1600),
            _compact_json(row.get("legacy_payload"), max_chars=900),
        )
        records.append(
            AssistantKnowledgeRecord(
                source_type="peer_profile",
                source_id=str(row.get("id") or ""),
                title=str(row.get("name") or row.get("id") or ""),
                summary=profile_text[:700],
                text=profile_text,
                peer_id=str(row.get("id") or "") or None,
                event_type=str(row.get("tier") or "") or None,
                updated_at=str(row.get("updated_at") or ""),
            )
        )
    return records


def _summary_lines(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")


def _compact_json(value: Any, *, max_chars: int) -> str:
    if value in ({}, [], None, ""):
        return ""
    try:
        text_value = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text_value = str(value)
    return text_value[:max_chars]


def _join_text(*parts: Any) -> str:
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


__all__ = [
    "ASSISTANT_KNOWLEDGE_VERSION",
    "ASSISTANT_KNOWLEDGE_SOURCE_TYPES",
    "AssistantKnowledgeRecord",
    "delete_assistant_knowledge",
    "index_assistant_knowledge",
    "iter_assistant_knowledge_records",
    "search_assistant_knowledge",
]
