"""Qdrant RRF 하이브리드 검색 — Dense + Sparse 융합"""

import logging
from typing import Optional

from qdrant_client.models import (
    Fusion,
    FusionQuery,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    SparseVector,
)

from src.db.qdrant_client import COLLECTION_MAIN, get_qdrant_client

log = logging.getLogger(__name__)

RRF_K = 60
TOP_K_PREFETCH = 50


def hybrid_search(
    query: str,
    top_k: int = 10,
    peer_id: Optional[str] = None,
    event_type: Optional[str] = None,
) -> list[dict]:
    """BGE-M3 Dense+Sparse RRF 하이브리드 검색.

    Args:
        query: 검색 쿼리.
        top_k: 반환할 결과 수.
        peer_id: Peer사 필터 (없으면 전체).
        event_type: 이벤트 타입 필터.

    Returns:
        검색 결과 목록 (rdb_id, title, summary 등 포함).
    """
    try:
        from src.rag.embedder import embed_text

        vectors = embed_text(query, mode="both")
    except Exception as e:
        log.warning("임베딩 실패. BM25 폴백: %s", e)
        return []

    client = get_qdrant_client()

    filter_conditions = None
    if peer_id or event_type:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        if peer_id:
            conditions.append(FieldCondition(key="peer_id", match=MatchValue(value=peer_id)))
        if event_type:
            conditions.append(FieldCondition(key="event_type", match=MatchValue(value=event_type)))
        filter_conditions = Filter(must=conditions)

    try:
        results = client.query_points(
            collection_name=COLLECTION_MAIN,
            prefetch=[
                Prefetch(
                    query=NamedVector(name="dense", vector=vectors["dense"]),
                    limit=TOP_K_PREFETCH,
                    filter=filter_conditions,
                ),
                Prefetch(
                    query=NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(
                            indices=vectors["sparse"]["indices"],
                            values=vectors["sparse"]["values"],
                        ),
                    ),
                    limit=TOP_K_PREFETCH,
                    filter=filter_conditions,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return [p.payload for p in results.points]
    except Exception as e:
        log.error("Qdrant 검색 실패: %s", e)
        return []
