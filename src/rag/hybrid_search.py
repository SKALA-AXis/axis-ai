# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인으로 하이브리드 검색 도입, ruff 포맷
#   2026-04-28 박지원 — 크롤러 구현·전처리 agent 수정 반영 및 일부 변경 revert
#   2026-06-08 박진 — 챗봇 RAG 추가·고도화 및 목업 삭제, 포맷 정리
"""Qdrant RRF 하이브리드 검색 — Dense + Sparse 융합"""

import logging
from typing import Optional

from qdrant_client.models import (
    Fusion,
    FusionQuery,
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
    company: Optional[str] = None,
    event_type: Optional[str] = None,
    peer_id: Optional[str] = None,
    raise_on_failure: bool = False,
    peer_ids: Optional[list[str]] = None,
    published_before_ts: Optional[int] = None,
) -> list[dict]:
    """BGE-M3 Dense+Sparse RRF 하이브리드 검색.

    Args:
        query: 검색 쿼리.
        top_k: 반환할 결과 수.
        company: 회사 필터 (없으면 전체).
        event_type: 이벤트 타입 필터.

        raise_on_failure: True면 embedding/Qdrant 장애를 빈 결과로 숨기지 않고 예외로 전달.
        peer_ids: 복수 회사 필터 — 지정 시 company/peer_id 단일 필터보다 우선.
        published_before_ts: published_at <= 이 Unix timestamp 인 포인트만 매칭 (선례 시간차).

    Returns:
        검색 결과 목록 (rdb_id, title, summary 등 포함).
    """
    try:
        from src.rag.embedder import embed_text

        vectors = embed_text(query, mode="both")
    except Exception as e:
        log.warning("임베딩 실패. BM25 폴백: %s", e)
        if raise_on_failure:
            raise RuntimeError("hybrid_search_embedding_failed") from e
        return []

    client = get_qdrant_client()

    filter_conditions = None
    company_filter = company or peer_id

    if company_filter or event_type or peer_ids or published_before_ts is not None:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

        conditions = []
        if peer_ids:
            conditions.append(FieldCondition(key="company", match=MatchAny(any=list(peer_ids))))
        elif company_filter:
            conditions.append(FieldCondition(key="company", match=MatchValue(value=company_filter)))
        if event_type:
            conditions.append(FieldCondition(key="event_type", match=MatchValue(value=event_type)))
        if published_before_ts is not None:
            conditions.append(
                FieldCondition(key="published_at", range=Range(lte=float(published_before_ts)))
            )
        filter_conditions = Filter(must=conditions)  # type: ignore[arg-type]

    try:
        results = client.query_points(
            collection_name=COLLECTION_MAIN,
            prefetch=[
                Prefetch(
                    query=vectors["dense"],
                    using="dense",
                    limit=TOP_K_PREFETCH,
                    filter=filter_conditions,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=vectors["sparse"]["indices"],
                        values=vectors["sparse"]["values"],
                    ),
                    using="sparse",
                    limit=TOP_K_PREFETCH,
                    filter=filter_conditions,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return [{**p.payload, "score": p.score} for p in results.points if p.payload is not None]
    except Exception as e:
        log.warning("Qdrant query_points 검색 실패. REST fallback 시도: %s", e)

    try:
        from src.rag.qdrant_compat import legacy_rrf_search

        hits = legacy_rrf_search(
            collection_name=COLLECTION_MAIN,
            dense_vector=vectors["dense"],
            sparse_vector=vectors["sparse"],
            limit=top_k,
            prefetch_limit=TOP_K_PREFETCH,
            query_filter=filter_conditions,
        )
        return [
            {**(hit.get("payload") or {}), "score": hit.get("score")}
            for hit in hits
            if hit.get("payload")
        ]
    except Exception as e:
        log.error("Qdrant REST fallback 검색 실패: %s", e)
        if raise_on_failure:
            raise RuntimeError("hybrid_search_qdrant_failed") from e
        return []
