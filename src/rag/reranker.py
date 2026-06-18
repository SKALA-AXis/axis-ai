# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인으로 추가 후 ruff format 적용
#   2026-04-28 박지원 — 크롤러 로직 개선 작업 반영 및 해당 변경 되돌림(revert)
"""BGE-reranker-v2-m3 재랭킹"""

import logging

log = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            log.info("BGE-reranker 모델 로딩 중...")
            _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
            log.info("BGE-reranker 모델 로딩 완료")
        except Exception as e:
            log.error("Reranker 로딩 실패: %s", e)
            raise
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """후보 문서를 query 기준으로 재랭킹한다.

    Args:
        query: 검색 쿼리.
        candidates: hybrid_search 결과 목록.
        top_k: 최종 반환 수.

    Returns:
        재랭킹된 결과 목록 (rerank_score 필드 추가).
    """
    if not candidates:
        return []
    try:
        reranker = get_reranker()
        pairs = [(query, c.get("summary", c.get("title", ""))) for c in candidates]
        scores = reranker.compute_score(pairs, normalize=True)
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
    except Exception as e:
        log.error("Reranker 실패: %s", e)
        return candidates[:top_k]
