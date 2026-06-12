"""검색 단위 테스트"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException


def test_search_returns_empty_on_qdrant_failure():
    with (
        patch("src.rag.hybrid_search.get_qdrant_client") as mock_client,
        patch("src.rag.qdrant_compat.legacy_rrf_search") as mock_legacy_search,
    ):
        mock_client.return_value.query_points.side_effect = Exception("연결 실패")
        mock_legacy_search.side_effect = Exception("REST 연결 실패")
        from src.rag.hybrid_search import hybrid_search

        result = hybrid_search("삼성SDS 전략")
        assert result == []


def test_api_search_pipeline_uses_hybrid_search_and_rerank():
    from src.api.router import _run_search_pipeline
    from src.schemas import SearchRequest

    raw_hit = {
        "rdb_id": 123,
        "company": "lg_cns",
        "title": "LG CNS AI partnership",
        "summary": "AI 사업 협력 확대",
        "event_type": "partnership",
        "importance": "high",
        "published_at": 1_717_200_000,
        "score": 0.62,
    }
    with (
        patch("src.rag.hybrid_search.hybrid_search", return_value=[raw_hit]) as hybrid,
        patch(
            "src.rag.reranker.rerank",
            return_value=[{**raw_hit, "rerank_score": 0.91}],
        ) as rerank,
    ):
        hits, timings = _run_search_pipeline(SearchRequest(query="AI partnership", top_k=3))

    hybrid.assert_called_once()
    assert hybrid.call_args.kwargs["raise_on_failure"] is True
    rerank.assert_called_once()
    assert hits[0]["rdb_id"] == 123
    assert hits[0]["rerank_score"] == 0.91
    assert timings["search_ms"] >= 0
    assert timings["rerank_ms"] >= 0


def test_api_search_pipeline_surfaces_rag_failure():
    from src.api.router import _run_search_pipeline
    from src.schemas import SearchRequest

    with patch("src.rag.hybrid_search.hybrid_search", side_effect=RuntimeError("qdrant down")):
        with pytest.raises(HTTPException) as exc_info:
            _run_search_pipeline(SearchRequest(query="AI partnership"))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail["code"] == "SEARCH_RAG_UNAVAILABLE"
