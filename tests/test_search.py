"""검색 단위 테스트"""

from unittest.mock import patch


def test_search_returns_empty_on_qdrant_failure():
    with patch("src.rag.hybrid_search.get_qdrant_client") as mock_client:
        mock_client.return_value.query_points.side_effect = Exception("연결 실패")
        from src.rag.hybrid_search import hybrid_search

        result = hybrid_search("삼성SDS 전략")
        assert result == []
