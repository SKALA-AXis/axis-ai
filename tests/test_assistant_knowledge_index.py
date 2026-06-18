# 작성일: 2026-06-08
# 작성자: 박진
# 변경이력:
#   2026-06-08 박진 — 챗봇 에이전트 및 어시스턴트 RAG 추가에 따른 지식 인덱스 테스트 작성
#   2026-06-11 박지원 — ruff 포매팅 정리 적용
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.rag import assistant_knowledge_index as indexer


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[Any] = []

    def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        self.points.extend(points)

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[SimpleNamespace(name=indexer.COLLECTION_DOCUMENTS)])


def test_index_can_exclude_card_news_analysis(monkeypatch):
    fake_client = _FakeQdrantClient()
    records = [
        indexer.AssistantKnowledgeRecord(
            source_type="integrated_issue",
            source_id="issue-1",
            title="통합 이슈",
            text="통합 이슈 본문",
        ),
        indexer.AssistantKnowledgeRecord(
            source_type="card_news_analysis",
            source_id="card-1",
            title="목업 카드",
            text="목업 카드 본문",
        ),
        indexer.AssistantKnowledgeRecord(
            source_type="peer_profile",
            source_id="peer-1",
            title="피어 프로필",
            text="피어 프로필 본문",
        ),
    ]

    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(indexer, "ensure_collections", lambda _client: None)
    monkeypatch.setattr(
        indexer,
        "iter_assistant_knowledge_records",
        lambda *, limit_per_source: iter(records),
    )
    monkeypatch.setattr(
        indexer,
        "embed_text",
        lambda _text, mode: {"dense": [0.0], "sparse": {"indices": [1], "values": [1.0]}},
    )

    result = indexer.index_assistant_knowledge(
        exclude_source_types=["card_news_analysis"],
        batch_size=10,
    )

    assert result["indexed"] == 2
    assert result["by_source"] == {"integrated_issue": 1, "peer_profile": 1}
    assert [point.payload["source_type"] for point in fake_client.points] == [
        "integrated_issue",
        "peer_profile",
    ]


def test_delete_assistant_knowledge_targets_selected_source_type(monkeypatch):
    fake_client = _FakeQdrantClient()
    calls: list[dict[str, Any]] = []

    def fake_delete_by_filter(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(indexer, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr("src.rag.qdrant_compat.delete_by_filter", fake_delete_by_filter)

    result = indexer.delete_assistant_knowledge(source_types=["card_news_analysis"])

    assert result["source_types"] == ["card_news_analysis"]
    assert list(result["deleted"]) == ["card_news_analysis"]
    assert calls[0]["collection_name"] == indexer.COLLECTION_DOCUMENTS
    query_filter = calls[0]["query_filter"].model_dump(mode="json")
    assert {
        condition["key"]: condition["match"]["value"] for condition in query_filter["must"]
    } == {
        "knowledge_version": indexer.ASSISTANT_KNOWLEDGE_VERSION,
        "source_type": "card_news_analysis",
    }
