from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.rag import content_index


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[Any] = []

    def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        self.points.extend(points)

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(
            collections=[SimpleNamespace(name=content_index.COLLECTION_DOCUMENTS)]
        )


def test_raw_article_chunks_keep_exact_lookup_metadata():
    chunks = content_index.raw_article_chunks(
        {
            "id": 123,
            "title": "뉴스 제목",
            "content": "본문 " * 1200,
            "source_type": "naver_news",
            "source_name": "Naver",
            "url": "https://example.test/news/123",
            "peer_id": "lg_cns",
        }
    )

    assert len(chunks) > 1
    assert {chunk.raw_article_id for chunk in chunks} == {123}
    assert {chunk.source_table for chunk in chunks} == {content_index.TABLE_RAW_ARTICLES}
    assert {chunk.content_kind for chunk in chunks} == {content_index.KIND_RAW_ARTICLE_BODY}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.payload()["source_id"] == "123" for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == ("본문 " * 1200).strip()


def test_index_raw_article_record_upserts_payload(monkeypatch):
    fake_client = _FakeQdrantClient()

    monkeypatch.setattr(content_index, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(content_index, "ensure_collections", lambda _client: None)
    monkeypatch.setattr(content_index, "_delete_existing_chunks", lambda *_args: None)
    monkeypatch.setattr(
        content_index,
        "embed_text",
        lambda _text, mode: {"dense": [0.0], "sparse": {"indices": [1], "values": [1.0]}},
    )

    result = content_index.index_raw_article_record(
        {
            "id": 55,
            "title": "원문 뉴스",
            "content": "정확한 본문",
            "source_type": "news",
            "peer_id": "samsung_sds",
        }
    )

    assert result["indexed"] == 1
    payload = fake_client.points[0].payload
    assert payload["content_index_version"] == content_index.CONTENT_INDEX_VERSION
    assert payload["content_kind"] == content_index.KIND_RAW_ARTICLE_BODY
    assert payload["raw_article_id"] == 55
    assert payload["peer_id"] == "samsung_sds"
    assert payload["text"] == "정확한 본문"


def test_get_raw_article_body_reassembles_chunks_by_id(monkeypatch):
    payloads = [
        {
            "content_index_version": content_index.CONTENT_INDEX_VERSION,
            "source_table": content_index.TABLE_RAW_ARTICLES,
            "source_id": "77",
            "content_kind": content_index.KIND_RAW_ARTICLE_BODY,
            "raw_article_id": 77,
            "chunk_index": 1,
            "char_start": 3,
            "text": "DEF",
        },
        {
            "content_index_version": content_index.CONTENT_INDEX_VERSION,
            "source_table": content_index.TABLE_RAW_ARTICLES,
            "source_id": "77",
            "content_kind": content_index.KIND_RAW_ARTICLE_BODY,
            "raw_article_id": 77,
            "chunk_index": 0,
            "char_start": 0,
            "text": "ABC",
        },
    ]

    class _FakeScrollClient:
        def scroll(self, **_kwargs):
            return [SimpleNamespace(payload=payload) for payload in payloads], None

    monkeypatch.setattr(content_index, "get_qdrant_client", lambda: _FakeScrollClient())

    result = content_index.get_raw_article_body(77, fallback_to_rdb=False)

    assert result.source == "vdb"
    assert result.text == "ABCDEF"
    assert result.content_kinds == [content_index.KIND_RAW_ARTICLE_BODY]


def test_attach_vdb_context_to_row_adds_card_and_issue_context(monkeypatch):
    monkeypatch.setattr(
        content_index,
        "get_integrated_issue_context",
        lambda _issue_id, fallback_to_rdb: content_index.ContentFetchResult(
            "통합 이슈 VDB 본문",
            [{"chunk_index": 0}],
            "vdb",
            [content_index.KIND_INTEGRATED_ISSUE],
        ),
    )
    monkeypatch.setattr(
        content_index,
        "get_card_analysis_context",
        lambda _card_id, fallback_to_rdb: content_index.ContentFetchResult(
            "분석/시사점/대응 VDB 본문",
            [{"chunk_index": 0}, {"chunk_index": 1}],
            "vdb",
            list(content_index.CARD_CONTENT_KINDS),
        ),
    )

    row = content_index.attach_vdb_context_to_row(
        {"id": "CARD-1", "integrated_issue_id": "11111111-1111-1111-1111-111111111111"}
    )

    context = row["evidence_payload"]["vdb_context"]
    assert context["integrated_issue"]["text"] == "통합 이슈 VDB 본문"
    assert context["card_analysis"]["text"] == "분석/시사점/대응 VDB 본문"
    assert context["card_analysis"]["chunk_count"] == 2


def test_search_content_chunks_returns_payload_hits(monkeypatch):
    class _FakeSearchClient:
        def query_points(self, **_kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="point-1",
                        score=0.88,
                        payload={
                            "content_kind": content_index.KIND_RAW_ARTICLE_BODY,
                            "raw_article_id": 99,
                            "title": "검색된 뉴스",
                            "text": "검색 본문 청크",
                        },
                    )
                ]
            )

    monkeypatch.setattr(content_index, "get_qdrant_client", lambda: _FakeSearchClient())
    monkeypatch.setattr(
        content_index,
        "embed_text",
        lambda _text, mode: {"dense": [0.0], "sparse": {"indices": [1], "values": [1.0]}},
    )

    hits = content_index.search_content_chunks(
        "검색 질의",
        source_table=content_index.TABLE_RAW_ARTICLES,
        content_kind=content_index.KIND_RAW_ARTICLE_BODY,
    )

    assert hits == [
        {
            "content_kind": content_index.KIND_RAW_ARTICLE_BODY,
            "raw_article_id": 99,
            "title": "검색된 뉴스",
            "text": "검색 본문 청크",
            "score": 0.88,
            "point_id": "point-1",
        }
    ]
