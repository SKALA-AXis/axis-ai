"""QdrantPrecedentSearch 어댑터 단위 테스트 — hybrid_search 는 monkeypatch."""

import time
from datetime import date
from typing import Any

import pytest

import src.rag.hybrid_search as hybrid_search_module
from src.analysis.models import AnalysisInputBundle, PrecedentCandidate, RetrievedCard
from src.rag.precedent_search import (
    PrecedentSearchEmptyError,
    QdrantPrecedentSearch,
    _bundle_query_text,
)

_DAY = 86_400


def _bundle(**overrides: Any) -> AnalysisInputBundle:
    base: dict[str, Any] = {
        "bundle_id": "news:1",
        "cluster_id": "777",
        "source_type": "news",
        "companies": ["samsung_sds"],
        "sectors": ["ax"],
        "event_type": "partnership",
        "items": [],
        "facts": [],
        "evidence_snippets": [{"text": "삼성SDS OpenAI 리셀러 계약"}],
        "sources": [],
    }
    base.update(overrides)
    return AnalysisInputBundle(**base)


def _hit(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "card_news_id": "IC-20260601-001",
        "company": "lg_cns",
        "event_type": "partnership",
        "published_at": int(time.time()) - 30 * _DAY,
        "cluster_id": 555,
        "title": "LG CNS 팔란티어 제휴",
        "sector": "ax",
        "score": 0.81,
        "rdb_id": 42,
    }
    base.update(overrides)
    return base


def _patch_search(monkeypatch: pytest.MonkeyPatch, hits: list[dict[str, Any]]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_search(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        captured["query"] = query
        captured.update(kwargs)
        return hits

    monkeypatch.setattr(hybrid_search_module, "hybrid_search", fake_search)
    return captured


class TestFindPrecedents:
    def test_maps_hits_to_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_search(monkeypatch, [_hit()])

        out = QdrantPrecedentSearch().find_precedents(
            bundle=_bundle(), peers=["lg_cns"], min_days_since=7, top_k=5
        )

        assert len(out) == 1
        candidate = out[0]
        assert isinstance(candidate, PrecedentCandidate)
        assert candidate.card_id == "IC-20260601-001"
        assert candidate.company_id == "lg_cns"
        assert candidate.event_type == "partnership"
        assert candidate.cosine == pytest.approx(0.81)
        assert candidate.headline == "LG CNS 팔란티어 제휴"
        assert 29 <= candidate.days_since <= 31

    def test_passes_peers_and_time_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_search(monkeypatch, [_hit()])

        QdrantPrecedentSearch().find_precedents(
            bundle=_bundle(), peers=["lg_cns", "posco_dx"], min_days_since=7, top_k=5
        )

        assert captured["peer_ids"] == ["lg_cns", "posco_dx"]
        cutoff = captured["published_before_ts"]
        assert int(time.time()) - 7 * _DAY - 60 <= cutoff <= int(time.time()) - 7 * _DAY + 60
        assert "삼성SDS" in captured["query"]

    def test_excludes_same_cluster_and_missing_card_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_search(
            monkeypatch,
            [
                _hit(cluster_id=777),  # bundle 과 같은 클러스터 — 자기 자신
                _hit(card_news_id=""),  # card_id 없음 — 화면 복원 불가
                _hit(card_news_id="IC-OK", cluster_id=1),
            ],
        )

        out = QdrantPrecedentSearch().find_precedents(
            bundle=_bundle(), peers=["lg_cns"], min_days_since=7, top_k=5
        )

        assert [c.card_id for c in out] == ["IC-OK"]

    def test_caps_at_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_search(monkeypatch, [_hit(card_news_id=f"IC-{i}", cluster_id=i) for i in range(10)])

        out = QdrantPrecedentSearch().find_precedents(
            bundle=_bundle(), peers=[], min_days_since=7, top_k=3
        )

        assert len(out) == 3

    def test_empty_results_raise_for_db_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_search(monkeypatch, [])

        with pytest.raises(PrecedentSearchEmptyError):
            QdrantPrecedentSearch().find_precedents(
                bundle=_bundle(), peers=["lg_cns"], min_days_since=7, top_k=5
            )


class TestSearchByBundle:
    def test_maps_hits_to_retrieved_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_search(monkeypatch, [_hit()])

        out = QdrantPrecedentSearch().search_by_bundle(_bundle(), top_k=3)

        assert len(out) == 1
        card = out[0]
        assert isinstance(card, RetrievedCard)
        assert card.card_id == "IC-20260601-001"
        assert card.sector == "ax"
        assert card.company_id == "lg_cns"
        assert card.cosine == pytest.approx(0.81)
        # Layer 3 RAG 는 시간차·회사 필터 없음.
        assert "published_before_ts" not in captured
        assert "peer_ids" not in captured

    def test_empty_query_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bundle = _bundle(
            evidence_snippets=[], facts=[], items=[], companies=[], sectors=[], event_type=None
        )
        out = QdrantPrecedentSearch().search_by_bundle(bundle, top_k=3)
        assert out == []


class TestBundleQueryText:
    def test_prefers_evidence_snippets(self) -> None:
        assert _bundle_query_text(_bundle()) == "삼성SDS OpenAI 리셀러 계약"

    def test_falls_back_to_facts_then_items(self) -> None:
        bundle = _bundle(evidence_snippets=[], facts=[{"statement": "사실 텍스트"}])
        assert _bundle_query_text(bundle) == "사실 텍스트"

        bundle = _bundle(evidence_snippets=[], facts=[], items=[{"title": "아이템 제목"}])
        assert _bundle_query_text(bundle) == "아이템 제목"

    def test_keyword_fallback_when_no_text(self) -> None:
        bundle = _bundle(evidence_snippets=[], facts=[], items=[{"id": 1}])
        text = _bundle_query_text(bundle)
        assert "samsung_sds" in text
        assert "partnership" in text

    def test_truncates_long_text(self) -> None:
        bundle = _bundle(evidence_snippets=[{"text": "가" * 2000}])
        assert len(_bundle_query_text(bundle)) <= 512


class TestBuilderIntegration:
    """AnalysisContextBuilder 가 어댑터 결과를 그대로 쓰고, 예외 시 무사히 폴백하는지."""

    def test_builder_uses_adapter_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.analysis_context_builder import AnalysisContextBuilder

        _patch_search(monkeypatch, [_hit()])
        builder = AnalysisContextBuilder(qdrant_search=QdrantPrecedentSearch())

        out = builder._find_precedents(
            input_bundle=_bundle(),
            peers=["lg_cns"],
            sectors=["ax"],
            min_days_since=7,
            top_k=5,
            anchor=date.today(),
        )

        assert [c.card_id for c in out] == ["IC-20260601-001"]

    def test_builder_survives_adapter_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.services.analysis_context_builder import AnalysisContextBuilder

        def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(hybrid_search_module, "hybrid_search", boom)
        builder = AnalysisContextBuilder(qdrant_search=QdrantPrecedentSearch())

        # 어댑터 예외 → DB fallback 경로로 내려가며 예외가 전파되지 않아야 한다.
        # (로컬 DB 유무에 따라 fallback 결과 수는 달라질 수 있어 내용은 단언하지 않는다.)
        out = builder._find_precedents(
            input_bundle=_bundle(),
            peers=["lg_cns"],
            sectors=["ax"],
            min_days_since=7,
            top_k=5,
            anchor=date.today(),
        )

        assert isinstance(out, list)
