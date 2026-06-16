"""W4-2 — AnalysisContextBuilder 압축 / token budget 단위 테스트.

DB query 는 mock 처리. compression 로직 위주.
"""

from __future__ import annotations

from unittest.mock import patch

from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    FinancialSeries,
    FinancialSeriesPoint,
    PrecedentCandidate,
    RetrievedCard,
    SectorPulseRow,
    TimelineEntry,
)
from src.services.analysis_context_builder import (
    AnalysisContextBuilder,
    _compress_to_budget,
    _estimate_token_count,
)


def _stub_bundle() -> AnalysisInputBundle:
    return AnalysisInputBundle(
        bundle_id="news:1",
        cluster_id="1",
        source_type="news",
        companies=["samsung_sds"],
        sectors=["ax"],
        event_type="partnership",
        items=[],
        facts=[],
        evidence_snippets=[],
        sources=[],
        metadata={},
    )


def _heavy_context() -> AnalysisContext:
    ctx = AnalysisContext()
    ctx.peer_event_timeline_recent = [
        TimelineEntry(
            company_id=f"peer_{i % 3}",
            event_date="2026-05-15",
            card_id=f"CN-{i:04d}",
            event_type="partnership",
            sector="ax",
            headline="A" * 200,
            importance="high",
            importance_score=0.9,
        )
        for i in range(30)
    ]
    ctx.sector_pulse_recent = [
        SectorPulseRow(
            sector="ax",
            week_start=f"2026-05-{w:02d}",
            event_count=12,
            peer_event_count=6,
            general_event_count=6,
            intensity_avg=0.6,
        )
        for w in range(1, 10)
    ]
    ctx.financial_trend = {
        f"peer_{i}::revenue_total": FinancialSeries(
            company_id=f"peer_{i}",
            metric_name_canonical="revenue_total",
            points=[
                FinancialSeriesPoint(
                    period_year=2024 + (q // 4),
                    period_quarter=f"Q{(q % 4) + 1}",
                    value_numeric=1_000_000.0 + q,
                )
                for q in range(20)
            ],
        )
        for i in range(3)
    }
    ctx.event_chain_candidates = [
        PrecedentCandidate(
            card_id=f"CN-prev-{i}",
            company_id="peer_0",
            event_type="partnership",
            days_since=14,
            cosine=0.8,
            headline="C" * 100,
        )
        for i in range(8)
    ]
    ctx.similar_cards_rag = [
        RetrievedCard(card_id=f"CN-rag-{i}", cosine=0.8, headline="D" * 100) for i in range(6)
    ]
    return ctx


def test_compression_reduces_token_count():
    ctx = _heavy_context()
    initial = _estimate_token_count(ctx)
    compressed = _compress_to_budget(ctx, budget=4000)
    after = _estimate_token_count(compressed)
    assert after <= initial
    # 압축 우선순위 적용: RAG 2개 이하, sector_pulse 2주 이하, event_chain 3 이하.
    assert len(compressed.similar_cards_rag) <= 2
    assert len(compressed.sector_pulse_recent) <= 2
    assert len(compressed.event_chain_candidates) <= 3


def test_builder_returns_empty_context_when_db_unavailable():
    """SessionLocal 가 실패하더라도 fallback 으로 빈 context 반환."""
    builder = AnalysisContextBuilder()
    # 모든 query 가 fallback 으로 빈 결과를 내도록 SessionLocal patch.
    with patch(
        "src.services.analysis_context_builder.SessionLocal",
        side_effect=RuntimeError("db unavailable"),
    ):
        ctx = builder.build(
            input_bundle=_stub_bundle(),
            profile_context=None,
            integrated_issue={"main_company": "samsung_sds"},
        )
    assert isinstance(ctx, AnalysisContext)
    assert ctx.is_empty()
    assert ctx.available_layer_count() == 0


def test_builder_accepts_as_of_for_point_in_time_backfill():
    """as_of(point-in-time 백필 기준일) 가 모든 query 메서드로 무사히 전달되는지 — 시그니처 회귀.

    SessionLocal 을 patch 해 DB 없이도 실행. anchor 를 한 메서드라도 빠뜨리면 호출 시
    TypeError 로 드러난다. (SQL 클램프 자체의 정확성은 live-DB smoke 로 별도 검증.)
    """
    from datetime import date

    builder = AnalysisContextBuilder()
    with patch(
        "src.services.analysis_context_builder.SessionLocal",
        side_effect=RuntimeError("db unavailable"),
    ):
        ctx = builder.build(
            input_bundle=_stub_bundle(),
            profile_context=None,
            integrated_issue={"main_company": "samsung_sds"},
            as_of=date(2026, 1, 15),
        )
    assert isinstance(ctx, AnalysisContext)
    assert ctx.is_empty()


def test_available_layer_count_consistent_with_provenance():
    ctx = _heavy_context()
    # compression 전엔 5 layers —
    # peer_event_timeline_recent / sector_pulse_recent /
    # financial_trend / event_chain_candidates / similar_cards_rag.
    assert ctx.available_layer_count() == 5
