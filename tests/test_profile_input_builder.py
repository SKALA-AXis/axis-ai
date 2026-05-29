from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.services.profile_input_builder import ProfileInputBuilder, normalize_business_area


class _Row:
    def __init__(self, **kwargs: Any) -> None:
        self._mapping = kwargs


class _Result:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def first(self) -> _Row | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[_Row]:
        return self._rows


class _FakeSession:
    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _Result:
        sql = str(statement)
        if "FROM peer_companies" in sql:
            return _Result(
                [_Row(id="lg_cns", name="LG CNS", keywords=["LG CNS"], core_keywords=["AX"])]
            )
        if "FROM (" in sql and "raw_article_financial_metrics" in sql:
            return _Result([_Row(period_year=2026, period_quarter=1)])
        if "FROM raw_articles" in sql and "source_type = 'dart'" in sql:
            return _Result(
                [
                    _Row(
                        id=14499,
                        source_type="dart",
                        source_name="dart",
                        title="분기보고서 (2026.03)",
                        content=(
                            "클라우드&AI 사업은 AX를 지원한다. "
                            "스마트엔지니어링은 스마트팩토리와 스마트물류를 포함한다. "
                            "Digital Business Service는 SI/SM을 포함한다."
                        ),
                        url="https://dart.example",
                        published_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
                    )
                ]
            )
        if "FROM raw_article_financial_metrics fm" in sql:
            return _Result(
                [
                    _Row(
                        id=1,
                        raw_article_id=195,
                        source_type="ir",
                        source_name="ir_pdf",
                        peer_id="lg_cns",
                        period="2026Q1",
                        period_year=2026,
                        period_quarter=1,
                        metric_name="revenue_total",
                        metric_label="매출",
                        metric_scope="segment",
                        business_area="클라우드&AI",
                        value_numeric=7950,
                        value_krwbn=7950,
                        value_krw=795000000000,
                        unit="억원",
                        currency="KRW",
                        confidence=0.88,
                        evidence_text="전년대비매출480억원증가(YoY +6.7%)",
                        url="https://ir.example",
                        title="2026년 1분기 경영실적 발표",
                        published_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
                    ),
                    _Row(
                        id=2,
                        raw_article_id=195,
                        source_type="ir",
                        source_name="ir_pdf",
                        peer_id="lg_cns",
                        period="2026Q1",
                        period_year=2026,
                        period_quarter=1,
                        metric_name="operating_profit",
                        metric_label="영업이익",
                        metric_scope="company_total",
                        business_area="company_total",
                        value_numeric=573,
                        value_krwbn=573,
                        value_krw=57300000000,
                        unit="억원",
                        currency="KRW",
                        confidence=0.88,
                        evidence_text="[영업이익] [매출] 당기순이익 | 1Q26 573",
                        url="https://ir.example",
                        title="2026년 1분기 경영실적 발표",
                        published_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
                    ),
                ]
            )
        if "FROM raw_article_business_signals bs" in sql:
            return _Result(
                [
                    _Row(
                        id=300465,
                        raw_article_id=195,
                        source_type="ir",
                        source_name="ir_pdf",
                        peer_id="lg_cns",
                        period="2026Q1",
                        period_year=2026,
                        period_quarter=1,
                        business_area="ai_ax",
                        signal_type="business_update",
                        sentiment="positive",
                        summary="AgenticWorks 기반 사업 확장이 본격화되고 있다.",
                        evidence_text="AgenticWorks 기반 사업 확장 본격화",
                        confidence=0.78,
                        url="https://ir.example",
                        title="2026년 1분기 경영실적 발표",
                        published_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
                    )
                ]
            )
        if "FROM raw_articles" in sql and "source_type = 'official'" in sql:
            return _Result(
                [
                    _Row(
                        id=36282,
                        source_type="official",
                        source_name="LG CNS Press",
                        title="LG CNS, 북미 제조 AX 시장 확대",
                        content="Factova 기반 스마트팩토리 솔루션으로 북미 제조 AX 시장 공략",
                        url="https://official.example",
                        published_at=datetime(2026, 5, 19, tzinfo=timezone.utc),
                        matched_companies=["lg_cns"],
                        matched_sectors=["ax"],
                    )
                ]
            )
        return _Result([])


def test_normalize_business_area_maps_source_labels() -> None:
    assert normalize_business_area("ai_ax") == "클라우드&AI"
    assert normalize_business_area("logistics") == "스마트 엔지니어링"
    assert normalize_business_area("Enterprise IT") == "Digital Business Service"


def test_profile_input_builder_builds_evidence_pack() -> None:
    builder = ProfileInputBuilder(session_factory=lambda: _FakeSession())

    pack = builder.build("lg_cns")

    assert pack["company"]["id"] == "lg_cns"
    assert pack["period"] == "2026Q1"
    assert len(pack["business_area_evidence"]) == 3
    assert pack["direction_evidence"][0]["business_area"] == "클라우드&AI"
    assert pack["financial_evidence"][0]["yoy_pct"] == 6.7
    assert all(item["metric"] != "operating_profit" for item in pack["financial_evidence"])
    assert pack["execution_evidence"][0]["business_area"] == "스마트 엔지니어링"
    assert pack["source_index"]
