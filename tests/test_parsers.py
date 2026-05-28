from datetime import datetime

from scripts.reprocess_ir_analysis import (
    _business_signals_from_parser_result,
    _metrics_from_parser_result,
)
from src.crawler.base import RawArticle
from src.crawler.sources.dart import _infer_header_rows
from src.extractors.ir_llm_analysis_extractor import (
    _normalize_llm_metrics,
    _normalize_llm_signals,
)
from src.parsers.dart_parser import DartParser
from src.parsers.ir_parser import IRParser
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.parser_router import DocumentParserRouter
from src.preprocessing.preprocessing import _parsed_document_metadata_patch


def test_ir_parser_parses_ir_crawler_article() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="[PAGE 1]\n2026년 1분기 경영실적\n매출액 9,365억원\n영업이익 300억원",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "pdf_url": "https://example.com/ir.pdf",
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_pages": 1,
            "pdf_parsed_pages": 1,
            "pdf_page_blocks": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "2026년 1분기 경영실적"},
                        {"text": "매출액 9,365억원"},
                        {"text": "영업이익 300억원"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)

    assert parsed["ok"] is True
    assert parsed["peer_id"] == "test_peer"
    assert parsed["period"] == "2026Q1"
    assert parsed["revenue_total_krwbn"] == 9365
    assert parsed["operating_profit_krwbn"] == 300
    assert parsed["financial_record"]["ir_page"] == 1


def test_ir_parser_extracts_financial_table_matrix_by_period() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 4,
                    "blocks": [
                        {"text": "Financial Results (단위: 억원)"},
                        {"text": "구분 2026년 1분기 2025년 1분기 2025년 2024년"},
                        {"text": "매출액 3,352 3,489 13,929 13,828"},
                        {"text": "영업이익 300 280 957 911"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert [candidate["period"] for candidate in table_candidates[:4]] == [
        "2026Q1",
        "2025Q1",
        "2025",
        "2024",
    ]
    assert [candidate["value_krwbn"] for candidate in table_candidates[:4]] == [
        3352,
        3489,
        13929,
        13828,
    ]
    assert table_candidates[0]["type"] == "revenue_total"
    assert table_candidates[0]["business_area"] == "company_total"
    assert table_candidates[0]["metric_scope"] == "company_total"
    assert table_candidates[0]["is_historical"] is False
    assert table_candidates[1]["is_historical"] is True
    assert table_candidates[4]["type"] == "operating_profit"
    assert table_candidates[4]["period"] == "2026Q1"
    assert parsed["financial_tables"][0]["page"] == 4


def test_ir_parser_normalizes_krw_bn_table_unit_to_eokwon() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 4,
                    "blocks": [
                        {"text": "Financial Results (Unit: KRW bn)"},
                        {"text": "구분 2026년 1분기 2025년 1분기"},
                        {"text": "Revenue 335.2 348.9"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix" and candidate.get("type") == "revenue_total"
    ]

    assert revenue_candidates[0]["unit"] == "십억원"
    assert revenue_candidates[0]["value_krwbn"] == 3352


def test_ir_parser_preserves_unknown_table_business_area_label() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 5,
                    "blocks": [
                        {"text": "Segment Revenue (단위: 억원)"},
                        {"text": "구분 2026년 1분기 2025년 1분기"},
                        {"text": "Digital Logistics Revenue 1,742 1,889"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    candidate = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    )

    assert candidate["metric_scope"] == "segment"
    assert candidate["business_area"] == "Digital Logistics"


def test_ir_parser_does_not_treat_margin_as_business_area() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 6,
                    "blocks": [
                        {"text": "Operating Profit Margin (%)"},
                        {"text": "구분 2026년 1분기 2025년 1분기"},
                        {"text": "Margin 4.8% 4.9%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    candidate = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    )

    assert candidate["type"] == "operating_margin"
    assert candidate["metric_scope"] == "company_total"
    assert candidate["business_area"] == "company_total"


def test_ir_parser_does_not_apply_amount_unit_to_margin_rows() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 7,
                    "blocks": [
                        {"text": "Hi-tech"},
                        {"text": "(단위: 십억원)"},
                        {"text": "구분 1Q25 2Q25 3Q25 4Q25 1Q26"},
                        {"text": "OP Margin"},
                        {"text": "솔루션 864 973 11.2% 972 11.1%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    margin_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("type") == "operating_margin"
    ]

    assert [candidate["period"] for candidate in margin_candidates] == ["2025Q3", "2026Q1"]
    assert [candidate["value_pct"] for candidate in margin_candidates] == [11.2, 11.1]
    assert all(candidate["unit"] == "%" for candidate in margin_candidates)
    assert all(
        "단위 근거: (단위: 십억원)" in candidate["evidence_text"] for candidate in margin_candidates
    )
    assert margin_candidates[0]["business_area"] == "솔루션"

    metrics = _metrics_from_parser_result(
        {
            "id": 103,
            "company": ["test_peer"],
            "title": article.title,
            "url": article.url,
            "source_name": article.source_name,
            "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
        },
        parsed,
        {"period": "2026Q1", "peer_id": "test_peer"},
    )
    margin_metrics = [metric for metric in metrics if metric["metric_name"] == "operating_margin"]
    assert all(metric["value_krwbn"] is None for metric in margin_metrics)
    assert all(metric["unit"] == "%" for metric in margin_metrics)


def test_ir_parser_does_not_store_percentage_cells_as_amount_metrics() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 10,
                    "blocks": [
                        {"text": "매출액 (단위: 억원)"},
                        {"text": "구분 1Q26 4Q25 QoQ 1Q25 YoY"},
                        {"text": "매출액"},
                        {"text": "IT서비스 16,105 16,690 -3.5% 16,004 +0.6%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix" and candidate.get("type") == "revenue_total"
    ]

    assert [candidate["period"] for candidate in revenue_candidates] == [
        "2026Q1",
        "2025Q4",
        "2025Q1",
    ]
    assert [candidate["value_krwbn"] for candidate in revenue_candidates] == [
        16105,
        16690,
        16004,
    ]
    assert all(candidate["unit"] == "억원" for candidate in revenue_candidates)
    assert all(
        "단위 근거: 매출액 (단위: 억원)" in candidate["evidence_text"]
        for candidate in revenue_candidates
    )
    revenue_qoq = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("type") == "revenue_total_qoq"
    )
    revenue_yoy = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("type") == "revenue_total_yoy"
    )
    assert revenue_qoq["period"] == "2026Q1"
    assert revenue_qoq["value_pct"] == -3.5
    assert revenue_qoq["unit"] == "%"
    assert revenue_yoy["period"] == "2026Q1"
    assert revenue_yoy["value_pct"] == 0.6
    assert revenue_yoy["unit"] == "%"


def test_ir_parser_maps_hierarchical_year_quarter_headers() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2025년 4분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 2, 10),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2025, "quarter": 4},
            "pdf_page_blocks": [
                {
                    "page": 3,
                    "blocks": [
                        {"text": "손익현황 (단위: 억원)"},
                        {"text": "구분 2024년 2025년"},
                        {"text": "4분기 연간 3분기 4분기 QoQ YoY 연간 YoY"},
                        {"text": "매출액 11,596 37,136 10,543 13,227 25.5% 14.1% 42,521 14.5%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("base_metric_type") == "revenue_total"
    ]

    assert [
        (
            candidate["type"],
            candidate["period"],
            candidate.get("value_krwbn") or candidate.get("value_pct"),
        )
        for candidate in revenue_candidates
    ] == [
        ("revenue_total", "2024Q4", 11596),
        ("revenue_total", "2024", 37136),
        ("revenue_total", "2025Q3", 10543),
        ("revenue_total", "2025Q4", 13227),
        ("revenue_total_qoq", "2025Q4", 25.5),
        ("revenue_total_yoy", "2025Q4", 14.1),
        ("revenue_total", "2025", 42521),
        ("revenue_total_yoy", "2025", 14.5),
    ]
    qoq = revenue_candidates[4]
    assert qoq["comparison_base_period"] == "2025Q3"
    assert qoq["comparison_base_value"] == 10543
    yoy = revenue_candidates[5]
    assert yoy["comparison_base_period"] == "2024Q4"
    assert yoy["comparison_base_value"] == 11596


def test_ir_parser_maps_split_year_quarter_pair_headers() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 4,
                    "blocks": [
                        {"text": "부문별 손익현황 (단위: 억원)"},
                        {"text": "구분 2025년 2026년"},
                        {"text": "1분기 1분기 YoY"},
                        {"text": "매 출 액 8,330 9,357 12.3%"},
                        {"text": "SI 2,996 3,568 19.1%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("base_metric_type") == "revenue_total"
    ]

    assert [
        (
            candidate["type"],
            candidate["period"],
            candidate.get("value_krwbn") or candidate.get("value_pct"),
            candidate["business_area"],
        )
        for candidate in revenue_candidates
    ] == [
        ("revenue_total", "2025Q1", 8330, "company_total"),
        ("revenue_total", "2026Q1", 9357, "company_total"),
        ("revenue_total_yoy", "2026Q1", 12.3, "company_total"),
        ("revenue_total", "2025Q1", 2996, "SI"),
        ("revenue_total", "2026Q1", 3568, "SI"),
        ("revenue_total_yoy", "2026Q1", 19.1, "SI"),
    ]
    yoy = revenue_candidates[2]
    assert yoy["comparison_base_period"] == "2025Q1"
    assert yoy["comparison_base_value"] == 8330


def test_ir_parser_reconstructs_table_lines_from_pdf_block_positions() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 4,
                    "blocks": [
                        {"bbox": [20, 20, 180, 35], "text": "부문별 손익현황 (단위: 억원)"},
                        {"bbox": [20, 60, 80, 80], "text": "구분"},
                        {"bbox": [120, 60, 180, 80], "text": "2025년"},
                        {"bbox": [220, 60, 280, 80], "text": "2026년"},
                        {"bbox": [120, 90, 180, 110], "text": "1분기"},
                        {"bbox": [220, 90, 280, 110], "text": "1분기"},
                        {"bbox": [320, 90, 360, 110], "text": "YoY"},
                        {"bbox": [20, 130, 80, 150], "text": "매 출 액"},
                        {"bbox": [120, 130, 180, 150], "text": "8,330"},
                        {"bbox": [220, 130, 280, 150], "text": "9,357"},
                        {"bbox": [320, 130, 360, 150], "text": "12.3%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("base_metric_type") == "revenue_total"
    ]

    assert [
        (
            candidate["type"],
            candidate["period"],
            candidate.get("value_krwbn") or candidate.get("value_pct"),
        )
        for candidate in revenue_candidates
    ] == [
        ("revenue_total", "2025Q1", 8330),
        ("revenue_total", "2026Q1", 9357),
        ("revenue_total_yoy", "2026Q1", 12.3),
    ]


def test_ir_reprocess_stores_only_report_period_from_table_candidates() -> None:
    article = {
        "id": 205,
        "company": ["test_peer"],
        "title": "테스트사 2025년 4분기 IR",
        "url": "https://example.com/ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2025Q4", "period_year": 2025, "period_quarter": 4},
    }
    parser_result = {
        "period": "2025Q4",
        "period_year": 2025,
        "period_quarter": 4,
        "candidates": [
            {
                "type": "revenue_total",
                "value_krwbn": 37136,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2024",
                "period_type": "year",
                "source": "ir_table_matrix",
                "confidence": 0.88,
            },
            {
                "type": "revenue_total",
                "value_krwbn": 13227,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2025Q4",
                "period_type": "quarter",
                "source": "ir_table_matrix",
                "confidence": 0.88,
            },
            {
                "type": "revenue_total_yoy",
                "value_pct": 14.5,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2025",
                "period_type": "year",
                "source": "ir_table_matrix",
                "confidence": 0.88,
            },
            {
                "type": "revenue_total_yoy",
                "value_pct": 14.1,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2025Q4",
                "period_type": "quarter",
                "source": "ir_table_matrix",
                "confidence": 0.88,
            },
        ],
    }

    metrics = _metrics_from_parser_result(
        article,
        parser_result,
        {"period": "2025Q4", "peer_id": "test_peer"},
    )

    assert [
        (metric["metric_name"], metric["period"], metric["value_numeric"]) for metric in metrics
    ] == [
        ("revenue_total", "2025Q4", 13227),
        ("revenue_total_yoy", "2025Q4", 14.1),
    ]


def test_ir_parser_uses_table_page_context_for_generic_metric_rows() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 7,
                    "blocks": [
                        {"text": "Cloud 사업"},
                        {"text": "Revenue (단위: 억원)"},
                        {"text": "구분 2026년 1분기 2025년 1분기"},
                        {"text": "Revenue 1,200 1,050"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    candidate = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    )

    assert candidate["metric_scope"] == "segment"
    assert candidate["business_area"] == "Cloud 사업"


def test_ir_parser_stops_table_matrix_before_narrative_text() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 8,
                    "blocks": [
                        {"text": "물류 사업 실적 (단위: 억원)"},
                        {"text": "구분 2025년 1분기 2025년 2분기 2025년 3분기 2025년 4분기"},
                        {"text": "Revenue 500 510 520 530"},
                        {"text": "Operating Profit 40 41 42 43"},
                        {
                            "text": (
                                "AI Transformation 및 DT 기반의 고부가 비즈니스 모델로 "
                                "개편 진행중 AI 솔루션 기반의 프로세스 자동화를 통한 "
                                "운영효율성 개선으로 2025년 1분기 매출 1, 2025년 2분기 매출 2, "
                                "2025년 3분기 매출 3, 2025년 4분기 매출 4"
                            )
                        },
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert len(table_candidates) == 8
    assert {candidate["business_area"] for candidate in table_candidates} == {"물류 사업"}
    assert all(
        "AI Transformation" not in candidate["business_area"] for candidate in table_candidates
    )


def test_ir_parser_does_not_use_narrative_context_as_business_area() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 9,
                    "blocks": [
                        {
                            "text": (
                                "AI Transformation 및 DT 기반의 고부가 비즈니스 모델로 개편 진행중 "
                                "AI 솔루션 기반의 프로세스 자동화를 통한 운영효율성 개선"
                            )
                        },
                        {"text": "Revenue (단위: 억원)"},
                        {"text": "구분 2025년 1분기 2025년 2분기"},
                        {"text": "Revenue 5,300 5,860"},
                        {"text": "Operating Profit 310 290"},
                        {"text": "Margin 4.9% 7.6%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert table_candidates
    assert {candidate["business_area"] for candidate in table_candidates} == {"company_total"}


def test_ir_parser_ignores_cost_and_gross_profit_rows_as_revenue() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="삼성SDS 2023년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2023, 4, 30),
        peer_id="samsung_sds",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2023, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 9,
                    "blocks": [
                        {"text": "손익 요약 (단위: 억원)"},
                        {"text": "구분 2022년 1분기 2022년 4분기 2023년 1분기"},
                        {"text": "매출액 34,000 42,000 35,000"},
                        {"text": "매출원가 28,000 38,000 29,000"},
                        {"text": "총이익 6,000 4,000 6,000"},
                        {"text": "총이익률(%) 17.6% 9.5% 17.1%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert len(table_candidates) == 3
    assert {
        candidate["metric_name"] if "metric_name" in candidate else candidate["type"]
        for candidate in table_candidates
    } == {"revenue_total"}
    assert all(candidate["business_area"] == "company_total" for candidate in table_candidates)


def test_ir_parser_does_not_use_punctuation_as_segment_label() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="포스코DX 2022년 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2023, 3, 1),
        peer_id="posco_dx",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2022, "quarter": 4},
            "pdf_page_blocks": [
                {
                    "page": 7,
                    "blocks": [
                        {"text": "매출액 (단위: 억원)"},
                        {"text": "구분 2018 2019 2020 2021 2022"},
                        {"text": "매출액 9,271 9,698 9,642 8,693 11,527"},
                        {"text": "' 18 19 20 21 22"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert len(table_candidates) == 5
    assert {candidate["business_area"] for candidate in table_candidates} == {"company_total"}


def test_ir_parser_keeps_table_flow_in_metric_evidence() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="현대오토에버 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="hyundai_autoever",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 10,
                    "blocks": [
                        {"text": "부문별 손익현황_26년 1분기 (연결 재무제표 기준)"},
                        {"text": "(단위: 억원)"},
                        {"text": "구분 23년 24년 25년 25년 1분기 26년 1분기"},
                        {"text": "매출액 30,650 37,136 42,521 8,330 9,357"},
                        {"text": "SI 10,098 12,789 16,572 2,996 3,568"},
                        {"text": "ITO 14,157 16,304 17,672 3,412 3,810"},
                        {"text": "Enterprise IT 24,255 29,093 34,244 6,408 7,378"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    enterprise_it = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
        and candidate.get("business_area") == "Enterprise IT"
        and candidate.get("period") == "2026Q1"
    )

    assert enterprise_it["type"] == "revenue_total"
    assert enterprise_it["metric_parent_label"] == "매출액"
    assert "부문별 손익현황" in enterprise_it["evidence_text"]
    assert "매출액 > Enterprise IT" in enterprise_it["evidence_text"]
    assert "26년 1분기 7,378" in enterprise_it["evidence_text"]


def test_ir_parser_infers_quarter_from_quarterly_table_title() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="현대오토에버 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="hyundai_autoever",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 4,
                    "blocks": [
                        {"text": "분기별손익계산서(연결재무제표기준)"},
                        {"text": "(단위: 억원)"},
                        {"text": "구분 2025년 2026년"},
                        {"text": "매출액 8,330 10,421"},
                        {"text": "SI 2,996 3,878"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix" and candidate.get("type") == "revenue_total"
    ]

    assert any(
        candidate.get("period") == "2026Q1"
        and candidate.get("business_area") == "company_total"
        and candidate.get("value_krwbn") == 10421
        for candidate in candidates
    )
    assert any(
        candidate.get("period") == "2025Q1"
        and candidate.get("business_area") == "SI"
        and candidate.get("value_krwbn") == 2996
        for candidate in candidates
    )


def test_ir_parser_does_not_apply_gpm_parent_to_amount_segment_rows() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="현대오토에버 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="hyundai_autoever",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 10,
                    "blocks": [
                        {"text": "부문별손익현황_26년1분기(연결재무제표기준)"},
                        {"text": "(단위: 억원)"},
                        {"text": "구분 23년 24년 25년 25년1분기 26년1분기"},
                        {"text": "매출액 30,650 37,136 42,521 8,330 9,357"},
                        {"text": "Enterprise IT 24,255 29,093 34,244 6,408 7,378"},
                        {"text": "GPM 8.6% 8.8% 9.9% 6.7% 7.3%"},
                        {"text": "차량SW 6,395 8,044 8,277 1,922 1,979"},
                        {"text": "GPM 20.3% 17.3% 12.7% 14.1% 9.7%"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix" and candidate.get("period") == "2026Q1"
    ]

    assert any(
        candidate.get("type") == "gross_margin"
        and candidate.get("business_area") == "Enterprise IT"
        and candidate.get("value_pct") == 7.3
        for candidate in candidates
    )
    assert any(
        candidate.get("type") == "revenue_total"
        and candidate.get("business_area") == "차량SW"
        and candidate.get("value_krwbn") == 1979
        for candidate in candidates
    )
    assert any(
        candidate.get("type") == "gross_margin"
        and candidate.get("business_area") == "차량SW"
        and candidate.get("value_pct") == 9.7
        for candidate in candidates
    )
    assert not any(
        candidate.get("type") == "gross_margin"
        and candidate.get("business_area") == "차량SW"
        and candidate.get("value_krwbn") == 1979
        for candidate in candidates
    )


def test_ir_reprocess_skips_non_report_period_table_candidates() -> None:
    article = {
        "id": 101,
        "company": ["test_peer"],
        "title": "테스트사 2026년 1분기 IR Presentation",
        "url": "https://example.com/ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "period_type": "quarter",
        "candidates": [
            {
                "page": 4,
                "type": "revenue_total",
                "value_kind": "amount_krwbn",
                "value_krwbn": 13828,
                "raw": "매출액 2024년 13,828 (억원)",
                "source": "ir_table_matrix",
                "source_table_uid": "ir-p4-t1",
                "row_label": "매출액",
                "column_label": "2024년",
                "period": "2024",
                "period_year": 2024,
                "period_type": "year",
                "is_historical": True,
                "metric_scope": "company_total",
                "business_area": None,
                "confidence": 0.88,
            }
        ],
    }

    metrics = _metrics_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "test_peer"},
    )

    assert metrics == []


def test_ir_reprocess_maps_llm_metrics_and_signals_separately() -> None:
    article = {
        "id": 102,
        "company": ["hyundai_autoever"],
        "title": "현대오토에버 2026년 1분기 IR Presentation",
        "url": "https://example.com/ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "period_type": "quarter",
        "candidates": [
            {
                "page": 6,
                "type": "revenue_total",
                "value_krwbn": 7378,
                "source": "ir_llm_analysis",
                "period": "2026Q1",
                "period_year": 2026,
                "period_quarter": 1,
                "period_type": "quarter",
                "metric_scope": "segment",
                "business_area": "Enterprise IT",
                "confidence": 0.86,
                "evidence_text": (
                    "부문별 손익현황 > 매출액 > Enterprise IT | "
                    "23년 24,255 | 24년 29,093 | 26년 1분기 7,378 (억원)"
                ),
            }
        ],
        "llm_business_signals": [
            {
                "business_area": "Enterprise IT",
                "signal_type": "growth",
                "sentiment": "positive",
                "summary": "신규 DX 프로젝트 수주 확대로 매출이 증가했다.",
                "evidence_text": (
                    "신규 DX 사업 수주 확대 및 생산성 개선 활동으로 전년 대비 "
                    "매출과 영업이익이 증가했습니다."
                ),
                "source_page": 6,
                "confidence": 0.82,
            }
        ],
    }

    metrics = _metrics_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "hyundai_autoever"},
    )
    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "hyundai_autoever"},
    )

    assert metrics[0]["metric_name"] == "revenue_total"
    assert metrics[0]["business_area"] == "Enterprise IT"
    assert metrics[0]["value_krwbn"] == 7378
    assert metrics[0]["extraction_method"] == "ir_llm.analysis"
    assert "Enterprise IT" in metrics[0]["evidence_text"]
    assert signals[0]["business_area"] == "Enterprise IT"
    assert signals[0]["signal_type"] == "growth"
    assert signals[0]["extraction_method"] == "ir_llm.analysis"
    assert "신규 DX 사업 수주 확대" in signals[0]["evidence_text"]


def test_ir_llm_analysis_filters_sk_ax_portfolio_companies() -> None:
    signals = _normalize_llm_signals(
        [
            {
                "business_area": "SK에코플랜트",
                "signal_type": "growth",
                "sentiment": "positive",
                "summary": "반도체사업 실적 호조로 매출 및 영업이익 증가.",
                "evidence_text": "SK에코플랜트는 반도체사업 실적 호조로 수익성이 개선되었습니다.",
                "source_page": 8,
                "confidence": 0.9,
            },
            {
                "business_area": "IT서비스",
                "signal_type": "growth",
                "sentiment": "positive",
                "summary": "AI Transformation 수요 확대로 IT서비스 매출이 증가.",
                "evidence_text": (
                    "SK AX IT서비스 부문은 AI Transformation 수요 확대로 성장했습니다."
                ),
                "source_page": 6,
                "confidence": 0.9,
            },
        ],
        peer_id="sk_ax",
    )
    metrics = _normalize_llm_metrics(
        [
            {
                "metric_name": "revenue_total",
                "metric_scope": "segment",
                "business_area": "SK스퀘어",
                "period": "2026Q1",
                "value_numeric": 1000,
                "unit": "억원",
                "evidence_text": "SK스퀘어 매출액 1,000억원",
                "confidence": 0.9,
            },
            {
                "metric_name": "revenue_total",
                "metric_scope": "segment",
                "business_area": "Enterprise IT",
                "period": "2026Q1",
                "value_numeric": 7378,
                "unit": "억원",
                "evidence_text": "SK AX Enterprise IT 매출액 7,378억원",
                "confidence": 0.9,
            },
        ],
        peer_id="sk_ax",
    )

    assert [signal["business_area"] for signal in signals] == ["IT서비스"]
    assert [metric["business_area"] for metric in metrics] == ["Enterprise IT"]


def test_ir_parser_filters_low_value_chunks_and_keeps_business_evidence() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="삼성SDS 2026년 1분기 실적발표",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="samsung_sds",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 1,
                    "blocks": [
                        {"text": "Samsung SDS 2026년 1분기 실적발표"},
                    ],
                },
                {
                    "page": 2,
                    "blocks": [
                        {
                            "text": (
                                "DISCLAIMER This presentation includes forward-looking "
                                "statements and should not be distributed without permission."
                            )
                        },
                    ],
                },
                {
                    "page": 3,
                    "blocks": [
                        {
                            "text": (
                                "Cloud 사업은 MSP와 생성형 AI 수요 확대를 기반으로 성장했습니다. "
                                "GPU 기반 클라우드 전환 프로젝트와 기업 데이터 플랫폼 고도화가 "
                                "동시에 진행되며 매출액 9,365억원, 영업이익 300억원을 기록했습니다."
                            )
                        },
                        {
                            "text": (
                                "Brity Copilot과 FabriX 중심의 AI 사업은 제조, 금융 고객의 "
                                "업무 자동화 프로젝트로 확대되고 있습니다."
                            )
                        },
                    ],
                },
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    chunk_text = "\n".join(chunk["text"] for chunk in parsed["document_chunks"])

    assert "DISCLAIMER" not in chunk_text
    assert "Cloud 사업" in chunk_text
    assert parsed["document_chunks"][0]["section_key"] == "cloud"
    assert parsed["revenue_total_krwbn"] == 9365
    assert parsed["operating_profit_krwbn"] == 300


def test_ir_parser_extracts_additional_financial_metric_candidates() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content=(
            "[PAGE 1]\n"
            "2026년 1분기 경영실적 종합\n"
            "매출액 9,365억원\n"
            "영업이익 300억원\n"
            "당기순이익 210억원\n"
            "EBITDA 450억원\n"
            "수주잔고 1.2조원\n"
            "CAPEX 80억원\n"
            "영업이익률 3.2%"
        ),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="test_peer",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    candidates_by_type = {candidate["type"]: candidate for candidate in parsed["candidates"]}

    assert parsed["financial_record"]["net_income_krwbn"] == 210
    assert parsed["financial_record"]["ebitda_krwbn"] == 450
    assert parsed["financial_record"]["backlog_krwbn"] == 12000
    assert parsed["financial_record"]["capex_krwbn"] == 80
    assert parsed["financial_record"]["operating_margin_pct"] == 3.2
    assert candidates_by_type["operating_margin"]["value_pct"] == 3.2
    assert candidates_by_type["backlog"]["value_krwbn"] == 12000


def test_ir_parser_marks_business_segment_metrics() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="삼성SDS 2026년 1분기 실적발표",
        content=(
            "[PAGE 1]\n"
            "2026년 1분기\n"
            "Cloud 사업은 MSP와 GPU 수요 확대에 따라 매출액 1,200억원을 기록했습니다."
        ),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="samsung_sds",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    revenue_candidate = {candidate["type"]: candidate for candidate in parsed["candidates"]}[
        "revenue_total"
    ]

    assert revenue_candidate["metric_scope"] == "segment"
    assert revenue_candidate["business_area"] == "cloud"


def test_ir_parser_keeps_multiple_metric_candidates() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="삼성SDS 2026년 1분기 실적발표",
        content=(
            "[PAGE 1]\n"
            "2026년 1분기 Financial Results\n"
            "Cloud 사업 매출액 1,200억원 영업이익률 12.5%\n"
            "물류 사업 매출액 900억원 영업이익률 4.1%"
        ),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="samsung_sds",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    revenue_candidates = [
        candidate for candidate in parsed["candidates"] if candidate["type"] == "revenue_total"
    ]
    margin_candidates = [
        candidate for candidate in parsed["candidates"] if candidate["type"] == "operating_margin"
    ]

    assert [candidate["value_krwbn"] for candidate in revenue_candidates] == [1200, 900]
    assert [candidate["value_pct"] for candidate in margin_candidates] == [12.5, 4.1]
    assert {candidate["business_area"] for candidate in revenue_candidates} == {
        "cloud",
        "logistics",
    }
    assert all(candidate["metric_scope"] == "segment" for candidate in revenue_candidates)


def test_ir_parser_uses_nearby_heading_for_business_area() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content=("[PAGE 1]\n2026년 1분기\nERP AI agent 사업\n매출액 800억원\n영업이익률 9.1%"),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    candidates_by_type = {candidate["type"]: candidate for candidate in parsed["candidates"]}

    assert candidates_by_type["revenue_total"]["metric_scope"] == "segment"
    assert candidates_by_type["revenue_total"]["business_area"] == "ai_ax"
    assert candidates_by_type["operating_margin"]["business_area"] == "ai_ax"


def test_ir_parser_does_not_mark_sk_metric_as_portfolio_without_related_entity() -> None:
    article = RawArticle(
        url="https://example.com/sk-inc-ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content=(
            "[PAGE 1]\n"
            "2026년 1분기\n"
            "Portfolio Overview\n"
            "SK AX 경영실적\n"
            "수주잔고 21.6조원\n"
            "영업이익률 4.9%"
        ),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    candidates_by_type = {candidate["type"]: candidate for candidate in parsed["candidates"]}

    assert candidates_by_type["backlog"]["metric_scope"] == "company_total"
    assert candidates_by_type["backlog"]["business_area"] == "company_total"
    assert candidates_by_type["backlog"]["entity_name"] == "sk_ax"
    assert candidates_by_type["operating_margin"]["metric_scope"] == "company_total"
    assert candidates_by_type["operating_margin"]["business_area"] == "company_total"


def test_ir_parser_marks_sk_portfolio_metrics_separately() -> None:
    article = RawArticle(
        url="https://example.com/sk-inc-ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content=(
            "[PAGE 1]\n"
            "2026년 1분기\n"
            "SK Inc. at a Glance Portfolio\n"
            "SK바이오팜 매출액 1조원, SK스퀘어 영업이익 500억원"
        ),
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
    )

    parsed = IRParser().parse_article(article)
    candidates_by_type = {candidate["type"]: candidate for candidate in parsed["candidates"]}

    assert candidates_by_type["revenue_total"]["metric_scope"] == "portfolio_company"
    assert candidates_by_type["revenue_total"]["entity_name"] == "sk_biopharmaceuticals"
    assert candidates_by_type["operating_profit"]["metric_scope"] == "portfolio_company"


def test_ir_parser_skips_sk_unlisted_subsidiary_appendix_table() -> None:
    article = RawArticle(
        url="https://example.com/sk-inc-ir.pdf",
        title="SK AX 2026년 1분기 IR Presentation",
        content="",
        source_name="ir_pdf",
        published_at=datetime(2026, 4, 30),
        peer_id="sk_ax",
        source_type="ir",
        content_type="pdf",
        extra={
            "date_info": {"year": 2026, "quarter": 1},
            "pdf_page_blocks": [
                {
                    "page": 29,
                    "blocks": [
                        {"text": "[Appendix] 주요비상장자회사분기별실적"},
                        {"text": "(단위: 십억원)"},
                        {"text": "구분 1Q23 2Q23 3Q23 4Q23 1Q24 2Q24"},
                        {"text": "영업이익 114 70 39 58 281 42"},
                    ],
                }
            ],
        },
    )

    parsed = IRParser().parse_article(article)
    table_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "ir_table_matrix"
    ]

    assert table_candidates == []


def test_ir_rule_based_business_signals_require_directional_evidence() -> None:
    article = {
        "id": 201,
        "company": ["test_peer"],
        "title": "테스트사 IR",
        "url": "https://example.com/ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c1",
                "chunk_index": 1,
                "page": 3,
                "section_key": "business",
                "text": (
                    "클라우드 매출은 1분기 주요 실적 지표입니다. "
                    "클라우드 MSP 수요 확대와 GPU 인프라 구축으로 성장세가 강화되었습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "test_peer"},
    )

    assert len(signals) == 1
    assert {signal["signal_type"] for signal in signals} == {"growth"}
    assert all(
        "클라우드 매출은 1분기 주요 실적 지표" not in signal["evidence_text"] for signal in signals
    )
    assert all(
        signal["extraction_method"] == "ir_parser.document_chunks.rule_based.v2"
        for signal in signals
    )


def test_ir_rule_based_business_signals_skip_appendix_portfolio_sentences() -> None:
    article = {
        "id": 202,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c1",
                "chunk_index": 1,
                "page": 29,
                "section_key": "business",
                "text": (
                    "[Appendix] 주요비상장자회사분기별실적에서 자회사 매출이 증가했습니다. "
                    "SK AX AI Transformation 사업은 고객 수요 확대로 성장했습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert len(signals) == 1
    assert signals[0]["business_area"] == "ai_ax"
    assert signals[0]["signal_type"] == "growth"
    assert "주요비상장자회사" not in signals[0]["evidence_text"]


def test_ir_rule_based_business_signals_include_sk_ax_highlight_sentences() -> None:
    article = {
        "id": 203,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c1",
                "chunk_index": 1,
                "page": 9,
                "section_key": "business",
                "text": (
                    "신규 AI DX 프로젝트 수주 견조한 가운데, 기저 효과로 매출 감소했습니다. "
                    "포트폴리오 Mix 변화 및 생산성 향상에 따른 마진 개선 효과로 "
                    "영업이익은 증가했습니다. "
                    "AI Transformation 및 DT 기반의 고부가 비즈니스 모델로 개편 진행 중입니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    evidence = " ".join(signal["evidence_text"] for signal in signals)
    signal_types = {signal["signal_type"] for signal in signals}
    assert "수주 견조" in evidence
    assert "마진 개선" in evidence
    assert "AI Transformation" in evidence
    assert {"orders_pipeline", "growth", "strategy"} <= signal_types


def test_ir_business_signals_do_not_inherit_cloud_section_for_sk_ax_it_service() -> None:
    article = {
        "id": 204,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c1",
                "chunk_index": 1,
                "page": 4,
                "section_key": "cloud",
                "text": (
                    "IT서비스 EBITDA Highlights 감가상각비와 영업이익이 감소했습니다. "
                    "신규 AI DX 프로젝트 수주 견조한 가운데 기저효과로 매출이 감소했습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals
    business_areas = {signal["business_area"] for signal in signals}
    assert "cloud" not in business_areas
    assert "company_total" in business_areas


def test_ir_business_signals_skip_semiconductor_chunk_without_sk_ax_context() -> None:
    article = {
        "id": 205,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c2",
                "chunk_index": 2,
                "page": 5,
                "section_key": "cloud",
                "text": (
                    "FY25, 반도체 Fab 등 주요 프로젝트 착공으로 전년 대비 매출 성장했습니다. "
                    "프로젝트 공정률 상승 기반으로 수주 확대가 이어지고 있습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals == []


def test_ir_business_signal_summary_strips_numeric_table_blob() -> None:
    article = {
        "id": 206,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c3",
                "chunk_index": 3,
                "page": 6,
                "section_key": "business",
                "text": (
                    "1Q26 경영실적현황: IT서비스부문(SK AX) "
                    "신규 AI DX 사업수주 확대 및 마진 개선이 지속되고 있습니다. "
                    "매출 530 586 영업이익 31 29 EBITDA 63 68 영업이익률 4.9% 7.6% 9.4%."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals
    summary = signals[0]["summary"]
    assert "530" not in summary
    assert "586" not in summary
    assert "신규 AI DX 사업수주 확대" in summary


def test_ir_business_signals_ignore_table_like_rows_and_keep_narrative_sentences() -> None:
    article = {
        "id": 207,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c4",
                "chunk_index": 4,
                "page": 7,
                "section_key": "business",
                "text": (
                    "신규 AI DX 프로젝트 수주 확대와 마진 개선이 지속되고 있습니다.\n"
                    "매출 530 586 영업이익 31 29 EBITDA 63 68 영업이익률 4.9% 7.6% 9.4%\n"
                    "AI Transformation 기반의 고부가 사업모델 개편을 추진하고 있습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals
    evidence = " ".join(signal["evidence_text"] for signal in signals)
    assert "매출 530 586" not in evidence
    assert "영업이익률 4.9%" not in evidence
    assert "수주 확대" in evidence
    assert "사업모델 개편" in evidence


def test_ir_business_signals_keep_page_when_sk_ax_context_exists_with_affiliate_mentions() -> None:
    article = {
        "id": 208,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c5",
                "chunk_index": 5,
                "page": 8,
                "section_key": "business",
                "text": (
                    "SK에코플랜트는 반도체 Fab 프로젝트 착공으로 매출이 성장했습니다. "
                    "SK AX는 AI Transformation 수요 확대로 IT서비스 수주가 증가했습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals
    evidence = " ".join(signal["evidence_text"] for signal in signals)
    assert "SK AX" in evidence
    assert any(signal["business_area"] == "company_total" for signal in signals)


def test_ir_business_signals_skip_non_sk_ax_affiliate_chunks() -> None:
    article = {
        "id": 209,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c6",
                "chunk_index": 6,
                "page": 5,
                "section_key": "business",
                "text": (
                    "SK에코플랜트 EBITDA Highlights Hi-tech 및 Asset Lifecycle 실적호조로 "
                    "전사매출 증가(YoY +99.3%)"
                ),
            },
            {
                "chunk_id": "c7",
                "chunk_index": 7,
                "page": 9,
                "section_key": "business",
                "text": (
                    "1Q26 경영실적현황: IT서비스부문(SK AX) "
                    "신규 AI DX 프로젝트 수주 견조한 가운데 기저 효과로 매출 감소했습니다. "
                    "AI Transformation 기반의 고부가 비즈니스 모델로 개편 진행 중입니다."
                ),
            },
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert signals
    pages = {signal["source_page"] for signal in signals}
    evidence = " ".join(signal["evidence_text"] for signal in signals)
    assert pages == {9}
    assert "SK에코플랜트" not in evidence
    assert "IT서비스부문(SK AX)" in evidence
    assert all(signal["business_area"] == "company_total" for signal in signals)


def test_ir_business_signals_keep_one_representative_signal_per_evidence() -> None:
    article = {
        "id": 210,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "document_chunks": [
            {
                "chunk_id": "c8",
                "chunk_index": 8,
                "page": 9,
                "section_key": "business",
                "text": (
                    "1Q26 경영실적현황: IT서비스부문(SK AX) "
                    "신규 AI DX 사업 수주 확대와 마진 개선이 지속되고 있습니다."
                ),
            }
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    expected_evidence = (
        "1Q26 경영실적현황: IT서비스부문(SK AX) "
        "신규 AI DX 사업 수주 확대와 마진 개선이 지속되고 있습니다."
    )
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "orders_pipeline"
    assert signals[0]["evidence_text"] == expected_evidence


def test_ir_llm_business_signals_keep_summary_but_dedupe_same_evidence() -> None:
    article = {
        "id": 211,
        "company": ["sk_ax"],
        "title": "SK AX IR",
        "url": "https://example.com/sk-ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "llm_business_signals": [
            {
                "business_area": "company_total",
                "signal_type": "investment",
                "sentiment": "negative",
                "summary": "데이터센터 매각 영향이 있었다.",
                "evidence_text": (
                    "해외 프로젝트 종료 및 판교 데이터센터 매각 영향으로 Top-line 축소"
                ),
                "source_page": 9,
            },
            {
                "business_area": "company_total",
                "signal_type": "risk",
                "sentiment": "negative",
                "summary": "해외 프로젝트 종료와 데이터센터 매각으로 매출이 감소했다.",
                "evidence_text": (
                    "해외 프로젝트 종료 및 판교 데이터센터 매각 영향으로 Top-line 축소"
                ),
                "source_page": 9,
            },
        ],
    }

    signals = _business_signals_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "sk_ax"},
    )

    assert len(signals) == 1
    assert signals[0]["signal_type"] == "risk"
    assert signals[0]["summary"] == "해외 프로젝트 종료와 데이터센터 매각으로 매출이 감소했다."
    assert (
        signals[0]["evidence_text"]
        == "해외 프로젝트 종료 및 판교 데이터센터 매각 영향으로 Top-line 축소"
    )


def test_ir_metrics_keep_one_representative_candidate_for_report_period() -> None:
    article = {
        "id": 204,
        "company": ["test_peer"],
        "title": "테스트사 IR",
        "url": "https://example.com/ir.pdf",
        "source_name": "ir_pdf",
        "extra": {"period": "2026Q1", "period_year": 2026, "period_quarter": 1},
    }
    parser_result = {
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "candidates": [
            {
                "type": "revenue_total",
                "value_krwbn": 1000,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2025Q4",
                "source": "ir_table_matrix",
                "confidence": 0.9,
            },
            {
                "type": "revenue_total",
                "value_krwbn": 1200,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2026Q1",
                "source": "ir_parser_text",
                "confidence": 0.85,
            },
            {
                "type": "revenue_total",
                "value_krwbn": 1234,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "period": "2026Q1",
                "source": "ir_table_matrix",
                "confidence": 0.88,
            },
        ],
    }

    metrics = _metrics_from_parser_result(
        article,
        parser_result,
        {"period": "2026Q1", "peer_id": "test_peer"},
    )

    assert len(metrics) == 1
    assert metrics[0]["period"] == "2026Q1"
    assert metrics[0]["metric_name"] == "revenue_total"
    assert metrics[0]["value_krwbn"] == 1234
    assert metrics[0]["extraction_method"] == "ir_parser.table_matrix"


def test_dart_parser_parses_dart_crawler_article() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260506000123",
        title="분기보고서 (2025.09)",
        content="제3분기 연결 기준 매출액 1,234억원 영업이익 56억원",
        source_name="dart",
        published_at=datetime(2025, 11, 14),
        peer_id="test_peer",
        source_type="dart",
        content_type="api",
        extra={
            "corp_code": "00123456",
            "rcept_no": "20260506000123",
            "report_name": "분기보고서 (2025.09)",
            "rcept_dt": "20251114",
            "disclosure_type": "A",
            "disclosure_type_label": "regular",
            "document_fetched": True,
            "content_chars": 38,
        },
    )

    parsed = DartParser().parse_article(article)

    assert parsed["ok"] is True
    assert parsed["period"] == "2025Q3"
    assert parsed["period_type"] == "quarter"
    assert parsed["rcept_no"] == "20260506000123"
    assert parsed["revenue_total_krwbn"] == 1234
    assert parsed["operating_profit_krwbn"] == 56
    assert parsed["financial_record"]["dart_rcept_no"] == "20260506000123"


def test_dart_parser_parses_dart_statement_table_amounts() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260310002989",
        title="사업보고서 (2025.12)",
        content=(
            "포 괄 손 익 계 산 서 삼성에스디에스주식회사 (단위: 원) "
            "과 목 주석 제 41 (당) 기 제 40 (전) 기 "
            "매출액 23,31 5,464,637,894,164 5,447,352,234,920 "
            "영업이익 520,433,363,884 507,423,177,759 "
            "연 결 포 괄 손 익 계 산 서 삼성에스디에스주식회사와 그 종속기업 (단위: 원) "
            "과 목 주석 제 41 (당) 기 제 40 (전) 기 "
            "매출액 4,25,33 13,929,868,497,711 13,828,232,033,800 "
            "매출원가 26,33 11,849,842,590,785 11,815,899,879,484 "
            "영업이익 4 957,102,744,609 911,096,907,001"
        ),
        source_name="dart",
        published_at=datetime(2026, 3, 10),
        peer_id="samsung_sds",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260310002989",
            "report_name": "사업보고서 (2025.12)",
            "document_fetched": True,
        },
    )

    parsed = DartParser().parse_article(article)

    assert parsed["period"] == "2025Q4"
    assert parsed["period_type"] == "annual"
    assert parsed["revenue_total_krwbn"] == 139298.68497711
    assert parsed["operating_profit_krwbn"] == 9571.02744609


def test_dart_parser_builds_section_tree_for_late_sections() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260310000001",
        title="사업보고서 (2025.12)",
        content=(
            "I. 회사의 개요\n"
            "1. 회사의 개요\n"
            "가. 회사의 법적ㆍ상업적 명칭\n"
            "회사의 명칭은 테스트 주식회사입니다.\n"
            "II. 사업의 내용\n"
            "1. 사업의 개요\n"
            "클라우드와 AI 사업을 영위합니다.\n"
            "XII. 상세표\n"
            "1. 연결대상 종속회사 현황(상세)\n"
            "상세표 본문입니다.\n"
            "전문가의 확인\n"
            "1. 전문가의 확인\n"
            "전문가 확인 본문입니다."
        ),
        source_name="dart",
        published_at=datetime(2026, 3, 10),
        peer_id="test_peer",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260310000001",
            "report_name": "사업보고서 (2025.12)",
            "document_fetched": True,
        },
    )

    parsed = DartParser().parse_article(article)

    assert "detailed_tables" in parsed["sections"]
    assert "expert_confirmation" in parsed["sections"]
    section_keys = [node["section_key"] for node in parsed["section_tree"]]
    assert section_keys == [
        "company_overview",
        "business",
        "detailed_tables",
        "expert_confirmation",
    ]
    assert parsed["section_tree"][0]["children"][0]["children"][0]["title"] == (
        "회사의 법적ㆍ상업적 명칭"
    )


def test_dart_parser_classifies_and_normalizes_financial_statement_tables() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260310000002",
        title="사업보고서 (2025.12)",
        content="III. 재무에 관한 사항\n1. 요약재무정보\n연결 포괄손익계산서",
        source_name="dart",
        published_at=datetime(2026, 3, 10),
        peer_id="test_peer",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260310000002",
            "report_name": "사업보고서 (2025.12)",
            "document_fetched": True,
            "tables": [
                {
                    "table_index": 3,
                    "title": "연결 포괄손익계산서",
                    "row_count": 4,
                    "column_count": 3,
                    "rows": [
                        ["과 목", "제 41 (당) 기", "제 40 (전) 기"],
                        ["매출액", "13,929,868", "13,828,232"],
                        ["영업이익", "957,102", "911,096"],
                        ["당기순이익", "730,000", "700,000"],
                    ],
                    "text": (
                        "연결 포괄손익계산서 (단위: 백만원)\n"
                        "매출액 | 13,929,868 | 13,828,232\n"
                        "영업이익 | 957,102 | 911,096\n"
                        "당기순이익 | 730,000 | 700,000"
                    ),
                }
            ],
        },
    )

    parsed = DartParser().parse_article(article)

    assert parsed["classified_tables"][0]["table_type"] == "income_statement"
    assert parsed["classified_tables"][0]["statement_scope"] == "consolidated"
    statement = parsed["financial_statements"][0]
    assert statement["table_type"] == "income_statement"
    assert statement["unit"] == "백만원"
    rows_by_metric = {row["metric_key"]: row for row in statement["rows"]}
    assert rows_by_metric["revenue_total"]["current_value_krwbn"] == 139298.68
    assert [value["period"] for value in rows_by_metric["revenue_total"]["values"]] == [
        "2025Q4",
        "2024Q4",
    ]
    assert rows_by_metric["revenue_total"]["values"][1]["is_historical"] is True
    assert rows_by_metric["operating_profit"]["current_value_krwbn"] == 9571.02
    revenue_candidate = {candidate["type"]: candidate for candidate in parsed["candidates"]}[
        "revenue_total"
    ]
    assert revenue_candidate["source"] == "structured_table"
    assert revenue_candidate["unit"] == "백만원"
    assert revenue_candidate["value_krw"] == 13929868000000.0
    assert revenue_candidate["confidence"] >= 0.9
    assert parsed["financial_record"]["metric_details"]["revenue_total"]["table_index"] == 3
    assert any(
        fact["fact_type"] == "financial_metric" and fact["metric"] == "revenue_total"
        for fact in parsed["analysis_facts"]
    )


def test_dart_parser_extracts_business_segment_sales_table() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260515000001",
        title="삼성에스디에스 분기보고서 (2026.03)",
        content="II. 사업의 내용\n2. 주요 제품 및 서비스\n",
        source_name="dart",
        published_at=datetime(2026, 5, 15),
        peer_id="samsung_sds",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260515000001",
            "report_name": "분기보고서 (2026.03)",
            "document_fetched": True,
            "tables": [
                {
                    "table_index": 7,
                    "title": "주요 제품 등의 현황",
                    "row_count": 7,
                    "column_count": 6,
                    "rows": [
                        ["사업부문", "품목", "2026년 1분기", "2025년 1분기", "2025년", "2024년"],
                        ["IT서비스", "클라우드", "690,866\n(20.6%)", "652,889\n(18.7%)"],
                        ["", "SI", "242,714\n(7.2%)", "235,593\n(6.8%)"],
                        ["", "ITO", "676,929\n(20.2%)", "711,860\n(20.4%)"],
                        ["", "소계", "1,610,509\n(48.0%)", "1,600,342\n(45.9%)"],
                        ["물류", "", "1,742,409\n(52.0%)", "1,889,422\n(54.1%)"],
                        ["합계", "", "3,352,918\n(100.0%)", "3,489,764\n(100.0%)"],
                    ],
                    "text": (
                        "2. 주요 제품 및 서비스\n"
                        "(단위 : 백만원)\n"
                        "사업부문 | 품목 | 매출액\n"
                        "IT서비스 | 클라우드 | 690,866 (20.6%)\n"
                        "IT서비스 | SI | 242,714 (7.2%)\n"
                        "IT서비스 | ITO | 676,929 (20.2%)\n"
                        "IT서비스 | 소계 | 1,610,509 (48.0%)\n"
                        "물류 |  | 1,742,409 (52.0%)\n"
                        "합계 |  | 3,352,918 (100.0%)"
                    ),
                }
            ],
        },
    )

    parsed = DartParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate["type"] == "revenue_total"
        and candidate.get("source") == "business_segment_table"
    ]

    assert [candidate["segment_label"] for candidate in revenue_candidates] == [
        "클라우드",
        "SI",
        "ITO",
        "IT서비스",
        "물류",
    ]
    assert [candidate["business_area"] for candidate in revenue_candidates] == [
        "클라우드",
        "SI",
        "ITO",
        "IT서비스",
        "물류",
    ]
    assert [candidate["standard_business_area"] for candidate in revenue_candidates] == [
        "cloud",
        "enterprise_it",
        "enterprise_it",
        "enterprise_it",
        "logistics",
    ]
    assert all(candidate["metric_scope"] == "segment" for candidate in revenue_candidates)
    assert revenue_candidates[0]["value_krwbn"] == 6908.66
    assert revenue_candidates[-1]["value_krwbn"] == 17424.09


def test_dart_parser_maps_sk_inc_segment_to_sk_ax_context() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260515000002",
        title="SK 분기보고서 (2026.03)",
        content="II. 사업의 내용\n2. 주요 제품 및 서비스\n",
        source_name="dart",
        published_at=datetime(2026, 5, 15),
        peer_id="sk_ax",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260515000002",
            "report_name": "분기보고서 (2026.03)",
            "document_fetched": True,
            "tables": [
                {
                    "table_index": 5,
                    "title": "주요 제품 및 서비스",
                    "row_count": 3,
                    "column_count": 4,
                    "rows": [
                        ["사업부문", "품목", "2026년 1분기", "2025년 1분기"],
                        ["SK주식회사", "C&C", "800,000", "760,000"],
                        ["합계", "", "800,000", "760,000"],
                    ],
                    "text": (
                        "주요 제품 및 서비스 (단위 : 백만원)\n"
                        "사업부문 | 품목 | 매출액\n"
                        "SK주식회사 | C&C | 800,000"
                    ),
                }
            ],
        },
    )

    parsed = DartParser().parse_article(article)
    revenue_candidates = [
        candidate
        for candidate in parsed["candidates"]
        if candidate["type"] == "revenue_total"
        and candidate.get("source") == "business_segment_table"
    ]

    assert len(revenue_candidates) == 1
    assert revenue_candidates[0]["business_area"] == "C&C"
    assert revenue_candidates[0]["standard_business_area"] == "sk_ax"
    assert revenue_candidates[0]["metric_scope"] == "segment"


def test_dart_parser_preserves_unknown_business_segment_label() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260515000003",
        title="임의회사 분기보고서 (2026.03)",
        content="II. 사업의 내용\n2. 주요 제품 및 서비스\n",
        source_name="dart",
        published_at=datetime(2026, 5, 15),
        peer_id="unknown_peer",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260515000003",
            "report_name": "분기보고서 (2026.03)",
            "document_fetched": True,
            "tables": [
                {
                    "table_index": 9,
                    "title": "주요 제품 및 서비스",
                    "rows": [
                        ["사업부문", "품목", "2026년 1분기"],
                        ["플랫폼운영", "구독서비스", "123,456"],
                    ],
                    "text": (
                        "주요 제품 및 서비스 (단위 : 백만원)\n"
                        "사업부문 | 품목 | 매출액\n"
                        "플랫폼운영 | 구독서비스 | 123,456"
                    ),
                }
            ],
        },
    )

    parsed = DartParser().parse_article(article)
    candidate = next(
        candidate
        for candidate in parsed["candidates"]
        if candidate.get("source") == "business_segment_table"
    )

    assert candidate["segment_label"] == "구독서비스"
    assert candidate["business_area"] == "구독서비스"
    assert candidate["standard_business_area"] is None
    assert candidate["metric_scope"] == "segment"


def test_dart_parser_enriches_topic_chunks_for_agent_analysis() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260310000003",
        title="사업보고서 (2025.12)",
        content=(
            "II. 사업의 내용\n"
            "1. 사업의 개요\n"
            "회사는 클라우드, 생성형 AI, AI 플랫폼 기반의 AX 사업을 확대하고 있습니다. "
            "제조 AX와 스마트팩토리 고객 사례를 중심으로 서비스를 고도화합니다. "
            "반복 업무 자동화와 보안 운영 최적화도 추진합니다."
        ),
        source_name="dart",
        published_at=datetime(2026, 3, 10),
        peer_id="test_peer",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20260310000003",
            "report_name": "사업보고서 (2025.12)",
            "document_fetched": True,
        },
    )

    parsed = DartParser().parse_article(article)

    chunk = parsed["document_chunks"][0]
    assert chunk["peer_id"] == "test_peer"
    assert chunk["period"] == "2025Q4"
    assert chunk["rcept_no"] == "20260310000003"
    assert "생성형 AI" in chunk["matched_keywords"]
    assert any(
        fact["fact_type"] == "business_context" and fact["topic"] == "ax"
        for fact in parsed["analysis_facts"]
    )


def test_dart_structured_table_header_inference_keeps_period_headers() -> None:
    rows = [
        ["과 목", "제 41 (당) 기", "제 40 (전) 기"],
        ["매출액", "13,929,868", "13,828,232"],
    ]

    assert _infer_header_rows(rows) == [["과 목", "제 41 (당) 기", "제 40 (전) 기"]]


def test_dart_parser_extracts_event_disclosure_fields_for_share_buyback() -> None:
    article = RawArticle(
        url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250828000123",
        title="주요사항보고서(자기주식 취득 결정)",
        content=(
            "주요사항보고서(자기주식 취득 결정)\n"
            "취득예정주식(주) 보통주식 498,494 기타주식 152,412\n"
            "취득예정금액(원) 10,211,738,640\n"
            "취득예상기간 2025.08.28 - 2025.11.27\n"
            "11. 기타 투자판단에 참고할 사항\n"
            "- 상기 1. 취득예정주식(주) 및 2. 취득예정금액(원)의 기타주식은 2우선주임\n"
            "- 상기 4. 보유예상기간과 관련하여 자기주식 취득 완료후 소각할 예정임"
        ),
        source_name="dart",
        published_at=datetime(2025, 8, 28),
        peer_id="test_peer",
        source_type="dart",
        content_type="api",
        extra={
            "rcept_no": "20250828000123",
            "report_name": "주요사항보고서(자기주식 취득 결정)",
            "document_fetched": True,
        },
    )

    parsed = DartParser().parse_article(article)

    assert parsed["disclosure_category"] == "event_disclosure"
    assert parsed["event_type"] == "share_buyback_decision"
    assert parsed["event_fields"]["target_shares_common"] == 498494
    assert parsed["event_fields"]["target_shares_preferred"] == 152412
    assert parsed["event_fields"]["target_amount_krw"] == 10211738640
    assert parsed["event_fields"]["acquisition_period"] == "2025.08.28 - 2025.11.27"
    assert parsed["financial_record"]["event_type"] == "share_buyback_decision"


def test_parser_quality_passes_dart_document_after_parser() -> None:
    item = {
        "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260506000123",
        "title": "분기보고서 (2025.09)",
        "content": "제3분기 연결 기준 매출액 1,234억원 영업이익 56억원 " * 3,
        "source_name": "dart",
        "source_type": "dart",
        "content_type": "api",
        "company": ["test_peer"],
        "extra": {
            "rcept_no": "20260506000123",
            "report_name": "분기보고서 (2025.09)",
            "rcept_dt": "20251114",
            "document_fetched": True,
        },
    }

    result, ok, reason = analyze_parser_quality_article(item)

    assert ok is True
    assert reason is None
    assert result["parser_quality_label"] == "pass"
    assert result["parser_result"]["period"] == "2025Q3"


def test_document_parser_router_routes_dart_to_dart_parser() -> None:
    item = {
        "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260506000123",
        "title": "분기보고서 (2025.09)",
        "content": "제3분기 연결 기준 매출액 1,234억원 영업이익 56억원",
        "source_name": "dart",
        "source_type": "dart",
        "company": ["test_peer"],
        "extra": {
            "rcept_no": "20260506000123",
            "report_name": "분기보고서 (2025.09)",
            "document_fetched": True,
        },
    }

    parsed = DocumentParserRouter().parse_article(item)

    assert parsed["source"] == "dart"
    assert parsed["period"] == "2025Q3"
    assert parsed["financial_record"]["dart_rcept_no"] == "20260506000123"


def test_document_parser_router_preserves_industry_report_metadata() -> None:
    item = {
        "url": "https://spri.kr/posts/view/23967",
        "title": "산업혁신을 이끄는 Vertical SaaS",
        "content": "Vertical SaaS 시장 성장과 정책 시사점",
        "source_name": "SPRi",
        "source_type": "trend_report",
        "content_type": "pdf",
        "extra": {
            "matched_sectors": ["ax"],
            "pdf_url": "https://www.spri.kr/download/23830",
            "page_count": 21,
        },
    }

    parsed = DocumentParserRouter().parse_article(item)

    assert parsed["ok"] is True
    assert parsed["source"] == "industry_trend"
    assert parsed["source_name"] == "SPRi"
    assert parsed["metadata"]["matched_sectors"] == ["ax"]


def test_document_parser_router_parses_securities_report() -> None:
    item = {
        "url": "https://example.com/report.pdf",
        "title": "[미래에셋증권] 삼성SDS 2026Q1 Review",
        "content": (
            "투자의견 BUY\n"
            "목표주가 220,000원\n"
            "현재주가 180,000원\n"
            "2026Q1 실적은 시장 기대치에 부합.\n"
            "클라우드와 AI 매출 성장이 지속된다."
        ),
        "source_name": "naver_research",
        "source_type": "securities_report",
        "publisher": "미래에셋증권",
        "company": ["samsung_sds"],
        "extra": {
            "firm": "미래에셋증권",
            "pdf_url": "https://example.com/report.pdf",
            "item_code": "018260",
            "pdf_pages": 12,
        },
    }

    parsed = DocumentParserRouter().parse_article(item)

    assert parsed["source"] == "securities_report"
    assert parsed["report_firm"] == "미래에셋증권"
    assert parsed["investment_opinion"] == "BUY"
    assert parsed["target_price_krw"] == 220000
    assert parsed["current_price_krw"] == 180000
    assert parsed["period"] == "2026Q1"
    assert parsed["metadata"]["item_code"] == "018260"
    assert len(parsed["highlights"]) >= 1


def test_securities_report_parser_uses_front_pages_for_opinion_and_prices() -> None:
    item = {
        "url": "https://example.com/report.pdf",
        "title": "[한화투자증권] 1Q26 Review : 하이닉스와 연결된 현금흐름",
        "content": (
            "[PAGE 1]\n"
            "투자의견 BUY 유지\n"
            "목표주가 810,000원\n"
            "현재주가 650,000원\n"
            "1Q26 Review 실적은 예상치를 상회했다.\n"
            "[PAGE 7]\n"
            "투자의견 및 목표주가 변동추이\n"
            "2024 2025 2026 81 9 괴리율 평균 최고 최저\n"
        ),
        "source_name": "naver_research",
        "source_type": "securities_report",
        "publisher": "한화투자증권",
        "company": ["sk_ax"],
        "extra": {"firm": "한화투자증권"},
    }

    parsed = DocumentParserRouter().parse_article(item)

    assert parsed["investment_opinion"] == "BUY"
    assert parsed["target_price_krw"] == 810000
    assert parsed["current_price_krw"] == 650000
    assert parsed["period"] == "2026Q1"
    assert all("목표주가 변동추이" not in chunk["text"] for chunk in parsed["document_chunks"])


def test_securities_report_parser_rejects_trailing_price_history_noise() -> None:
    item = {
        "url": "https://example.com/report.pdf",
        "title": "[DS투자증권] 단기 노이즈보다 중기 모멘텀에 주목",
        "content": (
            "[PAGE 1]\n"
            "목표주가 변동추이 및 투자의견 비율\n"
            "목표주가 2024 원\n"
            "투자의견 및 목표주가 변동추이\n"
            "매수 유지하고 목표주가 변동추이 9 81\n"
        ),
        "source_name": "naver_research",
        "source_type": "securities_report",
        "publisher": "DS투자증권",
        "company": ["hyundai_autoever"],
        "extra": {"firm": "DS투자증권"},
    }

    parsed = DocumentParserRouter().parse_article(item)

    assert parsed["investment_opinion"] is None
    assert parsed["target_price_krw"] is None


def test_securities_report_preprocess_metadata_patch_includes_analysis_fields() -> None:
    parser_result = {
        "parser": "securities_report_parser",
        "source_type": "securities_report",
        "report_firm": "미래에셋증권",
        "investment_opinion": "BUY",
        "target_price_krw": 220000,
        "current_price_krw": 180000,
        "period": "2026Q1",
        "topics": ["ai"],
        "topic_signals": {"ai": ["AI"]},
        "sections": [{"section_key": "summary"}],
        "document_chunks": [
            {
                "chunk_id": "summary:1",
                "text": "AI 데이터센터와 클라우드 수요가 성장한다." * 80,
            }
        ],
    }

    patch = _parsed_document_metadata_patch(
        source_type="securities_report",
        parser_result=parser_result,
        parser_quality_score=0.8,
        parser_quality_label="pass",
        parser_quality_reason="문서 파싱 품질 기준 통과",
    )

    assert patch["report_firm"] == "미래에셋증권"
    assert patch["investment_opinion"] == "BUY"
    assert patch["target_price_krw"] == 220000
    assert patch["current_price_krw"] == 180000
    assert patch["period"] == "2026Q1"
    assert patch["securities_report_sections"] == [{"section_key": "summary"}]
    assert patch["securities_report_document_chunks"][0]["text_is_truncated_for_metadata"] is True
    assert len(patch["securities_report_document_chunks"][0]["text"]) == 1200
