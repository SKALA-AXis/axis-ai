from datetime import datetime

from src.parsers.dart_parser import DartParser
from src.parsers.ir_parser import IRParser
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.parser_router import DocumentParserRouter
from src.crawler.base import RawArticle


def test_ir_parser_parses_ir_crawler_article() -> None:
    article = RawArticle(
        url="https://example.com/ir.pdf",
        title="테스트사 2026년 1분기 IR Presentation",
        content="[PAGE 1]\n2026년 1분기\n매출액 9,365억원\n영업이익 300억원",
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
                        {"text": "2026년 1분기"},
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
