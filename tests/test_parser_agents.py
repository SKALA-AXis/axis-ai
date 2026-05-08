from datetime import datetime

from src.agents.dart_parser_agent import DartParserAgent
from src.agents.ir_parser_agent import IRParserAgent
from src.agents.parser_agent import ParserAgent
from src.agents.parser_quality_agent import analyze_parser_quality_article
from src.crawler.base import RawArticle


def test_ir_parser_agent_parses_ir_crawler_article() -> None:
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

    parsed = IRParserAgent().parse_article(article)

    assert parsed["ok"] is True
    assert parsed["peer_id"] == "test_peer"
    assert parsed["period"] == "2026Q1"
    assert parsed["revenue_total_krwbn"] == 9365
    assert parsed["operating_profit_krwbn"] == 300
    assert parsed["financial_record"]["ir_page"] == 1


def test_dart_parser_agent_parses_dart_crawler_article() -> None:
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

    parsed = DartParserAgent().parse_article(article)

    assert parsed["ok"] is True
    assert parsed["period"] == "2025Q3"
    assert parsed["period_type"] == "quarter"
    assert parsed["rcept_no"] == "20260506000123"
    assert parsed["revenue_total_krwbn"] == 1234
    assert parsed["operating_profit_krwbn"] == 56
    assert parsed["financial_record"]["dart_rcept_no"] == "20260506000123"


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


def test_common_parser_agent_routes_dart_to_dart_parser() -> None:
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

    parsed = ParserAgent().parse_article(item)

    assert parsed["source"] == "dart"
    assert parsed["period"] == "2025Q3"
    assert parsed["financial_record"]["dart_rcept_no"] == "20260506000123"


def test_common_parser_agent_preserves_industry_report_metadata() -> None:
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

    parsed = ParserAgent().parse_article(item)

    assert parsed["ok"] is True
    assert parsed["source"] == "industry_trend"
    assert parsed["source_name"] == "SPRi"
    assert parsed["metadata"]["matched_sectors"] == ["ax"]
