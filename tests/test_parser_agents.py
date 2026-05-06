from datetime import datetime

from src.agents.dart_parser_agent import DartParserAgent
from src.agents.ir_parser_agent import IRParserAgent
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
