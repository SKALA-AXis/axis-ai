from datetime import datetime

from src.parsers.dart_parser import DartParser
from src.parsers.ir_parser import IRParser
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.parser_router import DocumentParserRouter
from src.crawler.base import RawArticle
from src.crawler.sources.dart import _infer_header_rows


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
    assert rows_by_metric["operating_profit"]["current_value_krwbn"] == 9571.02
    revenue_candidate = {
        candidate["type"]: candidate for candidate in parsed["candidates"]
    }["revenue_total"]
    assert revenue_candidate["source"] == "structured_table"
    assert revenue_candidate["unit"] == "백만원"
    assert revenue_candidate["value_krw"] == 13929868000000.0
    assert revenue_candidate["confidence"] >= 0.9
    assert parsed["financial_record"]["metric_details"]["revenue_total"]["table_index"] == 3
    assert any(
        fact["fact_type"] == "financial_metric" and fact["metric"] == "revenue_total"
        for fact in parsed["analysis_facts"]
    )


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
