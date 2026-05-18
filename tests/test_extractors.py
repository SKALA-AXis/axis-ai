from src.extractors.dart_analysis_extractor import (
    business_signals_from_dart,
    financial_metrics_from_dart,
)


def test_dart_financial_metrics_from_statement_rows() -> None:
    article = {
        "id": 10,
        "title": "사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["test_peer"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "test_peer",
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "period_type": "quarter",
        "financial_statements": [
            {
                "table_index": 3,
                "title": "연결 포괄손익계산서",
                "table_type": "income_statement",
                "statement_scope": "consolidated",
                "unit": "백만원",
                "rows": [
                    {
                        "metric_key": "revenue_total",
                        "label": "매출액",
                        "current_value_krwbn": 139298.68,
                    },
                    {
                        "metric_key": "operating_profit",
                        "label": "영업이익",
                        "current_value_krwbn": 9571.02,
                    },
                ],
            }
        ],
    }

    metrics = financial_metrics_from_dart(article, parser_result)

    assert [metric["metric_name"] for metric in metrics] == [
        "revenue_total",
        "operating_profit",
    ]
    assert metrics[0]["source_type"] == "dart"
    assert metrics[0]["value_krw"] == 13929868000000.0
    assert metrics[0]["source_table_uid"] == "dart-table-3"
    assert metrics[0]["confidence"] >= 0.9


def test_dart_business_signals_from_document_chunks() -> None:
    article = {
        "id": 11,
        "title": "사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["test_peer"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "test_peer",
        "period": "2026Q1",
        "period_year": 2026,
        "period_quarter": 1,
        "period_type": "quarter",
        "document_chunks": [
            {
                "chunk_id": "business:1",
                "chunk_index": 1,
                "section_key": "business",
                "section_title": "사업의 내용",
                "text": (
                    "회사는 클라우드 MSP 사업과 생성형 AI 기반 AX 사업을 확대하고 있습니다. "
                    "ERP AI agent와 SCM 자동화 서비스를 고도화하며 고객사의 전환 수요에 대응합니다."
                ),
            }
        ],
    }

    signals = business_signals_from_dart(article, parser_result)

    assert signals
    assert signals[0]["source_type"] == "dart"
    assert {signal["business_area"] for signal in signals} <= {"cloud", "ai_ax"}
    assert {signal["signal_type"] for signal in signals} <= {
        "growth",
        "strategy",
        "efficiency",
    }
    assert any("클라우드" in signal["evidence_text"] for signal in signals)


def test_dart_business_signals_follow_section_policy() -> None:
    article = {
        "id": 12,
        "title": "사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["test_peer"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "test_peer",
        "period": "2026Q1",
        "document_chunks": [
            {
                "chunk_id": "business:1",
                "chunk_index": 1,
                "section_key": "business",
                "section_title": "사업의 내용",
                "text": (
                    "클라우드 사업은 고객 전환 수요 증가로 확대되고 있습니다. "
                    "다만 일반적인 위험 문구가 함께 포함되어 있습니다."
                ),
            },
            {
                "chunk_id": "other:1",
                "chunk_index": 2,
                "section_key": "other",
                "section_title": "그 밖에 투자자 보호를 위하여 필요한 사항",
                "text": "규제 변화와 소송 위험으로 인한 불확실성이 존재합니다.",
            },
            {
                "chunk_id": "detailed_tables:1",
                "chunk_index": 3,
                "section_key": "detailed_tables",
                "section_title": "상세표",
                "text": "매출 증가 투자 위험 수주 계약 같은 단어가 반복되는 상세표입니다.",
            },
        ],
    }

    signals = business_signals_from_dart(article, parser_result)
    section_by_uid = {signal["source_chunk_uid"]: signal for signal in signals}

    assert section_by_uid["business:1"]["signal_type"] == "growth"
    assert section_by_uid["business:1"]["business_area"] == "cloud"
    assert section_by_uid["other:1"]["signal_type"] == "risk"
    assert "detailed_tables:1" not in section_by_uid


def test_dart_business_signals_do_not_match_ai_inside_financial_words() -> None:
    article = {
        "id": 13,
        "title": "사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["test_peer"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "test_peer",
        "period": "2026Q1",
        "document_chunks": [
            {
                "chunk_id": "financial:1",
                "chunk_index": 1,
                "section_key": "financial",
                "section_title": "재무에 관한 사항",
                "text": (
                    "Financial statement에는 매출 증가와 영업이익 개선 내용이 포함되어 있습니다."
                ),
            }
        ],
    }

    signals = business_signals_from_dart(article, parser_result)

    assert signals == []


def test_dart_business_signals_do_not_apply_multi_area_chunk_to_every_sentence() -> None:
    article = {
        "id": 14,
        "title": "사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["test_peer"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "test_peer",
        "period": "2026Q1",
        "document_chunks": [
            {
                "chunk_id": "business:multi",
                "chunk_index": 1,
                "section_key": "business",
                "section_title": "사업의 내용",
                "text": (
                    "클라우드 사업은 고객 수요 증가로 확대되고 있습니다. "
                    "AI 플랫폼은 업무 자동화 수요에 대응합니다. "
                    "일반 솔루션 매출도 증가했습니다."
                ),
            }
        ],
    }

    signals = business_signals_from_dart(article, parser_result)

    assert {signal["business_area"] for signal in signals} == {"cloud", "ai_ax"}
    assert all("일반 솔루션" not in signal["evidence_text"] for signal in signals)


def test_dart_sk_ax_financial_metrics_are_not_stored_as_company_total() -> None:
    article = {
        "id": 15,
        "title": "SK주식회사 사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["sk_ax"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "sk_ax",
        "period": "2026Q1",
        "financial_statements": [
            {
                "table_index": 3,
                "title": "연결 포괄손익계산서",
                "table_type": "income_statement",
                "statement_scope": "consolidated",
                "rows": [
                    {
                        "metric_key": "revenue_total",
                        "label": "매출액",
                        "current_value_krwbn": 10000,
                    }
                ],
            }
        ],
    }

    assert financial_metrics_from_dart(article, parser_result) == []


def test_dart_sk_ax_signals_keep_only_sk_ax_related_sentences() -> None:
    article = {
        "id": 16,
        "title": "SK주식회사 사업보고서",
        "url": "https://example.com/dart",
        "source_name": "dart",
        "company": ["sk_ax"],
        "extra": {},
    }
    parser_result = {
        "peer_id": "sk_ax",
        "period": "2026Q1",
        "document_chunks": [
            {
                "chunk_id": "business:sk",
                "chunk_index": 1,
                "section_key": "business",
                "section_title": "사업의 내용",
                "text": (
                    "SK하이닉스는 반도체 투자와 생산 확대를 추진하고 있습니다. "
                    "SK주식회사 C&C 부문은 AX와 클라우드 서비스를 확대하고 있습니다. "
                    "SK바이오팜은 신약 개발 투자를 지속하고 있습니다."
                ),
            }
        ],
    }

    signals = business_signals_from_dart(article, parser_result)

    assert signals
    assert all("C&C" in signal["evidence_text"] for signal in signals)
    assert all("SK하이닉스" not in signal["evidence_text"] for signal in signals)
    assert all("SK바이오팜" not in signal["evidence_text"] for signal in signals)
