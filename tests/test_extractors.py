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

    assert len(signals) == 1
    assert signals[0]["source_type"] == "dart"
    assert signals[0]["business_area"] in {"cloud", "ai_ax"}
    assert signals[0]["signal_type"] in {"growth", "strategy", "efficiency"}
    assert "클라우드" in signals[0]["evidence_text"]
