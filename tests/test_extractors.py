from scripts.reprocess_securities_report_analysis import (
    _metric_dedupe_key,
    _signal_dedupe_key,
)
from src.extractors.dart_analysis_extractor import (
    business_signals_from_dart,
    financial_metrics_from_dart,
)
from src.extractors.securities_report_analysis_extractor import (
    business_signals_from_securities_report,
    financial_metrics_from_securities_report,
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
    assert metrics[0]["business_area"] == "company_total"
    assert metrics[0]["value_krw"] == 13929868000000.0
    assert metrics[0]["source_table_uid"] == "dart-table-3"
    assert metrics[0]["confidence"] >= 0.9


def test_dart_financial_metrics_preserve_statement_column_periods() -> None:
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
        "period": "2025",
        "period_year": 2025,
        "period_quarter": None,
        "period_type": "annual",
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
                        "values": [
                            {
                                "column_index": 1,
                                "column_header": "제 41 (당) 기",
                                "raw": "13,929,868",
                                "value_krwbn": 139298.68,
                                "period": "2025",
                                "period_year": 2025,
                                "period_quarter": None,
                                "period_type": "annual",
                            },
                            {
                                "column_index": 2,
                                "column_header": "제 40 (전) 기",
                                "raw": "13,828,232",
                                "value_krwbn": 138282.32,
                                "period": "2024",
                                "period_year": 2024,
                                "period_quarter": None,
                                "period_type": "annual",
                            },
                        ],
                    },
                ],
            }
        ],
    }

    metrics = financial_metrics_from_dart(article, parser_result)

    assert [metric["period"] for metric in metrics] == ["2025", "2024"]
    assert metrics[1]["period_year"] == 2024
    assert metrics[1]["period_quarter"] is None
    assert metrics[1]["value_krwbn"] == 138282.32


def test_dart_financial_metrics_normalize_legacy_annual_q4_period() -> None:
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
        "period": "2025Q4",
        "period_year": 2025,
        "period_quarter": 4,
        "period_type": "annual",
        "financial_statements": [
            {
                "table_index": 3,
                "title": "연결 포괄손익계산서",
                "table_type": "income_statement",
                "statement_scope": "consolidated",
                "unit": "백만원",
                "rows": [
                    {
                        "metric_key": "operating_profit",
                        "label": "영업이익",
                        "values": [
                            {
                                "column_index": 1,
                                "column_header": "제 41 (당) 기",
                                "raw": "60,400",
                                "value_krwbn": 604,
                                "period": "2025Q4",
                                "period_year": 2025,
                                "period_quarter": 4,
                                "period_type": "annual",
                            },
                        ],
                    },
                ],
            }
        ],
    }

    metrics = financial_metrics_from_dart(article, parser_result)

    assert metrics[0]["metric_name"] == "operating_profit"
    assert metrics[0]["period"] == "2025"
    assert metrics[0]["period_year"] == 2025
    assert metrics[0]["period_quarter"] is None
    assert metrics[0]["period_type"] == "annual"


def test_dart_financial_metrics_include_segment_candidates_with_statements() -> None:
    article = {
        "id": 10,
        "title": "분기보고서",
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
                ],
            }
        ],
        "candidates": [
            {
                "type": "revenue_total",
                "metric_scope": "segment",
                "business_area": "클라우드",
                "value_krwbn": 6908.66,
                "raw": "클라우드 690,866 (백만원)",
                "table_index": 7,
                "confidence": 0.9,
            }
        ],
    }

    metrics = financial_metrics_from_dart(article, parser_result)

    assert any(metric["business_area"] == "클라우드" for metric in metrics)


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


def test_dart_sk_ax_financial_metrics_keep_only_sk_ax_business_segment() -> None:
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
        "candidates": [
            {
                "type": "revenue_total",
                "metric_scope": "segment",
                "business_area": "SK주식회사 사업부문",
                "standard_business_area": "sk_ax",
                "value_krwbn": 1234,
                "raw": "SK주식회사 사업부문 IT서비스 123,400 (백만원)",
                "source": "business_segment_table",
            },
            {
                "type": "revenue_total",
                "metric_scope": "segment",
                "business_area": "SK텔레콤",
                "value_krwbn": 9999,
                "raw": "SK텔레콤 999,900 (백만원)",
                "source": "business_segment_table",
            },
        ],
    }

    metrics = financial_metrics_from_dart(article, parser_result)

    assert len(metrics) == 1
    assert metrics[0]["business_area"] == "sk_ax"
    assert metrics[0]["value_krwbn"] == 1234


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


def test_securities_report_extracts_valuation_and_financial_metrics() -> None:
    article = {
        "id": 77,
        "company": ["samsung_sds"],
        "title": "[iM증권] 삼성SDS 2026Q1 Review",
        "content": (
            "목표주가 220,000원 현재주가 169,800원\n"
            "매출액 33,529억원 영업이익 783억원 영업이익률 2.3%\n"
        ),
        "url": "https://example.com/report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "samsung_sds",
        "period": "2026Q1",
        "report_firm": "iM증권",
        "target_price_krw": 220000,
        "current_price_krw": 169800,
    }

    metrics = financial_metrics_from_securities_report(article, parser_result)
    by_name = {metric["metric_name"]: metric for metric in metrics}

    assert by_name["target_price"]["value_numeric"] == 220000
    assert by_name["current_price"]["value_numeric"] == 169800
    assert by_name["upside_pct"]["unit"] == "%"
    assert by_name["revenue_total"]["value_krwbn"] == 33529
    assert by_name["operating_profit"]["value_krwbn"] == 783
    assert by_name["operating_margin"]["value_numeric"] == 2.3


def test_securities_report_does_not_store_per_pbr_roe_metrics() -> None:
    article = {
        "id": 177,
        "company": ["hyundai_autoever"],
        "title": "[증권사] 현대오토에버 Review",
        "content": "PER 18.5배 PBR 2.0배 ROE 4.7% EPS 5,731원",
        "url": "https://example.com/report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "report_firm": "테스트증권",
    }

    metrics = financial_metrics_from_securities_report(article, parser_result)
    metric_names = {metric["metric_name"] for metric in metrics}

    assert "eps" in metric_names
    assert "per" not in metric_names
    assert "pbr" not in metric_names
    assert "roe" not in metric_names


def test_securities_report_extracts_business_forecast_signals() -> None:
    article = {
        "id": 78,
        "company": ["samsung_sds"],
        "title": "[iM증권] 삼성SDS 2026Q1 Review",
        "content": "",
        "url": "https://example.com/report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "samsung_sds",
        "period": "2026Q1",
        "report_firm": "iM증권",
        "document_chunks": [
            {
                "chunk_id": "forecast:1",
                "section_key": "forecast",
                "section_title": "Forecast",
                "text": (
                    "AI 데이터센터 투자 확대와 클라우드 수요 증가로 "
                    "2026년 성장 모멘텀이 강화될 전망이다."
                ),
            }
        ],
    }

    signals = business_signals_from_securities_report(article, parser_result)

    assert {signal["business_area"] for signal in signals} == {"cloud"}
    assert {signal["signal_type"] for signal in signals} == {"growth"}


def test_securities_report_skips_table_like_signal_rows_and_keeps_one_signal() -> None:
    article = {
        "id": 79,
        "company": ["hyundai_autoever"],
        "title": "[증권사] 현대오토에버 Review",
        "content": "",
        "url": "https://example.com/report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "report_firm": "테스트증권",
        "document_chunks": [
            {
                "chunk_id": "forecast:1",
                "section_key": "forecast",
                "section_title": "Forecast",
                "text": (
                    "[표1] 현대오토에버의 분기 및 연간 실적 추이 및 전망 (단위: 십억원, %, %YoY) "
                    "1Q24 2Q24 3Q24 4Q24 1Q25 2Q25P 3Q25E 4Q25E 2024 2025E 2026E "
                    "AI 데이터센터 투자 확대와 클라우드 수요 증가로 2026년 성장 모멘텀이 "
                    "강화될 전망이다."
                ),
            }
        ],
    }

    signals = business_signals_from_securities_report(article, parser_result)

    assert len(signals) == 1
    assert signals[0]["signal_type"] == "growth"
    assert (
        signals[0]["evidence_text"]
        == "AI 데이터센터 투자 확대와 클라우드 수요 증가로 2026년 성장 모멘텀이 강화될 전망이다."
    )


def test_securities_report_sk_ax_keeps_only_it_service_context() -> None:
    article = {
        "id": 179,
        "company": ["sk_ax"],
        "title": "[증권사] SK Review",
        "content": (
            "자회사 지분가치 상승 등으로 현재 SK NAV는 78.2조원으로 추정된다. "
            "비상장자회사 SK에코플랜트의 더블다운 가치도 개선될 전망이다. "
            "SK C&C IT서비스 부문은 AX와 클라우드 전환 수요 확대로 매출 성장이 예상된다. "
            "이에 플랫폼 리벨런싱 성과가 가시화되면서 영업이익 개선에 기여할 전망이다."
        ),
        "url": "https://example.com/sk-report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "sk_ax",
        "period": "2026E",
        "report_firm": "테스트증권",
        "document_chunks": [
            {
                "chunk_id": "forecast:1",
                "section_key": "forecast",
                "section_title": "Forecast",
                "text": article["content"],
            }
        ],
    }

    signals = business_signals_from_securities_report(article, parser_result)

    assert signals
    evidence = " ".join(signal["evidence_text"] for signal in signals)
    assert "SK C&C IT서비스 부문" in evidence
    assert "영업이익 개선에 기여" in evidence
    assert "SK NAV" not in evidence
    assert "SK에코플랜트" not in evidence


def test_securities_report_sk_ax_does_not_store_holding_company_valuation_metrics() -> None:
    article = {
        "id": 180,
        "company": ["sk_ax"],
        "title": "[증권사] SK Review",
        "content": (
            "목표주가 220,000원 현재주가 169,800원 "
            "SK NAV는 78.2조원으로 추정된다. "
            "SK C&C IT서비스 부문 매출 7,692억원으로 추정된다."
        ),
        "url": "https://example.com/sk-report.pdf",
        "source_name": "naver_research",
        "extra": {},
    }
    parser_result = {
        "peer_id": "sk_ax",
        "period": "2026E",
        "report_firm": "테스트증권",
        "target_price_krw": 220000,
        "current_price_krw": 169800,
    }

    metrics = financial_metrics_from_securities_report(article, parser_result)
    metric_names = {metric["metric_name"] for metric in metrics}
    evidence = " ".join(metric["evidence_text"] for metric in metrics)

    assert "target_price" not in metric_names
    assert "current_price" not in metric_names
    assert "upside_pct" not in metric_names
    assert "SK NAV" not in evidence
    assert metric_names == {"revenue_total"}


def test_securities_report_metric_dedupe_key_ignores_small_numeric_formatting_diff() -> None:
    row1 = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "metric_name": "target_price",
        "business_area": None,
        "value_numeric": 150500,
        "unit": "원",
    }
    row2 = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "metric_name": "target_price",
        "business_area": None,
        "value_numeric": 150500.0,
        "unit": "원",
    }

    assert _metric_dedupe_key(row1) == _metric_dedupe_key(row2)


def test_securities_report_signal_dedupe_key_normalizes_whitespace() -> None:
    row1 = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "business_area": "company_total",
        "signal_type": "growth",
        "evidence_text": "동사 1H25 매출은 1.9조원으로 전년 동기 대비 13.7% 성장.",
    }
    row2 = {
        "peer_id": "hyundai_autoever",
        "period": "2024Q1",
        "business_area": "company_total",
        "signal_type": "growth",
        "evidence_text": "동사 1H25   매출은 1.9조원으로 전년 동기 대비 13.7% 성장. ",
    }

    assert _signal_dedupe_key(row1) == _signal_dedupe_key(row2)
