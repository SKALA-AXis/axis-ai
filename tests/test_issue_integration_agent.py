from __future__ import annotations

from src.agents.issue_integration_agent import (
    IssueIntegrationAgent,
    analysis_input_bundle_from_articles,
)
from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration import IntegratedIssueComposer, IntegrationPolicy


def test_issue_integration_promotes_parser_and_metric_without_title():
    bundle = analysis_input_bundle_from_articles(
        cluster_id=100,
        representative_id=100,
        articles=[
            {
                "id": 100,
                "title": "",
                "source_type": "dart",
                "source_name": "DART",
                "company": ["samsung_sds"],
                "financial_metrics": [
                    {
                        "metric_name": "revenue",
                        "metric_label": "매출액",
                        "value_numeric": 1000,
                        "unit": "억원",
                        "evidence_text": "매출액은 1000억원입니다.",
                    }
                ],
                "parser_result": {
                    "document_chunks": [
                        {
                            "chunk_id": "business:1",
                            "section_key": "business",
                            "section_title": "사업의 내용",
                            "text": "클라우드 MSP와 생성형 AI 기반 AX 사업을 확대하고 있습니다.",
                        }
                    ],
                    "topic_signals": [
                        {
                            "topic": "AX 사업 확대",
                            "evidence_text": "생성형 AI 기반 AX 사업을 확대",
                            "section_key": "business",
                        }
                    ],
                },
            }
        ],
        classification={"company": "samsung_sds", "sector": "ax"},
    )

    fact_text = " ".join(str(fact.get("fact") or "") for fact in bundle.facts)
    assert "AX 사업 확대" in fact_text
    assert "매출액" in fact_text
    assert any(fact.get("source_chunk_uid") == "business:1" for fact in bundle.facts)

    integrated = IssueIntegrationAgent().integrate_input_bundle(bundle)
    assert integrated["schema_version"] == "integrated_issue_v2"
    assert integrated["source_family"] == "filing"
    assert integrated["is_valid_summary"] is True
    assert integrated["evidence_ledger"]
    assert integrated["claim_ledger"]
    assert integrated["quality"]["has_parser_result"] is True
    assert integrated["quality"]["has_structured_metrics"] is True
    assert integrated["content_digest"]["has_content"] is True
    assert "클라우드 MSP" in integrated["content_digest"]["summary"]
    assert "사업 내용:" in integrated["content_digest"]["detailed_explanation"]
    assert integrated["content_digest"]["sections"]
    assert integrated["content_digest"]["section_count"] == len(
        integrated["content_digest"]["sections"]
    )
    assert integrated["content_digest"]["sections"][0]["source_count"] >= 1
    assert integrated["content_digest"]["body_extracts"]
    assert integrated["content_digest_storage"]["content_summary"] == integrated[
        "content_digest"
    ]["summary"]
    assert integrated["source_map"]["sources"][0]["source_index"] == 1
    assert integrated["source_map"]["raw_article_id_to_source_index"]["100"] == 1
    assert integrated["issue_frame"]["companies"]["primary"]["id"] == "samsung_sds"
    assert integrated["issue_frame"]["sectors"][0]["id"] == "ax"
    assert integrated["issue_frame"]["topics"]
    assert set(integrated["content_digest"]["sources"][0]) == {
        "source_index",
        "id",
        "title",
        "source",
        "summary",
        "key_points",
        "body_extracts",
        "url",
        "published_at",
        "publisher",
    }


def test_issue_integration_accepts_industry_trend_without_main_company():
    bundle = analysis_input_bundle_from_articles(
        cluster_id=200,
        representative_id=200,
        articles=[
            {
                "id": 200,
                "title": "글로벌 AI 산업 트렌드",
                "source_type": "spri",
                "source_name": "SPRi",
                "matched_sectors": ["ax"],
                "parser_result": {
                    "document_chunks": [
                        {
                            "chunk_id": "trend:1",
                            "section_key": "industry_trend",
                            "section_title": "산업 동향",
                            "text": (
                                "기업들은 생성형 AI를 업무 자동화와 소프트웨어 개발 "
                                "생산성 향상에 적용하고 있습니다."
                            ),
                        }
                    ]
                },
            }
        ],
        classification={"sector": "ax"},
    )

    integrated = IssueIntegrationAgent().integrate_input_bundle(bundle)
    assert integrated["main_company"] == ""
    assert integrated["scope_type"] == "industry"
    assert integrated["source_family"] == "trend"
    assert integrated["is_valid_summary"] is True
    assert integrated["document_subject"]


def test_issue_integration_budget_selection_uses_score_and_coverage():
    policy = IntegrationPolicy(target_evidence_tokens=80)
    composer = IntegratedIssueComposer(policy=policy)
    bundle = AnalysisInputBundle(
        bundle_id="dart:300",
        cluster_id="300",
        source_type="dart",
        companies=["samsung_sds"],
        sectors=["ax"],
        event_type="general_update",
        items=[{"id": 300, "source_type": "dart", "title": "사업보고서"}],
        facts=[
            {
                "fact_id": "article:300:title",
                "article_id": 300,
                "fact": "사업보고서",
                "evidence_text": "사업보고서",
                "source_type": "dart",
                "fact_type": "general_fact",
                "derived_from": "title",
            },
            {
                "fact_id": "article:300:metric:1",
                "article_id": 300,
                "fact": "매출액 1000억원",
                "evidence_text": "매출액은 1000억원입니다.",
                "source_type": "dart",
                "fact_type": "financial_metric",
                "derived_from": "financial_metric",
                "numbers_and_dates": ["1000억원"],
            },
            {
                "fact_id": "article:300:business_signal:1",
                "article_id": 300,
                "fact": "AX 사업 확대",
                "evidence_text": "생성형 AI 기반 AX 사업을 확대",
                "source_type": "dart",
                "fact_type": "business_signal",
                "derived_from": "business_signal",
            },
        ],
        evidence_snippets=[],
        sources=[{"article_id": 300, "title": "사업보고서", "source_type": "dart"}],
        metadata={"representative_id": 300},
    )

    integrated = composer.compose_non_news(bundle)
    selected_ids = set(integrated["quality"]["selection"]["selected_fact_ids"])
    assert "article:300:metric:1" in selected_ids
    assert "article:300:business_signal:1" in selected_ids
    assert integrated["quality"]["selection"]["selection_reason"] in {
        "within_budget_all_facts_selected",
        "budgeted_score_and_coverage_selection",
    }


def test_raw_article_row_contract_preserves_id_and_status_quality():
    raw_row = {
        "id": 26223,
        "source_name": "naver_news",
        "title": "대법, 최태원-노소영 재산분할 파기환송",
        "content": "대법원은 재산분할 판단을 파기환송했고 위자료 20억 원은 확정됐다.",
        "url": "http://www.whitepaper.co.kr/news/articleView.html?idxno=253220",
        "published_at": "2025-10-16 10:54:00.000 +0900",
        "collected_at": "2026-05-18 12:44:16.045 +0900",
        "processing_status": "SKIPPED",
        "source_type": "news",
        "publisher": "화이트페이퍼",
        "company": '["sk_ax"]',
        "language": "ko",
        "content_type": "html",
        "crawl_status": "success",
        "relevance_score": 0.25,
        "relevance_label": "irrelevant",
        "relevance_reason": "피어사가 핵심 주체가 아니라 단순 언급으로 판단",
        "matched_companies": '["sk_ax"]',
        "matched_sectors": '["other"]',
        "metadata": (
            '{"link_check":{"status":"success","final_url":"http://se-cu.com/ndsoft/error.html"},'
            '"image_urls":["https://www.whitepaper.co.kr/news/photo/202510/example.png"]}'
        ),
    }
    bundle = analysis_input_bundle_from_articles(
        cluster_id=None,
        representative_id=26223,
        articles=[raw_row],
    )

    assert bundle.companies == ["sk_ax"]
    assert bundle.sectors == ["other"]
    assert bundle.metadata["cluster_article_ids"] == [26223]
    assert bundle.sources[0]["article_id"] == 26223
    assert bundle.sources[0]["raw_article_id"] == 26223
    assert bundle.sources[0]["processing_status"] == "SKIPPED"
    assert bundle.sources[0]["relevance_label"] == "irrelevant"
    assert bundle.sources[0]["is_analysis_eligible"] is False
    assert bundle.sources[0]["link_check"]["status"] == "success"
    assert any(fact["fact_id"].startswith("raw_article:26223:") for fact in bundle.facts)
    assert all(fact["raw_article_id"] == 26223 for fact in bundle.facts)

    integrated = IntegratedIssueComposer().compose_non_news(bundle)

    assert integrated["raw_article_ids"] == [26223]
    assert integrated["is_valid_summary"] is False
    assert integrated["quality"]["eligible_source_count"] == 0
    assert integrated["quality"]["skipped_source_count"] == 1
    assert integrated["quality"]["irrelevant_source_count"] == 1
    assert "no_analysis_eligible_source_rows" in integrated["quality"]["review_flags"]
    assert "contains_irrelevant_source_rows" in integrated["quality"]["review_flags"]
    assert integrated["quality"]["selected_fact_count"] == 0
    assert integrated["evidence_ledger"] == []
    assert integrated["content_digest"]["basis_scope"] == "all_sources_for_review"
    assert "대법원은 재산분할 판단을 파기환송" in integrated["content_digest"]["summary"]
    assert "대법원은 재산분할 판단을 파기환송" in integrated["content_digest"][
        "detailed_explanation"
    ]
    assert integrated["content_digest"]["sources"][0]["id"] == 26223
    assert integrated["content_digest"]["sections"][0]["section"]
    assert integrated["content_digest"]["sections"][0]["detailed_explanation"]
    assert integrated["content_digest"]["sections"][0]["body_extracts"]
    assert integrated["source_map"]["sources"][0]["raw_article_id"] == 26223
    assert integrated["source_map"]["sources"][0]["source_index"] == 1
    assert integrated["source_map"]["sources"][0]["is_analysis_eligible"] is False
    assert integrated["source_map"]["basis_raw_article_ids"] == [26223]
    assert integrated["issue_frame"]["companies"]["primary"]["id"] == "sk_ax"
    assert integrated["issue_frame"]["event"]["type"] == "company"
    assert integrated["issue_frame"]["quality"]["frame_completeness"] in {"partial", "complete"}
    assert integrated["content_digest_storage"]["content_raw_article_ids"] == [26223]
    assert integrated["content_digest"]["sources"][0]["id"] == 26223
    assert integrated["content_digest"]["sources"][0]["source"] == "naver_news"
    assert integrated["content_digest"]["sources"][0]["publisher"] == "화이트페이퍼"
