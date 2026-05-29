from __future__ import annotations

from typing import Any

from src.analysis.models import AnalysisInputBundle
from src.services.issue_integration.agent_views import (
    analysis_agent_issue_input,
    card_news_issue_input,
)
from src.services.issue_integration.issue_composer import IntegratedIssueComposer


def test_integrated_issue_output_is_downstream_ready() -> None:
    issue = IntegratedIssueComposer().compose_non_news(_sample_bundle())

    assert issue["schema_version"] == "integrated_issue_v3"
    assert set(issue) == {
        "schema_version",
        "issue_brief",
        "analysis_ready_inputs",
        "content_digest",
        "issue_frame",
        "sources",
        "evidence",
        "quality",
        "metadata",
    }
    assert issue["issue_brief"]["analysis_scope"]["analyzed_source_ids"] == [26223]
    assert issue["content_digest"]["has_content"] is True
    assert "basis_raw_article_ids" not in issue["content_digest"]
    assert issue["issue_frame"]["source_scope"]["source_count"] == 1
    assert issue["sources"] == [
        {
            "id": 26223,
            "title": "SK AX, AI 플랫폼 사업 확대 공시",
            "source_name": "dart",
            "source_type": "dart",
            "publisher": "DART",
            "published_at": "2026-05-01T09:00:00+09:00",
            "url": "https://dart.fss.or.kr/sample",
            "relevance_label": "relevant",
            "relevance_score": 0.92,
        }
    ]

    evidence_facts = [
        fact
        for section in issue["evidence"]["by_section"]
        for fact in section["facts"]
    ]
    assert evidence_facts
    assert all("evidence_text" not in fact for fact in evidence_facts)
    assert isinstance(issue["evidence"]["references"], list)
    assert issue["quality"]["selected_fact_count"] >= 2


def test_agent_views_project_compact_inputs_from_integrated_issue() -> None:
    issue = IntegratedIssueComposer().compose_non_news(_sample_bundle())

    analysis_view = analysis_agent_issue_input(issue)
    card_view = card_news_issue_input(issue)

    assert analysis_view["view"] == "analysis_agent"
    assert analysis_view["gate"]["can_analyze"] is True
    assert analysis_view["identity"]["source_ids"] == [26223]
    assert analysis_view["content"]["summary"]
    assert analysis_view["evidence"]["fact_basis"]

    assert card_view["view"] == "card_news_agent"
    assert card_view["card_seed"]["headline"] == "SK AX, AI 플랫폼 사업 확대 공시"
    assert card_view["evidence"]["source_refs"] == analysis_view["source_refs"]


def _sample_bundle() -> AnalysisInputBundle:
    item: dict[str, Any] = {
        "id": 26223,
        "title": "SK AX, AI 플랫폼 사업 확대 공시",
        "content": (
            "SK AX는 2026년 AI 플랫폼 사업 매출이 전년 대비 15% 증가했다고 밝혔다. "
            "회사는 금융권과 제조 고객을 대상으로 생성형 AI 운영 플랫폼을 확대하고, "
            "클라우드 전환 프로젝트와 함께 수주 파이프라인을 강화하고 있다고 설명했다."
        ),
        "source_type": "dart",
        "source_name": "dart",
        "publisher": "DART",
        "published_at": "2026-05-01T09:00:00+09:00",
        "url": "https://dart.fss.or.kr/sample",
        "content_type": "html",
        "parser_result": {
            "document_type": "사업보고서",
            "document_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "section_key": "business",
                    "section_title": "사업 현황",
                    "text": (
                        "AI 플랫폼 매출은 전년 대비 15% 증가했고, "
                        "금융권과 제조 고객 중심으로 운영 플랫폼 적용이 확대됐다."
                    ),
                }
            ],
        },
        "financial_metrics": [
            {
                "metric_name": "ai_platform_revenue_growth",
                "value": "15",
                "unit": "%",
                "period": "2026",
            }
        ],
        "business_signals": [
            {
                "business_area": "AI platform",
                "signal_type": "expansion",
                "summary": "생성형 AI 운영 플랫폼 고객 확대",
            }
        ],
    }
    source = {
        "raw_article_id": 26223,
        "title": item["title"],
        "source_name": "dart",
        "source_type": "dart",
        "publisher": "DART",
        "published_at": item["published_at"],
        "url": item["url"],
        "processing_status": "PROCESSED",
        "crawl_status": "success",
        "relevance_label": "relevant",
        "relevance_score": 0.92,
        "is_analysis_eligible": True,
    }
    facts = [
        {
            "fact_id": "fact-business-1",
            "article_id": 26223,
            "raw_article_id": 26223,
            "fact": "SK AX의 AI 플랫폼 매출은 전년 대비 15% 증가했다.",
            "evidence_text": (
                "AI 플랫폼 매출은 전년 대비 15% 증가했고, "
                "금융권과 제조 고객 중심으로 운영 플랫폼 적용이 확대됐다."
            ),
            "fact_type": "business_signal",
            "derived_from": "parser_chunk",
            "business_area": "AI platform",
            "signal_type": "expansion",
            "numbers_and_dates": ["15%", "2026년"],
            "section_key": "business",
            "section_title": "사업 현황",
            "source_chunk_uid": "chunk-1",
            "confidence": 0.9,
        },
        {
            "fact_id": "fact-financial-1",
            "article_id": 26223,
            "raw_article_id": 26223,
            "fact": "AI 플랫폼 사업의 핵심 수치는 매출 성장률 15%다.",
            "evidence_text": "SK AX는 2026년 AI 플랫폼 사업 매출이 전년 대비 15% 증가했다고 밝혔다.",
            "fact_type": "financial_metric",
            "derived_from": "financial_metric",
            "metric_name": "ai_platform_revenue_growth",
            "metric_label": "AI 플랫폼 매출 성장률",
            "value": "15",
            "unit": "%",
            "period": "2026",
            "numbers_and_dates": ["15%", "2026년"],
            "confidence": 0.95,
        },
    ]
    return AnalysisInputBundle(
        bundle_id="dart:26223",
        cluster_id="26223",
        source_type="dart",
        companies=["sk_ax"],
        sectors=["ai"],
        event_type="business_expansion",
        items=[item],
        facts=facts,
        evidence_snippets=[
            {
                "article_id": 26223,
                "text": facts[0]["evidence_text"],
            }
        ],
        sources=[source],
        metadata={
            "representative_id": 26223,
            "cluster_article_ids": [26223],
        },
    )
