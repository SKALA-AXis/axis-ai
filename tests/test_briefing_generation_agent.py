from __future__ import annotations

import asyncio

from src.agents.briefing_generation_agent import (
    BriefingGenerationAgent,
    _display_copy_context,
    _normalize_mock_item,
)
from src.api.briefing_schemas import BriefingGenerateRequest


def _mock_item(card_id: str, issue_id: str, *, created_at: str) -> dict:
    return {
        "id": card_id,
        "integrated_issue_id": issue_id,
        "title": f"{card_id} 제목",
        "company": "samsung_sds",
        "peer_id": "samsung_sds",
        "primary_keyword_category": "ax",
        "summary_lines": ["표시 요약"],
        "source_raw_article_ids": [1],
        "sources": [{"title": "기사", "published_at": created_at}],
        "created_at": created_at,
        "importance_score": 0.9,
        "evidence_payload": {
            "integrated_issue_id": issue_id,
            "evidence_refs": [{"evidence_ref_id": "fact-1", "text": "상세 근거"}],
            "analysis_package": {
                "integrated_issue_id": issue_id,
                "integrated_issue": {
                    "main_issue": f"{card_id} 통합 이슈",
                    "integrated_text": f"{card_id} 통합 상세",
                    "business_signals": [{"signal": f"{card_id} 사업 신호"}],
                    "source_article_ids": [1],
                },
                "analysis": {
                    "analysis_summary": f"{card_id} 분석 상세",
                    "market_signal": f"{card_id} 시장 신호",
                    "strategic_meaning": [f"{card_id} 전략 의미"],
                    "confidence": 0.8,
                },
                "implication": {
                    "peer_implication": {"peer_meaning": f"{card_id} peer 의미"},
                    "skax_implication": {
                        "why_important": f"{card_id} 중요 이유",
                        "potential_impact": f"{card_id} 영향",
                        "recommended_actions": [f"{card_id} 대응"],
                    },
                    "confidence": 0.8,
                },
                "classification": {"sector": "ax", "sectors": ["ax"]},
                "validation": {"sc_score": 0.8},
            },
        },
    }


def test_briefing_filters_integrated_issue_ids_with_period():
    issue_id = "11111111-1111-1111-1111-111111111111"
    other_issue_id = "22222222-2222-2222-2222-222222222222"
    items = [
        _mock_item("CN-1", issue_id, created_at="2026-06-04T09:00:00+09:00"),
        _mock_item("CN-2", other_issue_id, created_at="2026-06-03T09:00:00+09:00"),
    ]

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily",
            anchor_date="2026-06-04",
            integrated_issue_ids=[issue_id, other_issue_id],
            use_mock=True,
            mock_items=items,
            refine_display_copy=False,
        )
    )

    assert result["related_card_ids"] == ["CN-1"]
    assert result["source_integrated_issue_ids"] == [issue_id]
    assert result["primary_card_news_id"] == "CN-1"
    assert result["provenance"]["requested_integrated_issue_ids"] == [
        issue_id,
        other_issue_id,
    ]
    assert result["provenance"]["excluded_integrated_issue_ids"] == [other_issue_id]


def test_briefing_display_copy_context_uses_analysis_units():
    issue_id = "11111111-1111-1111-1111-111111111111"
    selected_card = _normalize_mock_item(
        _mock_item("CN-1", issue_id, created_at="2026-06-04T09:00:00+09:00")
    )
    report = {
        "title": "일간 브리핑",
        "briefing_type": "daily",
        "date_from": "2026-06-04",
        "date_to": "2026-06-04",
        "period_label": "2026년 06월 04일",
        "related_card_ids": ["CN-1"],
        "source_integrated_issue_ids": [issue_id],
        "selected_cards": [],
        "key_change_cards": [],
        "interpretation_flow": {"steps": []},
        "hidden_details": [],
    }

    context = _display_copy_context(report, [selected_card])

    assert "analysis_units" in context
    assert "analysis_packages" not in context
    assert context["analysis_units"][0]["integrated_issue_id"] == issue_id
    assert context["source_integrated_issue_ids"] == [issue_id]


def test_briefing_schema_accepts_integrated_issue_ids():
    request = BriefingGenerateRequest(integrated_issue_ids=["111"])

    assert request.integrated_issue_ids == ["111"]
