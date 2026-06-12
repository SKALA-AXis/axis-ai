from __future__ import annotations

import asyncio

from src.agents.briefing import data_layer as briefing_data_layer
from src.agents.briefing_generation_agent import (
    BriefingGenerationAgent,
    _display_copy_context,
    _normalize_mock_item,
)
from src.api.briefing_schemas import BriefingGenerateRequest, BriefingGenerateResponse


def _mock_item(
    card_id: str,
    issue_id: str,
    *,
    created_at: str,
    recommended_actions: list[str] | None = None,
) -> dict:
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
                        "recommended_actions": recommended_actions or [f"{card_id} 대응"],
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


def test_briefing_input_output_contract_for_integrated_issue_basis():
    issue_id = "11111111-1111-1111-1111-111111111111"
    other_issue_id = "22222222-2222-2222-2222-222222222222"
    request = BriefingGenerateRequest(
        briefing_type="daily",
        anchor_date="2026-06-04",
        integrated_issue_ids=[issue_id, other_issue_id],
        peer_ids=["samsung_sds"],
        sectors=["ax"],
        user_context="임원용 회사 대응 관점",
        limit=10,
        save=False,
        refine_display_copy=False,
    )
    items = [
        _mock_item(
            "CN-1",
            issue_id,
            created_at="2026-06-04T09:00:00+09:00",
            recommended_actions=["제안서에서 데이터 보호와 보안성을 강조합니다."],
        ),
        _mock_item("CN-2", other_issue_id, created_at="2026-06-03T09:00:00+09:00"),
    ]

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type=request.briefing_type,
            anchor_date=request.anchor_date,
            integrated_issue_ids=request.integrated_issue_ids,
            peer_ids=request.peer_ids,
            sectors=request.sectors,
            requested_by_user_id=request.requested_by_user_id,
            ratios=request.ratios,
            user_context=request.user_context,
            limit=request.limit,
            save=request.save,
            use_mock=True,
            mock_items=items,
            refine_display_copy=request.refine_display_copy,
        )
    )

    response = BriefingGenerateResponse.model_validate(result)
    assert response.briefing_type == "daily"
    assert response.date_from == "2026-06-04"
    assert response.date_to == "2026-06-04"
    assert response.related_card_ids == ["CN-1"]
    assert response.source_integrated_issue_ids == [issue_id]
    assert response.primary_card_news_id == "CN-1"
    assert response.selected_cards[0]["integrated_issue_id"] == issue_id
    assert response.selected_cards[0]["has_analysis_package"] is True
    assert response.hidden_details[0]["analysis_package"]["integrated_issue_id"] == issue_id
    assert response.briefing_basis["sources_used"] == ["CN-1"]
    assert response.briefing_basis["source_integrated_issue_ids"] == [issue_id]
    assert response.provenance["requested_integrated_issue_ids"] == [issue_id, other_issue_id]
    assert response.provenance["excluded_integrated_issue_ids"] == [other_issue_id]
    assert response.provenance["briefing_analysis_basis"].startswith("integrated_issues primary")
    assert response.source_card_ids == ["CN-1"]
    assert response.executive_summary
    assert response.immediate_trends[0]["related_card_id"] == "CN-1"
    assert response.sections[0]["title"] == "핵심 인사이트 요약"
    assert response.dailySnapshot["sections"][0]["title"] == "오늘 바로 검토할 동향"
    assert response.weeklySnapshot["sections"][0]["title"] == "이번 주 핵심 변화"
    assert response.history[0]["primaryCount"] >= 1
    assert response.interpretation_flow["title"] == "해석 흐름 — 관찰부터 시사까지"
    assert response.briefingReport["label"] == "일간"
    assert response.briefingReport["briefingLead"] == response.briefing_lead
    assert response.briefingReport["selectedCards"][0]["id"] == "CN-1"
    assert response.briefingReport["signalCards"][0]["relatedCardIds"] == ["CN-1"]
    assert response.briefingReport["flowSteps"][0]["id"].startswith("generated-")
    assert response.flowSteps == response.briefingReport["flowSteps"]
    assert (
        response.interpretation_flow["reasoning_summary"]["disclosure_level"]
        == "summarized_intermediate_artifacts"
    )
    for step in response.interpretation_flow["steps"]:
        trace = step["reasoning_trace"]
        assert trace["source_inputs"]
        assert len(trace["intermediate_artifacts"]) == 4
        assert trace["output_items"]

    actions = response.briefing_basis["recommended_actions"]
    assert len(actions) == 3
    assert all(action.startswith("SK AX") for action in actions)
    assert all("제안서" not in action for action in actions)


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


def test_briefing_rewrites_program_actions_for_executives():
    issue_id = "11111111-1111-1111-1111-111111111111"
    items = [
        _mock_item(
            "CN-1",
            issue_id,
            created_at="2026-06-04T09:00:00+09:00",
            recommended_actions=[
                "제안서에서 데이터 보호와 보안성을 강조합니다.",
                "공공 부문 PoC 및 레퍼런스 확보 전략을 수립합니다.",
            ],
        )
    ]

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily",
            anchor_date="2026-06-04",
            integrated_issue_ids=[issue_id],
            use_mock=True,
            mock_items=items,
            refine_display_copy=False,
        )
    )

    actions = result["briefing_basis"]["recommended_actions"]
    assert len(actions) == 3
    assert all(action.startswith("SK AX") for action in actions)
    assert all("제안서" not in action for action in actions)
    assert all("PoC" not in action for action in actions)
    assert any("책임 조직" in action and "리스크 승인 권한" in action for action in actions)
    assert result["briefing_basis"]["action_details"][0]["use_case"] == "사업 우선순위"


def test_briefing_uses_integrated_issues_as_primary_lookup(monkeypatch):
    issue_id = "11111111-1111-1111-1111-111111111111"
    legacy_issue = "낡은 card_news 통합 이슈"

    def fake_rows(**_kwargs):
        return [
            {
                "integrated_issue_id": issue_id,
                "issue_key": "issue-key",
                "cluster_id": "cluster-1",
                "representative_raw_article_id": 1,
                "main_company": "samsung_sds",
                "event_type": "new_biz",
                "is_valid": True,
                "confidence": 0.82,
                "headline": "삼성SDS 생성형 AI 운영 플랫폼 확대",
                "one_line_summary": "운영 플랫폼 레퍼런스 확대",
                "source_ids": [1],
                "analyzed_source_ids": [1],
                "sectors": ["ax"],
                "mentioned_peer_companies": ["samsung_sds"],
                "content_summary": "운영 플랫폼 레퍼런스가 확대되고 있습니다.",
                "content_detailed_explanation": (
                    "제조와 금융 고객군에서 운영형 AI 신호가 확인됩니다."
                ),
                "issue_brief": {"headline": "운영형 AI 레퍼런스 확대"},
                "content_digest": {
                    "summary": "운영형 AI 수요가 확인됩니다.",
                    "detailed_explanation": "고객 평가 기준이 운영 책임으로 이동합니다.",
                },
                "issue_frame": {"frame": "운영형 AI"},
                "issue_payload": {
                    "is_valid_summary": True,
                    "main_issue": "통합 이슈 원문",
                    "integrated_text": "통합 이슈 상세",
                    "source_article_ids": [1],
                    "business_signals": [{"signal": "운영형 AI 레퍼런스"}],
                },
                "source_links": [
                    {
                        "source_name": "연합뉴스",
                        "published_at": "2026-06-04T09:00:00+09:00",
                        "title": "기사",
                        "url": "https://example.com/news",
                    }
                ],
                "source_names": ["연합뉴스"],
                "evidence_refs": [
                    {
                        "evidence_ref_id": "fact-1",
                        "text": "통합 근거",
                        "source_ids": [1],
                    }
                ],
                "content_sections": [],
                "anchor_card_id": "CN-1",
                "anchor_peer_id": "samsung_sds",
                "anchor_importance": "high",
                "anchor_importance_score": 0.9,
                "anchor_evidence_payload": {
                    "analysis_package": {
                        "integrated_issue": {"main_issue": legacy_issue},
                        "analysis": {
                            "analysis_summary": "card_news legacy 분석",
                            "market_signal": "card_news legacy 시장 신호",
                            "confidence": 0.8,
                        },
                        "implication": {
                            "skax_implication": {
                                "why_important": "운영 책임 기준을 확인해야 합니다.",
                                "potential_impact": "운영형 AI 수요 대응이 필요합니다.",
                                "recommended_actions": ["legacy 액션"],
                            },
                            "confidence": 0.8,
                        },
                        "classification": {"sector": "ax", "sectors": ["ax"]},
                        "validation": {"pass": True, "sc_score": 0.8},
                    }
                },
                "basis_at": "2026-06-04T09:00:00+09:00",
                "issue_created_at": "2026-06-04T09:00:00+09:00",
            }
        ]

    # 분리 후 실호출자는 src.agents.briefing.data_layer 내부 — 그 모듈 바인딩을 패치해야 효과 있음
    monkeypatch.setattr(briefing_data_layer, "_fetch_period_integrated_issue_rows", fake_rows)

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily",
            anchor_date="2026-06-04",
            integrated_issue_ids=[issue_id],
            use_mock=False,
            refine_display_copy=False,
        )
    )

    assert result["provenance"]["source_mode"] == "integrated_issue_period_lookup"
    assert result["related_card_ids"] == ["CN-1"]
    assert result["source_card_ids"] == ["CN-1"]
    assert result["source_integrated_issue_ids"] == [issue_id]
    assert result["primary_card_news_id"] == "CN-1"
    assert result["hidden_details"][0]["analysis_package"]["integrated_issue"]["main_issue"] == (
        "통합 이슈 원문"
    )
    assert (
        result["hidden_details"][0]["analysis_package"]["integrated_issue"]["main_issue"]
        != legacy_issue
    )
    assert result["immediate_trends"][0]["related_card_id"] == "CN-1"
    assert result["dailySnapshot"]["sections"][0]["items"][0]["source"].startswith("연합뉴스")
    first_trace = result["interpretation_flow"]["steps"][0]["reasoning_trace"]
    assert first_trace["source_inputs"][0]["integrated_issue_id"] == issue_id
    assert first_trace["intermediate_artifacts"][0]["name"] == "입력 근거 묶음"
    assert "연합뉴스" in first_trace["source_inputs"][0]["source_names"]
