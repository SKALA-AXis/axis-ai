"""W2-1 — Supervisor LangGraph 통합 테스트 (mock LLM + DB)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    ProfileContext,
)
from src.pipeline.analysis_flow_graph import (
    SupervisorDeps,
    _hard_validate,
    build_supervisor_graph,
)


def _stub_bundle() -> AnalysisInputBundle:
    return AnalysisInputBundle(
        bundle_id="news:cluster_42",
        cluster_id="42",
        source_type="news",
        companies=["samsung_sds"],
        sectors=["ax"],
        event_type="partnership",
        items=[{"id": 1, "title": "삼성SDS, OpenAI 파트너십 확대"}],
        facts=[],
        evidence_snippets=[],
        sources=[
            {
                "url": "https://example.com/article-1",
                "title": "삼성SDS, OpenAI 파트너십 확대",
                "source_name": "테크뉴스",
            }
        ],
        metadata={},
    )


def _stub_deps() -> SupervisorDeps:
    """모든 child agent 를 mock 으로 교체해 LLM 호출 0."""
    issue_integrator = MagicMock()
    issue_integrator.integrate_input_bundle.return_value = {
        "is_valid_summary": True,
        "main_company": "samsung_sds",
        "integrated_text": "삼성SDS, OpenAI 와 파트너십 확대.",
        "fact_basis": [{"fact_id": "fact_001", "evidence_text": "MOU 체결"}],
        "key_numbers": [{"value": "3.54조원"}],
        "consolidated_facts": [{"fact": "MOU 체결"}],
        "confidence": 0.7,
    }

    analyzer = MagicMock()
    analyzer.analyze.return_value = {
        "is_valid_analysis": True,
        "analysis_summary": "글로벌 AI 협력 가속",
        "strategic_meaning": ["MSP 시장 영향"],
        "market_signal": "기업용 AI 전환 가속",
        "impact_level": "high",
        "risk_or_opportunity": "opportunity",
        "confidence": 0.8,
    }

    implication = MagicMock()
    implication.generate.return_value = {
        "is_valid_implication": True,
        "implication_scope": "peer_and_skax",
        "peer_implication": {
            "company_id": "samsung_sds",
            "company_name_ko": "삼성SDS",
            "peer_meaning": "삼성SDS 외부 협력 확대.",
            "sourced_evidence_ids": ["fact_001"],
        },
        "skax_implication": {
            "why_important": "SK AX 의 에이전틱AI 시장 잠식 가능.",
            "potential_impact": "MSP 입찰 경쟁 심화.",
            "opportunities": ["에이전틱AI PoC 추진"],
            "threats": ["MSP 입찰 점유율 하락"],
            "recommended_actions": [
                "에이전틱AI 협업 모델 PoC 제안서 작성 추진",
                "MSP 입찰 사전 자격 점검 착수",
            ],
            "business_line_mapping": ["에이전틱AI", "MSP"],
        },
        "follow_up_questions": ["a", "b", "c"],
        "watch_points": [],
        "confidence": 0.7,
        "evidence_label": "moderate",
        "provenance": {
            "generator": "ImplicationAgent",
            "prompt_version": "implication-v4.0",
            "model": "gpt-4o",
            "used_context_layers": [],
        },
    }

    context_builder = MagicMock()
    context_builder.build.return_value = AnalysisContext()

    profile_context_loader = MagicMock()
    profile_context_loader.load.return_value = ProfileContext(
        skax_profile={"business_lines": ["에이전틱AI", "MSP"]},
        peer_profiles={"samsung_sds": {"peer_id": "samsung_sds"}},
        sector_context={"selected_sector_ids": ["ax"]},
    )

    card_news_composer = MagicMock()
    card_news_composer.generate_from_analysis_package.return_value = {
        "id": "CN-20260520-0042",
        "company": "samsung_sds",
        "title": "삼성SDS, OpenAI 파트너십 확대",
        "summary_lines": ["1. 협업", "2. 시장", "3. 영향"],
    }

    return SupervisorDeps(
        issue_integrator=issue_integrator,
        analyzer=analyzer,
        implication_agent=implication,
        context_builder=context_builder,
        profile_context_loader=profile_context_loader,
        card_news_composer=card_news_composer,
    )


def test_supervisor_graph_happy_path_writes_card():
    deps = _stub_deps()
    graph = build_supervisor_graph(deps)
    issue_id = "11111111-1111-1111-1111-111111111111"
    with (
        patch(
            "src.pipeline.analysis_flow_graph.save_integrated_issue",
            return_value=issue_id,
        ) as save_integrated_issue,
        patch("src.pipeline.analysis_flow_graph.save_card_news", return_value="CN-OK"),
        patch("src.pipeline.analysis_flow_graph.save_pipeline_log"),
    ):
        result = graph.invoke(
            {
                "input_bundle": _stub_bundle(),
                "classification": {"sector": "ax", "event_type": "partnership"},
                "errors": [],
                "human_review_flags": [],
            }
        )
    assert result.get("card_news_id") == "CN-OK"
    assert result.get("analysis_package") is not None
    assert result.get("integrated_issue", {}).get("integrated_issue_id") == issue_id
    assert result.get("card_news_payload", {}).get("integrated_issue_id") == issue_id
    validation = result.get("validation")
    assert validation is not None
    assert validation.passed is True
    assert validation.metrics is not None
    # R-1: issue_integrate 가 가장 먼저 실행되어야 한다.
    deps.issue_integrator.integrate_input_bundle.assert_called_once()
    save_integrated_issue.assert_called_once()
    # ProfileContext 는 integrated_issue.main_company (= samsung_sds) 를 받았는지.
    profile_call = deps.profile_context_loader.load.call_args
    assert "samsung_sds" in (profile_call.kwargs.get("companies") or [])


def test_supervisor_graph_uses_injected_profile_context():
    deps = _stub_deps()
    graph = build_supervisor_graph(deps)
    injected_profile = ProfileContext(
        skax_profile={"company_id": "sk_ax", "business_areas": [{"name": "AI 운영"}]},
        peer_profiles={
            "samsung_sds": {
                "company_id": "samsung_sds",
                "schema_version": "peer-profile-snapshot-v1",
                "business_areas": [{"name": "클라우드"}],
            }
        },
        sector_context={"selected_sector_ids": ["ax"]},
    )

    with (
        patch(
            "src.pipeline.analysis_flow_graph.save_integrated_issue",
            return_value="11111111-1111-1111-1111-111111111111",
        ),
        patch("src.pipeline.analysis_flow_graph.save_card_news", return_value="CN-OK"),
        patch("src.pipeline.analysis_flow_graph.save_pipeline_log"),
    ):
        result = graph.invoke(
            {
                "input_bundle": _stub_bundle(),
                "classification": {"sector": "ax", "event_type": "partnership"},
                "profile_context": injected_profile,
                "errors": [],
                "human_review_flags": [],
            }
        )

    assert result.get("analysis_package") is not None
    deps.profile_context_loader.load.assert_not_called()
    context_call = deps.context_builder.build.call_args
    assert context_call.kwargs.get("profile_context") is injected_profile


def test_supervisor_graph_routes_human_review_on_fake_numeric():
    """integrated_issue 의 fact_basis 에 없는 수치를 implication 이 만들면 validate fail."""
    deps = _stub_deps()
    # implication 이 출처에 없는 99.9% 를 등장시킴.
    deps.implication_agent.generate.return_value["skax_implication"]["why_important"] = (
        "시장 점유율 99.9% 확보가 예상된다."
    )
    deps.issue_integrator.integrate_input_bundle.return_value["fact_basis"] = []
    deps.issue_integrator.integrate_input_bundle.return_value["key_numbers"] = []
    graph = build_supervisor_graph(deps)
    with (
        patch(
            "src.pipeline.analysis_flow_graph.save_integrated_issue",
            return_value="11111111-1111-1111-1111-111111111111",
        ),
        patch("src.pipeline.analysis_flow_graph.save_card_news"),
        patch("src.pipeline.analysis_flow_graph.save_pipeline_log"),
    ):
        result = graph.invoke(
            {
                "input_bundle": _stub_bundle(),
                "classification": {"sector": "ax"},
                "errors": [],
                "human_review_flags": [],
            }
        )
    validation = result.get("validation")
    assert validation is not None
    assert validation.passed is False
    assert "numeric_violations" in validation.to_dict()
    assert result.get("card_news_id") is None
    assert any("news:cluster_42" in f for f in (result.get("human_review_flags") or []))


def test_hard_validate_blocks_certainty_expression():
    impl: dict[str, Any] = {
        "is_valid_implication": True,
        "skax_implication": {
            "why_important": "반드시 SK AX 가 MSP 시장을 잠식할 것이다.",
            "potential_impact": "",
            "recommended_actions": [],
        },
        "peer_implication": {},
        "provenance": {"model": "gpt-4o"},
    }
    report = _hard_validate(
        integrated_issue={"is_valid_summary": True},
        analysis={"is_valid_analysis": True},
        implication=impl,
        evidence_payload={},
        sources=[{"url": "https://x"}],
    )
    assert report.certainty_warnings
    assert any("반드시" in w for w in report.certainty_warnings)
