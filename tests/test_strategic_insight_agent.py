"""StrategicInsightAgent unit tests with mock LLM response."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "strategic_insight_agent_input.mock.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _load_env_file_for_integration() -> None:
    """Load local .env without shell expansion.

    DATABASE_URL may contain "$" in the password. `source .env` expands it and
    breaks authentication, so integration tests read the file directly.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    wanted = {"DATABASE_URL", "OPENAI_API_KEY"}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in wanted:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
    try:
        from src.db import postgres

        postgres.reconfigure_from_env()
    except Exception:
        pass


_load_env_file_for_integration()

from src.agents.strategic_insight_agent import StrategicInsightAgent  # noqa: E402


def _fake_llm_response(payload: dict[str, Any]) -> Any:
    response = MagicMock()
    response.content = json.dumps(payload, ensure_ascii=False)
    return response


def _companies_from_integrated_issue(integrated_issue: dict[str, Any]) -> list[str]:
    companies: list[str] = []
    for company_id in [
        integrated_issue.get("main_company"),
        *(integrated_issue.get("mentioned_peer_companies") or []),
    ]:
        if company_id and company_id not in companies:
            companies.append(company_id)
    return companies


def _assert_strategic_insight_schema(result: dict[str, Any]) -> None:
    assert set(result) == {
        "is_valid_strategic_insight",
        "analysis",
        "implication",
    }
    assert set(result["analysis"]) == {
        "is_valid_analysis",
        "analysis_scope",
        "analysis_summary",
        "strategic_meaning",
        "market_signal",
        "impact_level",
        "impact_reason",
        "risk_or_opportunity",
        "confidence",
        "reason",
    }
    assert set(result["implication"]) == {
        "is_valid_implication",
        "implication_scope",
        "peer_implication",
        "skax_implication",
        "follow_up_questions",
        "watch_points",
        "confidence",
        "evidence_label",
        "provenance",
    }
    assert set(result["implication"]["peer_implication"]) == {
        "company_id",
        "company_name_ko",
        "peer_meaning",
        "capability_change",
        "sourced_evidence_ids",
    }
    assert set(result["implication"]["skax_implication"]) == {
        "why_important",
        "potential_impact",
        "opportunities",
        "threats",
        "recommended_actions",
        "business_line_mapping",
    }


def test_strategic_insight_agent_returns_separated_blocks():
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": (
                "삼성SDS의 기업용 AI 협력 확대는 AI 전환 지원 역량을 강화하는 신호입니다."
            ),
            "strategic_meaning": [
                "기업용 AI 전환 지원을 중심으로 피어사의 사업 메시지가 강화됩니다.",
                "클라우드 기반 운영 역량이 AI 도입 지원과 함께 묶이고 있습니다.",
            ],
            "market_signal": "기업 고객의 AI 전환 수요가 운영 지원 역량과 함께 평가되고 있습니다.",
            "impact_level": "high",
            "impact_reason": (
                "통합 이슈에서 기업용 AI 협력 확대와 전환 지원 신호가 함께 확인됩니다."
            ),
            "risk_or_opportunity": "opportunity",
            "confidence": 0.82,
            "reason": "article:456:title 근거와 business_signals를 기반으로 해석했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": "samsung_sds",
                "company_name_ko": "삼성SDS",
                "peer_meaning": (
                    "삼성SDS는 기업용 AI 협력을 통해 AI 전환 지원 메시지를 강화하고 있습니다."
                ),
                "capability_change": "클라우드와 AI 전환 지원 역량을 결합하는 방향입니다.",
                "sourced_evidence_ids": ["article:456:title", "not-in-input"],
            },
            "skax_implication": {
                "why_important": (
                    "SK AX도 기업 고객의 AI 도입 이후 운영 지원 범위를 함께 제시해야 합니다."
                ),
                "potential_impact": (
                    "공공AX와 AI 운영 제안에서 운영 책임 설명의 중요도가 높아질 수 있습니다."
                ),
                "opportunities": ["기업용 AI 운영 지원 패키지 제안 강화"],
                "threats": ["피어사의 AI 전환 지원 메시지 선점"],
                "recommended_actions": ["공공AX와 AI 운영 연계 제안 구조 정리"],
                "business_line_mapping": ["공공AX", "AI 운영", "invalid"],
            },
            "follow_up_questions": ["협력 범위가 실제 고객 사례로 이어졌는가?"],
            "watch_points": ["기업용 AI 전환 수주 사례"],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))

    payload = _fixture()
    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is True
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is True
    assert result["analysis"]["analysis_scope"] == "peer_and_industry"
    assert result["implication"]["is_valid_implication"] is True
    assert result["implication"]["implication_scope"] == "peer_and_skax"
    assert result["implication"]["peer_implication"]["company_id"] == "samsung_sds"
    # Only fact_ids present in IntegratedIssue are kept, with referenced facts augmented.
    assert "article:456:title" in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    assert "not-in-input" not in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    # Only business lines from the supplied SK AX profile candidates are kept.
    assert result["implication"]["skax_implication"]["business_line_mapping"] == [
        "공공AX",
        "AI 운영",
    ]
    assert result["implication"]["provenance"]["generator"] == "StrategicInsightAgent"
    assert (
        result["implication"]["provenance"]["prompt_version"]
        == "strategic-insight-v1.4-specific-actions"
    )


def test_strategic_insight_agent_empty_when_integrated_issue_invalid():
    payload = _fixture()
    result = StrategicInsightAgent(llm=MagicMock()).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue={"is_valid_summary": False},
        classification=payload["classification"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is False
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is False
    assert result["implication"]["is_valid_implication"] is False


def test_strategic_insight_agent_filters_business_lines_by_profile_candidates():
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 공공 AI 협력 확대는 보안형 AX 수요 대응 신호입니다.",
            "strategic_meaning": ["공공 AX 적용 사례가 늘고 있습니다."],
            "market_signal": "공공 AI 도입에서 보안 운영 기준이 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 공공 적용과 보안 운영 신호가 함께 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": "samsung_sds",
                "company_name_ko": "삼성SDS",
                "peer_meaning": "공공 AI 적용 메시지를 강화하고 있습니다.",
                "capability_change": "보안형 AI 운영 역량이 강조됩니다.",
                "sourced_evidence_ids": ["article:456:title"],
            },
            "skax_implication": {
                "why_important": "SK AX도 공공 AI 운영 책임을 제안해야 합니다.",
                "potential_impact": "공공 고객의 보안 운영 평가 기준 대응이 중요해질 수 있습니다.",
                "opportunities": ["공공 AX 운영 패키지 제안"],
                "threats": ["피어사의 공공 AI 레퍼런스 메시지 선점"],
                "recommended_actions": ["공공 AX 보안 운영 시나리오를 제안서에 반영"],
                "business_line_mapping": ["에이전틱AI", "공공AX", "보안AX", "invalid"],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    payload = _fixture()
    profile_context = {
        **payload["profile_context"],
        "skax_profile": {
            "company_id": "sk_ax",
            "business_lines": ["공공AX", "보안AX"],
        },
    }

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["implication"]["skax_implication"]["business_line_mapping"] == [
        "공공AX",
        "보안AX",
    ]


def test_strategic_insight_agent_empty_business_line_candidates_returns_empty_mapping():
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 공공 AI 협력 확대는 보안형 AX 수요 대응 신호입니다.",
            "strategic_meaning": ["공공 AX 적용 사례가 늘고 있습니다."],
            "market_signal": "공공 AI 도입에서 보안 운영 기준이 중요해지고 있습니다.",
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 공공 적용과 보안 운영 신호가 함께 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": "samsung_sds",
                "company_name_ko": "삼성SDS",
                "peer_meaning": "공공 AI 적용 메시지를 강화하고 있습니다.",
                "capability_change": "보안형 AI 운영 역량이 강조됩니다.",
                "sourced_evidence_ids": ["article:456:title"],
            },
            "skax_implication": {
                "why_important": "SK AX도 공공 AI 운영 책임을 제안해야 합니다.",
                "potential_impact": "공공 고객의 보안 운영 평가 기준 대응이 중요해질 수 있습니다.",
                "opportunities": ["공공 AX 운영 패키지 제안"],
                "threats": ["피어사의 공공 AI 레퍼런스 메시지 선점"],
                "recommended_actions": ["공공 AX 보안 운영 시나리오를 제안서에 반영"],
                "business_line_mapping": ["에이전틱AI", "공공AX"],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    payload = _fixture()
    profile_context = {
        **payload["profile_context"],
        "skax_profile": {"company_id": "sk_ax"},
    }

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    assert result["implication"]["skax_implication"]["business_line_mapping"] == []


def test_strategic_insight_agent_concretizes_generic_recommended_actions():
    llm_payload = {
        "is_valid_strategic_insight": True,
        "analysis": {
            "is_valid_analysis": True,
            "analysis_scope": "peer_and_industry",
            "analysis_summary": "피어사의 보안형 생성형 AI 적용 신호입니다.",
            "strategic_meaning": ["챗GPT 에듀와 공공 협업 도구 적용이 확인됩니다."],
            "market_signal": "기업용 생성형 AI 도입에서 데이터 보호와 보안 검증이 중요합니다.",
            "impact_level": "high",
            "impact_reason": "통합 이슈에서 교육 PoC와 공공 협업 도구 적용이 확인됩니다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.8,
            "reason": "입력 fact_basis를 기반으로 판단했습니다.",
        },
        "implication": {
            "is_valid_implication": True,
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": "samsung_sds",
                "company_name_ko": "삼성SDS",
                "peer_meaning": (
                    "챗GPT 에듀와 브리티웍스 적용으로 교육 및 공공 AX 적용이 확인됩니다."
                ),
                "capability_change": "데이터 보호와 보안성을 강조한 AI 솔루션 적용 범위 확대",
                "sourced_evidence_ids": ["article:456:title"],
            },
            "skax_implication": {
                "why_important": "SK AX 제안에서 데이터 보호와 보안 검증 기준을 보여줘야 합니다.",
                "potential_impact": "고객은 보안성과 운영 책임 범위를 먼저 비교할 수 있습니다.",
                "opportunities": ["데이터 보호와 보안성을 강조한 제안서 개발"],
                "threats": ["보안 검증 근거 비교 압박"],
                "recommended_actions": [
                    "제안서에 데이터 보호와 보안성 강조",
                    "공공 및 교육 부문에서의 PoC 레퍼런스 강화",
                    "운영 모델에서 데이터 보관 위치와 운영 책임 범위를 강화합니다.",
                ],
                "business_line_mapping": ["공공AX"],
            },
            "follow_up_questions": [],
            "watch_points": [],
            "confidence": 0.78,
            "evidence_label": "moderate",
        },
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(llm_payload))
    payload = _fixture()
    profile_context = {
        **payload["profile_context"],
        "skax_profile": {
            "company_id": "sk_ax",
            "business_lines": ["공공AX"],
        },
    }

    result = StrategicInsightAgent(llm=llm).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=profile_context,
        analysis_context=payload["analysis_context"],
    )

    actions = result["implication"]["skax_implication"]["recommended_actions"]
    assert actions == [
        "제안서에 데이터 보호와 보안성 강조에 대해 적용 범위, 책임 범위, 검증 기준을 구체화합니다.",
        (
            "공공 및 교육 부문에서의 PoC 레퍼런스 강화에 대해 적용 범위, "
            "책임 범위, 검증 기준을 구체화합니다."
        ),
        "운영 모델에서 데이터 보관 위치와 운영 책임 범위를 강화합니다.",
    ]
    assert (
        "article:456:knou_poc" in result["implication"]["peer_implication"]["sourced_evidence_ids"]
    )
    assert "article:457:onai" in result["implication"]["peer_implication"]["sourced_evidence_ids"]


def test_strategic_insight_agent_uses_legacy_fallback_on_malformed_llm_output():
    payload = _fixture()
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response({"unexpected": "shape"}))
    fallback_analyzer = MagicMock()
    fallback_analyzer.analyze.return_value = {
        "is_valid_analysis": True,
        "analysis_summary": "fallback analysis",
    }
    fallback_implication = MagicMock()
    fallback_implication.generate.return_value = {
        "is_valid_implication": True,
        "peer_implication": {"peer_meaning": "fallback peer"},
        "skax_implication": {"why_important": "fallback skax"},
    }

    result = StrategicInsightAgent(
        llm=llm,
        fallback_analyzer=fallback_analyzer,
        fallback_implication_agent=fallback_implication,
    ).generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=payload["integrated_issue"],
        classification=payload["classification"],
        profile_context=payload["profile_context"],
        analysis_context=payload["analysis_context"],
    )

    assert result["is_valid_strategic_insight"] is True
    _assert_strategic_insight_schema(result)
    fallback_analyzer.analyze.assert_called_once()
    fallback_implication.generate.assert_called_once()


def test_strategic_insight_agent_with_mock_issue_and_common_db_profile():
    _load_env_file_for_integration()
    if os.getenv("RUN_STRATEGIC_INSIGHT_INTEGRATION") != "1":
        pytest.skip("Set RUN_STRATEGIC_INSIGHT_INTEGRATION=1 to call common DB and real LLM.")
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for common DB profile lookup.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the real StrategicInsightAgent LLM call.")

    from src.services.profile_context_loader import ProfileContextLoader

    payload = _fixture()
    integrated_issue = payload["integrated_issue"]
    classification = payload["classification"]
    sectors = classification.get("sectors") or [classification.get("sector")]

    profile_context = (
        ProfileContextLoader()
        .load(
            companies=_companies_from_integrated_issue(integrated_issue),
            sectors=[sector for sector in sectors if sector],
            event_type=classification.get("event_type"),
            strict=True,
        )
        .to_dict()
    )
    assert profile_context["peer_profiles"]
    peer_profile = profile_context["peer_profiles"].get(integrated_issue["main_company"]) or {}
    assert any(
        peer_profile.get(key)
        for key in (
            "schema_version",
            "profile_snapshot_version",
            "one_liner",
            "company_summary",
            "business_areas",
            "core_capabilities",
        )
    ), "common DB peer profile snapshot was not loaded; fail-soft fallback was returned."
    skax_profile = profile_context.get("skax_profile") or {}
    print(
        "[profile_context]",
        json.dumps(
            {
                "peer_profile_loaded": {
                    "company_id": peer_profile.get("company_id"),
                    "schema_version": peer_profile.get("schema_version"),
                    "profile_snapshot_version": peer_profile.get("profile_snapshot_version"),
                    "has_one_liner": bool(peer_profile.get("one_liner")),
                    "business_area_count": len(peer_profile.get("business_areas") or []),
                },
                "skax_profile_loaded": {
                    "company_id": skax_profile.get("company_id"),
                    "schema_version": skax_profile.get("schema_version"),
                    "profile_snapshot_version": skax_profile.get("profile_snapshot_version"),
                    "has_one_liner": bool(skax_profile.get("one_liner")),
                    "business_lines": skax_profile.get("business_lines"),
                    "business_area_count": len(skax_profile.get("business_areas") or []),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    result = StrategicInsightAgent().generate(
        input_bundle=payload["input_bundle"],
        integrated_issue=integrated_issue,
        classification=classification,
        profile_context=profile_context,
        analysis_context=payload.get("analysis_context") or {},
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    assert result["is_valid_strategic_insight"] is True
    _assert_strategic_insight_schema(result)
    assert result["analysis"]["is_valid_analysis"] is True
    assert result["implication"]["is_valid_implication"] is True
    assert result["implication"]["provenance"]["generator"] == "StrategicInsightAgent"
