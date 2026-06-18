# 작성일: 2026-05-21
# 작성자: 최종민
# 변경이력:
#   2026-05-21 최종민 — Layer B 분석 파이프라인(W1~W5) 구축의 일부로 추가
"""W1-1 — ImplicationAgent LLM 기반 시사점 Agent 테스트.

LLM 호출은 mock 처리하여 deterministic. fallback 경로도 함께 검증.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from src.agents.implication_agent import ImplicationAgent
from src.analysis.models import (
    AnalysisContext,
    AnalysisInputBundle,
    AnalysisInputMetadata,
    ClassificationPayload,
    ProfileContext,
    TimelineEntry,
)


def _bundle() -> AnalysisInputBundle:
    return AnalysisInputBundle(
        bundle_id="news:42",
        cluster_id="42",
        source_type="news",
        companies=["samsung_sds"],
        sectors=["ax"],
        event_type="partnership",
        items=[],
        facts=[],
        evidence_snippets=[],
        sources=[{"title": "삼성SDS, OpenAI 파트너십", "url": "https://x"}],
        metadata=AnalysisInputMetadata(
            representative_id=100,
            cluster_article_ids=[100],
            classification=ClassificationPayload(
                sector="ax",
                event_type="partnership",
                importance="high",
            ),
            created_at="2026-05-20T03:00:00+09:00",
        ).to_dict(),
    )


def _integrated_issue() -> dict[str, Any]:
    return {
        "is_valid_summary": True,
        "main_company": "samsung_sds",
        "integrated_text": "삼성SDS, OpenAI 와 한국 시장 협업 확대.",
        "fact_basis": [
            {"fact_id": "fact_001", "evidence_text": "MOU 체결"},
        ],
        "key_numbers": [{"value": "3.54조원"}],
        "consolidated_facts": [{"fact": "MOU 체결"}],
        "confidence": 0.7,
    }


def _analysis() -> dict[str, Any]:
    return {
        "is_valid_analysis": True,
        "analysis_summary": "글로벌 AI 협력 가속 신호",
        "strategic_meaning": ["MSP 시장 영향"],
        "market_signal": "기업용 AI 전환 가속",
        "impact_level": "high",
        "risk_or_opportunity": "opportunity",
        "confidence": 0.8,
    }


def _profile() -> ProfileContext:
    return ProfileContext(
        skax_profile={"company_id": "sk_ax", "business_lines": ["에이전틱AI", "MSP"]},
        peer_profiles={
            "samsung_sds": {
                "peer_id": "samsung_sds",
                "company_name": "삼성SDS",
            }
        },
        sector_context={"selected_sector_ids": ["ax"]},
    )


def _fake_llm_response(payload: dict[str, Any]) -> Any:
    response = MagicMock()
    response.content = json.dumps(payload, ensure_ascii=False)
    return response


def test_implication_agent_llm_path_returns_v4_schema():
    fake_payload = {
        "is_valid_implication": True,
        "implication_scope": "peer_and_skax",
        "peer_implication": {
            "company_id": "samsung_sds",
            "company_name_ko": "삼성SDS",
            "peer_meaning": "삼성SDS 외부 협력 확대.",
            "capability_change": "AI 인력 +15%.",
            "precedent_link": None,
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
        "follow_up_questions": ["...", "...", "..."],
        "watch_points": ["1분기 매출"],
        "confidence": 0.78,
        "evidence_label": "sufficient",
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(fake_payload))
    agent = ImplicationAgent(llm=llm)
    result = agent.generate(
        input_bundle=_bundle(),
        integrated_issue=_integrated_issue(),
        analysis=_analysis(),
        profile_context=_profile(),
    )
    assert result["is_valid_implication"] is True
    assert result["peer_implication"]["company_id"] == "samsung_sds"
    assert result["peer_implication"]["company_name_ko"] == "삼성SDS"
    assert "에이전틱AI" in result["skax_implication"]["business_line_mapping"]
    # backwards-compat flatten
    assert isinstance(result.get("opportunities"), list)
    assert isinstance(result.get("recommended_actions"), list)
    assert result["provenance"]["prompt_version"] == "implication-v4.0"


def test_implication_agent_v5_when_context_supplied():
    fake_payload = {
        "is_valid_implication": True,
        "peer_implication": {
            "company_id": "samsung_sds",
            "company_name_ko": "삼성SDS",
            "peer_meaning": "...",
        },
        "skax_implication": {
            "why_important": "AI 시장 모멘텀 가속.",
            "potential_impact": "MSP 경쟁 심화.",
            "recommended_actions": ["PoC 추진"],
            "business_line_mapping": ["MSP"],
        },
        "follow_up_questions": ["a", "b", "c"],
        "watch_points": [],
        "confidence": 0.7,
        "evidence_label": "moderate",
    }
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=_fake_llm_response(fake_payload))

    ctx = AnalysisContext()
    ctx.peer_event_timeline_recent = [
        TimelineEntry(
            company_id="samsung_sds",
            event_date="2026-05-10",
            card_id="CN-1",
            event_type="partnership",
            sector="ax",
            headline="h",
            importance="high",
            importance_score=0.9,
        )
    ]
    result = ImplicationAgent(llm=llm).generate(
        input_bundle=_bundle(),
        integrated_issue=_integrated_issue(),
        analysis=_analysis(),
        profile_context=_profile(),
        analysis_context=ctx,
    )
    assert result["provenance"]["prompt_version"] == "implication-v5.0"
    assert "peer_event_timeline_recent" in result["provenance"]["used_context_layers"]


def test_implication_agent_fallback_when_inputs_insufficient():
    agent = ImplicationAgent()
    invalid_summary = {"is_valid_summary": False}
    result = agent.generate(
        input_bundle=_bundle(),
        integrated_issue=invalid_summary,
        analysis={"is_valid_analysis": False},
        profile_context=_profile(),
    )
    assert result["provenance"]["generator"] == "ImplicationGenerator"


def test_implication_agent_fallback_when_llm_errors():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM timeout")
    agent = ImplicationAgent(llm=llm)
    result = agent.generate(
        input_bundle=_bundle(),
        integrated_issue=_integrated_issue(),
        analysis=_analysis(),
        profile_context=_profile(),
    )
    assert result["provenance"]["generator"] == "ImplicationGenerator"
