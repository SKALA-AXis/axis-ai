from __future__ import annotations

import json
from typing import Any

from src.agents.issue_integration_agent import analysis_input_bundle_from_articles
from src.analysis.analyzer import StrategicAnalyzer
from src.analysis.models import AnalysisResult


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLlm:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config: Any | None = None) -> _FakeResponse:
        del config
        self.prompts.append(prompt)
        return _FakeResponse(json.dumps(self.payload, ensure_ascii=False))


def test_analysis_agent_outputs_multi_source_content_contract(monkeypatch):
    fake_llm = _FakeLlm(
        {
            "is_valid_analysis": True,
            "source_profile": {
                "source_family": "filing",
                "source_type": "dart",
                "analysis_lens": "공시 항목과 사업 세그먼트 변화를 분리한다.",
                "materiality_focus": ["financial", "operation", "risk"],
            },
            "content_analysis": {
                "core_thesis": "클라우드와 AX 사업 확대가 공시 본문에 반복된다.",
                "key_developments": ["AX 사업 확대"],
                "strategic_vectors": ["AI AX"],
                "financial_or_operating_readouts": ["매출액 10억원"],
                "risk_factors": ["전망 표현 포함"],
                "timing_and_commitment": "계획 성격",
                "evidence_gaps": ["고객별 매출 부재"],
            },
            "detailed_findings": [
                {
                    "finding_type": "strategy",
                    "finding": "AX 사업을 성장 축으로 제시했다.",
                    "evidence_refs": ["fact_1"],
                    "source_article_ids": [10],
                    "confidence": 0.8,
                }
            ],
            "evidence_map": [
                {
                    "claim": "AX 사업 확대",
                    "evidence": "사업의 내용",
                    "fact_ids": ["fact_1"],
                    "source_article_ids": [10],
                }
            ],
            "uncertainty_notes": ["전망 표현은 확정 실적으로 해석하지 않는다."],
            "analysis_summary": "AX 사업 확대 신호가 공시에서 확인된다.",
            "strategic_meaning": ["AX 포트폴리오 강화"],
            "market_signal": "기업 AX 전환 수요 대응",
            "impact_level": "medium",
            "impact_reason": "공시 근거는 있으나 고객별 수치가 제한적이다.",
            "risk_or_opportunity": "opportunity",
            "confidence": 0.78,
            "handoff": {
                "summary_agent_inputs": ["AX 사업 확대"],
                "insight_agent_inputs": ["포트폴리오 강화"],
                "skax_response_agent_inputs": ["AX 경쟁 구도 관찰"],
            },
            "reason": "fact 기반 분석",
        }
    )
    monkeypatch.setattr("src.analysis.analyzer._get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        "src.observability.tracing_config",
        lambda **kwargs: {},
        raising=False,
    )

    result = StrategicAnalyzer().analyze(
        integrated_issue={
            "is_valid_summary": True,
            "cluster_id": 10,
            "representative_id": 10,
            "main_company": "samsung_sds",
            "integrated_text": "사업의 내용에서 AX 사업 확대가 언급됐다.",
            "consolidated_facts": [
                {
                    "fact_id": "fact_1",
                    "fact": "AX 사업 확대",
                    "source_article_ids": [10],
                    "evidence_texts": ["사업의 내용"],
                }
            ],
            "fact_basis": [{"fact_ids": ["fact_1"], "evidence_text": "사업의 내용"}],
        },
        classification={"sector": "ax", "event_type": "general_update"},
        cluster_metadata={
            "bundle_id": "dart:10",
            "source_type": "dart",
            "source_types": ["dart"],
            "document_source_family": "filing",
            "cluster_size": 1,
            "source_count": 1,
        },
    )

    typed = AnalysisResult.from_dict(result)
    assert typed.analysis_mode == "multi_source_document_intelligence"
    assert typed.source_profile["source_family"] == "filing"
    assert typed.content_analysis["core_thesis"].startswith("클라우드")
    assert typed.detailed_findings[0]["finding_type"] == "strategy"
    assert typed.evidence_map[0]["fact_ids"] == ["fact_1"]
    assert typed.handoff["skax_response_agent_inputs"] == ["AX 경쟁 구도 관찰"]
    assert "Evidence-first Multi-source Document Intelligence" in fake_llm.prompts[0]


def test_analysis_input_bundle_promotes_parser_result_chunks_to_facts():
    bundle = analysis_input_bundle_from_articles(
        cluster_id=10,
        representative_id=10,
        articles=[
            {
                "id": 10,
                "title": "사업보고서",
                "source_type": "dart",
                "source_name": "dart",
                "company": ["samsung_sds"],
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
    assert "클라우드 MSP" in fact_text
    assert any(fact.get("source_chunk_uid") == "business:1" for fact in bundle.facts)
