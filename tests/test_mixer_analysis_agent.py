from __future__ import annotations

import asyncio
import json
from typing import Any

from src.agents import mixer_analysis_agent as mixer_module
from src.agents.mixer_analysis_agent import MixerAnalysisAgent
from src.api.mixer_schemas import MixerAnalysisRequest, MixerAnalysisResponse
from src.services.analysis_units import (
    QUALITY_SUMMARY_ONLY_FALLBACK,
    AnalysisUnit,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any, config: Any | None = None) -> _FakeResponse:
        del config
        self.prompts.append(prompt)
        return _FakeResponse(
            json.dumps(
                {
                    "mix_insight": "두 이슈는 기술을 운영 성과 근거로 제시합니다.",
                    "common_pattern": {
                        "finding": "운영 성과를 설명하는 신호가 반복됩니다.",
                        "rationale": "각 이슈의 분석과 시사점이 같은 판단 기준을 보입니다.",
                        "evidence": [
                            {"card_id": "CN-1", "text": "근거 1"},
                            {"card_id": "CN-2", "text": "근거 2"},
                        ],
                        "evidence_card_ids": ["CN-1", "CN-2"],
                    },
                    "comparison_point": {
                        "finding": "한쪽은 실행 기반, 다른 쪽은 성장 근거를 앞세웁니다.",
                        "rationale": "성과를 설명하는 방식이 다릅니다.",
                        "evidence": [
                            {"card_id": "CN-1", "text": "실행 기반"},
                            {"card_id": "CN-2", "text": "성장 근거"},
                        ],
                        "evidence_card_ids": ["CN-1", "CN-2"],
                    },
                    "hidden_conclusion": {
                        "finding": "핵심 신호는 기술이 성과 설명 근거로 쓰인다는 점입니다.",
                        "rationale": "두 이슈를 함께 볼 때 판단 기준 변화가 확인됩니다.",
                        "evidence": [
                            {"card_id": "CN-1", "text": "성과 근거 1"},
                            {"card_id": "CN-2", "text": "성과 근거 2"},
                        ],
                        "evidence_card_ids": ["CN-1", "CN-2"],
                    },
                    "recommended_action_basis": ["제안 메시지에 운영 성과 근거를 앞세웁니다."],
                    "action_details": [
                        {
                            "action": "제안 첫 장에 운영 성과 근거를 배치합니다.",
                            "why": "두 이슈 모두 성과 기준을 근거로 제시하기 때문입니다.",
                            "use_case": "제안 전략",
                            "evidence": [
                                {"card_id": "CN-1", "text": "근거 1"},
                                {"card_id": "CN-2", "text": "근거 2"},
                            ],
                            "evidence_card_ids": ["CN-1", "CN-2"],
                        }
                    ],
                    "recommended_actions": ["제안 첫 장에 운영 성과 근거를 배치합니다."],
                    "connections": [
                        {
                            "source_card_id": "CN-1",
                            "target_card_id": "CN-2",
                            "label": "unsupported",
                            "weight": 2,
                        }
                    ],
                    "cross_card_findings": [
                        {
                            "finding": "enum 방어 테스트",
                            "evidence_card_ids": ["CN-1", "CN-2"],
                            "pattern_type": "unsupported",
                        }
                    ],
                    "reasoning_steps": [
                        {
                            "step_idx": 1,
                            "phase": "unsupported",
                            "inputs_used": ["CN-1", "CN-2"],
                            "answer": "enum 방어 테스트",
                        }
                    ],
                    "sources_used": ["CN-1", "CN-2"],
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )
        )


def _unit(card_id: str, issue_id: str, *, flags: list[str] | None = None) -> AnalysisUnit:
    return AnalysisUnit(
        integrated_issue_id=issue_id,
        card_id=card_id,
        integrated_issue={
            "main_issue": f"{card_id} 통합 이슈",
            "integrated_text": f"{card_id} 통합 상세",
            "business_signals": [{"signal": f"{card_id} 사업 신호"}],
        },
        analysis={
            "analysis_summary": f"{card_id} 분석 상세",
            "market_signal": f"{card_id} 시장 신호",
            "confidence": 0.8,
        },
        implication={
            "skax_implication": {
                "why_important": f"{card_id} 시사점",
                "recommended_actions": [f"{card_id} 대응"],
            }
        },
        classification={"sector": "ax"},
        validation={"sc_score": 0.8},
        source_raw_article_ids=[1 if card_id == "CN-1" else 2],
        evidence_refs=[{"evidence_ref_id": f"{card_id}-fact", "text": "상세 근거"}],
        display_summary=["표시 요약"],
        card={"id": card_id, "title": f"{card_id} 제목", "company": "samsung_sds"},
        quality_flags=flags or [],
    )


def test_mixer_accepts_integrated_issue_ids_and_exposes_sources(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]
    fake_llm = _FakeLLM()
    calls: dict[str, list[str]] = {}

    def _load(issue_ids_arg: list[str]) -> list[AnalysisUnit]:
        calls["issue_ids"] = issue_ids_arg
        return units

    monkeypatch.setattr(mixer_module, "load_analysis_units_by_integrated_issue_ids", _load)
    monkeypatch.setattr(mixer_module, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        mixer_module,
        "_repair_mixer_result_quality",
        lambda **kwargs: kwargs["result"],
    )
    monkeypatch.setattr(mixer_module, "_generate_mix_level_implication", lambda **kwargs: {})

    result = asyncio.run(MixerAnalysisAgent().analyze(integrated_issue_ids=issue_ids))

    assert calls["issue_ids"] == issue_ids
    assert result["sources_used"] == ["CN-1", "CN-2"]
    assert result["source_integrated_issue_ids"] == issue_ids
    assert result["provenance"]["source_integrated_issue_ids"] == issue_ids
    assert result["radar_axes"][0]["calculation"]
    assert result["radar_axes"][0]["meaning"]
    assert isinstance(result["follow_up_checks"], list)
    assert "통합 상세" in fake_llm.prompts[0]
    assert "표시 요약(최하위 보조)" in fake_llm.prompts[0]
    assert "임원/의사결정자" in fake_llm.prompts[0]
    assert len(result["recommended_actions"]) == 3
    assert all("경영진 리뷰 안건" not in action for action in result["recommended_actions"])
    assert all("제안 첫 장" not in action for action in result["recommended_actions"])
    assert all("모니터링" not in action for action in result["recommended_actions"])
    assert all(action.strip() for action in result["recommended_actions"])
    assert any(
        "고객군" in action and "책임 조직" in action for action in result["recommended_actions"]
    )
    assert any(detail["use_case"] == "사업 우선순위" for detail in result["action_details"])
    response = MixerAnalysisResponse.model_validate(result)
    assert all(connection.label in {"similar", "contrast"} for connection in response.connections)
    assert all(
        finding.pattern_type in {"convergent_strategy", "divergent_strategy", "acceleration_signal"}
        for finding in response.cross_card_findings
    )
    reasoning_phases = [step.phase for step in response.reasoning_steps]
    assert reasoning_phases[0] == "per_card"
    assert "cross_card" in reasoning_phases
    assert reasoning_phases[-1] == "synthesis"


def test_mixer_card_ids_are_interpreted_as_analysis_units(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]

    monkeypatch.setattr(mixer_module, "load_analysis_units_by_card_ids", lambda ids: units)
    monkeypatch.setattr(mixer_module, "_get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        mixer_module,
        "_repair_mixer_result_quality",
        lambda **kwargs: kwargs["result"],
    )
    monkeypatch.setattr(mixer_module, "_generate_mix_level_implication", lambda **kwargs: {})

    result = asyncio.run(MixerAnalysisAgent().analyze(card_ids=["CN-1", "CN-2"]))

    assert result["source_integrated_issue_ids"] == issue_ids


def test_mixer_quick_mode_skips_mix_level_implication(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]

    monkeypatch.setattr(
        mixer_module, "load_analysis_units_by_integrated_issue_ids", lambda ids: units
    )
    monkeypatch.setattr(mixer_module, "_get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        mixer_module,
        "_repair_mixer_result_quality",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("quick mode must not repair via LLM")
        ),
    )
    monkeypatch.setattr(
        mixer_module,
        "_generate_mix_level_implication",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("quick mode must not call implication")
        ),
    )

    result = asyncio.run(
        MixerAnalysisAgent().analyze(integrated_issue_ids=issue_ids, analysis_mode="quick")
    )

    assert result["provenance"]["analysis_mode"] == "quick"
    assert result["provenance"]["analysis_quality"] == "fast"
    assert result["analysis_depth"]["mode"] == "quick"
    assert result["analysis_depth"]["omitted_steps"]
    assert result["deep_dive_sections"] == []
    assert len(result["recommended_actions"]) == 3


def test_mixer_deep_mode_runs_quality_and_implication(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]
    calls = {"repair": 0, "implication": 0}

    monkeypatch.setattr(
        mixer_module, "load_analysis_units_by_integrated_issue_ids", lambda ids: units
    )
    monkeypatch.setattr(mixer_module, "_get_llm", lambda: _FakeLLM())

    def _repair(**kwargs):
        calls["repair"] += 1
        return kwargs["result"]

    def _implication(**kwargs):
        del kwargs
        calls["implication"] += 1
        return {}

    monkeypatch.setattr(mixer_module, "_repair_mixer_result_quality", _repair)
    monkeypatch.setattr(mixer_module, "_generate_mix_level_implication", _implication)

    result = asyncio.run(
        MixerAnalysisAgent().analyze(integrated_issue_ids=issue_ids, analysis_mode="deep")
    )

    assert calls == {"repair": 1, "implication": 1}
    assert result["provenance"]["analysis_mode"] == "deep"
    assert result["provenance"]["analysis_quality"] == "detailed"
    assert result["analysis_depth"]["mode"] == "deep"
    assert result["deep_dive_sections"]


def test_mixer_quality_flags_lower_confidence_and_warn(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [
        _unit("CN-1", issue_ids[0], flags=[QUALITY_SUMMARY_ONLY_FALLBACK]),
        _unit("CN-2", issue_ids[1]),
    ]

    monkeypatch.setattr(
        mixer_module,
        "load_analysis_units_by_integrated_issue_ids",
        lambda ids: units,
    )
    monkeypatch.setattr(mixer_module, "_get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        mixer_module,
        "_repair_mixer_result_quality",
        lambda **kwargs: kwargs["result"],
    )
    monkeypatch.setattr(mixer_module, "_generate_mix_level_implication", lambda **kwargs: {})

    result = asyncio.run(MixerAnalysisAgent().analyze(integrated_issue_ids=issue_ids))

    assert result["confidence"] == 0.55
    assert result["provenance"]["quality_flags"] == [QUALITY_SUMMARY_ONLY_FALLBACK]
    assert "통합 분석 연결이 제한" in result["warning"]
    assert QUALITY_SUMMARY_ONLY_FALLBACK not in result["warning"]
    assert "confidence <" not in result["warning"]
    assert "다른 카드 조합" not in result["warning"]


def test_mixer_missing_credentials_error_is_sanitized(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]

    monkeypatch.setattr(
        mixer_module,
        "load_analysis_units_by_integrated_issue_ids",
        lambda ids: units,
    )

    class _MissingCredentialLLM:
        def invoke(self, prompt, config=None):  # noqa: ANN001
            del prompt, config
            raise ValueError(
                "Missing credentials. Please pass an `api_key`, `workload_identity`, "
                "`admin_api_key`, or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY` "
                "environment variable."
            )

    monkeypatch.setattr(mixer_module, "_get_llm", lambda: _MissingCredentialLLM())

    result = asyncio.run(
        MixerAnalysisAgent().analyze(integrated_issue_ids=issue_ids, analysis_mode="quick")
    )

    assert result["confidence"] == 0.0
    assert "AI 모델 인증 정보가 설정되지 않아" in result["warning"]
    assert "Missing credentials" not in result["warning"]
    assert "api_key" not in result["warning"]


def test_mixer_schema_requires_card_or_integrated_issue_ids():
    assert MixerAnalysisRequest(integrated_issue_ids=["a"]).integrated_issue_ids == ["a"]
    assert (
        MixerAnalysisRequest(integrated_issue_ids=["a"], analysis_mode="deep").analysis_mode
        == "deep"
    )

    try:
        MixerAnalysisRequest()
    except ValueError as exc:
        assert "integrated_issue_ids" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected validation error")


def test_mixer_action_quality_rejects_program_artifacts():
    assert mixer_module._is_generic_action_text("제안서 첫 장에 운영 성과 근거를 배치합니다.")
    assert mixer_module._is_generic_action_text("후속 모니터링 항목을 다음 반복 신호로 둡니다.")
    assert not mixer_module._is_generic_action_text(
        "금융 고객군 우선순위를 정하고 오퍼링 책임 조직과 리스크 승인 권한을 지정한다."
    )
