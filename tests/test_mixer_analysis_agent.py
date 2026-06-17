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
    analysis_unit_from_card,
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
                    "radar_axis_interpretations": [
                        {
                            "axis": "peer_strategic_shift",
                            "analysis_prompt": "전략 전환 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "전략 전환 축은 이번 묶음에서 근거 부족으로 보며, "
                                "신사업이나 M&A 신호가 반복되는지 확인해야 합니다."
                            ),
                        },
                        {
                            "axis": "tech_investment",
                            "analysis_prompt": "기술 투자 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "기술 투자 축은 두 이슈의 기술 근거가 운영 성과 설명에 "
                                "쓰인다는 점으로 해석됩니다."
                            ),
                        },
                        {
                            "axis": "market_position",
                            "analysis_prompt": "시장 포지션 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "시장 포지션 축은 여러 peer의 성과 근거가 경쟁 위치 판단으로 "
                                "이어지는지 확인하는 축입니다."
                            ),
                        },
                        {
                            "axis": "partnership_momentum",
                            "analysis_prompt": "파트너십 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "파트너십 축은 이번 묶음에서 근거 부족으로 보며, "
                                "협력 구조가 후속 카드에서 반복되는지 봐야 합니다."
                            ),
                        },
                        {
                            "axis": "regulatory_risk",
                            "analysis_prompt": "규제 리스크 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "규제 리스크 축은 이번 묶음에서 근거 부족으로 보며, "
                                "정책 조건이 의사결정 제약으로 등장하는지 확인해야 합니다."
                            ),
                        },
                        {
                            "axis": "talent_movement",
                            "analysis_prompt": "인재 이동 축을 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "인재 이동 축은 이번 묶음에서 근거 부족으로 보며, "
                                "조직 변화가 실행 역량 신호로 연결되는지 확인해야 합니다."
                            ),
                        },
                    ],
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


class _RadarFillLLM:
    def __init__(self) -> None:
        self.prompts: list[Any] = []

    def invoke(self, prompt: Any, config: Any | None = None) -> _FakeResponse:
        del config
        self.prompts.append(prompt)
        return _FakeResponse(
            json.dumps(
                {
                    "radar_axis_interpretations": [
                        {
                            "axis": "시장 포지션",
                            "analysis_prompt": "시장 포지션 변화를 어떻게 읽을지 묻습니다.",
                            "interpretation": (
                                "여러 peer의 이슈가 함께 묶여 있어 개별 뉴스보다 "
                                "시장 포지션 변화 신호로 읽어야 합니다."
                            ),
                        }
                    ]
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


def test_analysis_unit_preserves_source_published_at_for_mixer_prompt():
    unit = analysis_unit_from_card(
        {
            "id": "CN-1",
            "title": "원문 날짜 보존",
            "company": "samsung_sds",
            "source_raw_article_ids": [10],
            "source_links": [
                {
                    "raw_article_id": 10,
                    "title": "원문 기사",
                    "source_name": "연합뉴스",
                    "published_at": "2026-06-10T09:30:00+09:00",
                    "url": "https://example.com/news",
                }
            ],
            "evidence_payload": {
                "analysis": {"analysis_summary": "분석"},
                "implication": {"skax_implication": {"why_important": "시사점"}},
            },
        }
    )

    card = unit.to_card_like()
    assert card["evidence_payload"]["source_links"][0]["published_at"].startswith("2026-06-10")
    prompt = mixer_module._format_analysis_units([card])
    assert "2026-06-10T09:30:00+09:00" in prompt


def test_mixer_validation_does_not_persist_ellipsis_when_clipping():
    long_hidden = (
        "핵심 신호는 AX와 클라우드 판단 기준이 기술 보유 선언에서 실제 고객군, "
        "운영 책임 조직, 보안 승인 권한, 수익화 근거를 함께 제시할 수 있는 실행 조건으로 "
        "이동한다는 점입니다."
    )
    long_action = (
        "SK AX는 금융과 공공 고객군을 분리해 우선 공략군을 정하고, 각 고객군별 "
        "오퍼링 책임 조직과 보안 리스크 승인 권한을 함께 지정해 상품화 기준을 명확히 한다."
    )
    content = json.dumps(
        {
            "mix_insight": long_hidden,
            "common_pattern": {
                "finding": "실행 조건을 성과 근거로 제시하는 움직임이 반복됩니다.",
                "rationale": "두 이슈 모두 고객군과 운영 조건을 판단 근거로 제시합니다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "comparison_point": {
                "finding": "한쪽은 고객군을, 다른 쪽은 운영 책임을 더 앞세웁니다.",
                "rationale": "같은 실행 조건 안에서도 강조점이 다릅니다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "hidden_conclusion": {
                "finding": long_hidden,
                "rationale": "여러 이슈를 함께 보면 실행 조건이 판단 기준으로 확인됩니다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "recommended_actions": [long_action],
            "action_details": [
                {
                    "action": long_action,
                    "why": "고객군과 운영 책임이 함께 제시될 때 실행 판단으로 이어집니다.",
                    "use_case": "사업 우선순위",
                    "evidence_card_ids": ["CN-1", "CN-2"],
                }
            ],
            "sources_used": ["CN-1", "CN-2"],
            "confidence": 0.8,
        },
        ensure_ascii=False,
    )

    result = mixer_module._parse_and_validate(
        content,
        cards=[{"id": "CN-1"}, {"id": "CN-2"}],
        card_ids=["CN-1", "CN-2"],
    )

    assert "…" not in result["final_one_liner"]
    assert "..." not in result["final_one_liner"]
    assert "…" not in result["sk_ax_implication"]
    assert "..." not in result["sk_ax_implication"]
    assert len(result["final_one_liner"]) > 100


def test_mixer_validation_normalizes_display_sentence_endings():
    content = json.dumps(
        {
            "mix_insight": (
                "AI 관련 발표는 더 이상 기능 소개만으로 충분하지 않고, "
                "인프라 투자 논의가 실제 업무 시스템 적용 범위와 외부 매출 구조로 "
                "이어질 수 있는지를 설명하는 근거가."
            ),
            "common_pattern": {
                "finding": "기술 적용 범위가 고객 설득 기준으로 이어진다",
                "rationale": "두 이슈 모두 적용 범위와 매출 구조를 판단 근거로 제시한다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "comparison_point": {
                "finding": "한쪽은 업무 적용 범위, 다른 쪽은 매출 구조를 앞세운다.",
                "rationale": "같은 AI 흐름 안에서도 설명해야 하는 성과 기준이 다르다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "hidden_conclusion": {
                "finding": (
                    "AI 발표의 판단 기준은 기능 소개에서 적용 범위와 수익화 근거로 이동한다."
                ),
                "rationale": "두 이슈를 함께 보면 기술 메시지가 사업 성과 설명으로 바뀐다.",
                "evidence_card_ids": ["CN-1", "CN-2"],
            },
            "recommended_actions": ["고객군별 AI 적용 범위와 수익화 기준을 먼저 정한다."],
            "action_details": [
                {
                    "action": "고객군별 AI 적용 범위와 수익화 기준을 먼저 정한다.",
                    "why": "기능 소개만으로는 고객 설득 근거가 부족하다.",
                    "use_case": "오퍼링/상품화",
                    "evidence_card_ids": ["CN-1", "CN-2"],
                }
            ],
            "sources_used": ["CN-1", "CN-2"],
            "confidence": 0.8,
        },
        ensure_ascii=False,
    )

    result = mixer_module._parse_and_validate(
        content,
        cards=[{"id": "CN-1"}, {"id": "CN-2"}],
        card_ids=["CN-1", "CN-2"],
    )

    assert result["mix_insight"] == ""
    assert result["common_pattern"]["finding"].endswith("이어집니다.")
    assert result["recommended_actions"][0].endswith("정합니다.")
    assert result["action_details"][0]["why"].endswith("니다.")
    assert "mix_insight:empty" in mixer_module._mixer_sentence_quality_issues(result)


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
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: fake_llm)
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
    assert result["radar_axes"][0]["analysis_prompt"]
    assert result["radar_axes"][0]["prompted_interpretation"]
    assert isinstance(result["follow_up_checks"], list)
    assert result["follow_up_checks"][0]["answer"]
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


def test_radar_interpretation_uses_llm_text_without_fallback():
    radar_axis = {
        "axis": "regulatory_risk",
        "score": 0.65,
        "explanation": "규제 리스크 판단 설명입니다.",
        "meaning": "보조 판단 신호입니다.",
        "support_count": 1,
        "total_count": 3,
        "matched_card_ids": ["CN-1"],
    }
    result = mixer_module._merge_radar_axis_interpretations(
        [radar_axis],
        [
            {
                "axis": "regulatory_risk",
                "interpretation": "규제 이벤트가 의사결정 리스크로 작동하는지 봅니다.",
            }
        ],
    )

    interpretation = result[0]["prompted_interpretation"]
    assert interpretation == "규제 이벤트가 의사결정 리스크로 작동하는지 봅니다."

    empty_result = mixer_module._merge_radar_axis_interpretations([radar_axis], [])
    assert empty_result[0]["prompted_interpretation"] == ""


def test_radar_interpretation_accepts_korean_axis_label():
    radar_axis = {
        "axis": "market_position",
        "score": 1.0,
        "explanation": "시장 포지션 판단 설명입니다.",
        "meaning": "강한 판단 신호입니다.",
        "support_count": 4,
        "total_count": 4,
        "matched_card_ids": ["CN-1", "CN-2", "CN-3", "CN-4"],
    }

    result = mixer_module._merge_radar_axis_interpretations(
        [radar_axis],
        [
            {
                "axis": "시장 포지션",
                "interpretation": "여러 peer의 이슈가 시장 포지션 변화로 읽힙니다.",
            }
        ],
    )

    assert result[0]["prompted_interpretation"] == "여러 peer의 이슈가 시장 포지션 변화로 읽힙니다."


def test_missing_radar_interpretation_is_filled_by_llm(monkeypatch):
    fake_llm = _RadarFillLLM()
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: fake_llm)
    radar_axis = {
        "axis": "market_position",
        "score": 1.0,
        "explanation": "여러 peer에 걸친 시장 신호인지 보는 값입니다.",
        "meaning": "강한 판단 신호입니다.",
        "support_count": 2,
        "total_count": 2,
        "matched_card_ids": ["CN-1", "CN-2"],
        "analysis_prompt": "시장 포지션 축을 어떻게 읽을지 묻습니다.",
        "prompted_interpretation": "",
    }
    cards = [
        {"id": "CN-1", "title": "A사 시장 확대", "peer_id": "A사"},
        {"id": "CN-2", "title": "B사 전략 제휴", "peer_id": "B사"},
    ]

    result = mixer_module._fill_missing_radar_axis_interpretations(
        [radar_axis],
        cards,
        "quick",
    )

    assert fake_llm.prompts
    assert "market_position" in str(fake_llm.prompts[0])
    assert "CN-1" in str(fake_llm.prompts[0])
    assert result[0]["prompted_interpretation"] == (
        "여러 peer의 이슈가 함께 묶여 있어 개별 뉴스보다 시장 포지션 변화 신호로 읽어야 합니다."
    )


def test_mixer_card_ids_are_interpreted_as_analysis_units(monkeypatch):
    issue_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    units = [_unit("CN-1", issue_ids[0]), _unit("CN-2", issue_ids[1])]

    monkeypatch.setattr(mixer_module, "load_analysis_units_by_card_ids", lambda ids: units)
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: _FakeLLM())
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
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: _FakeLLM())
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
    assert result["provenance"]["llm_model"] == mixer_module._QUICK_LLM_MODEL
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
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: _FakeLLM())

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
    assert result["provenance"]["llm_model"] == mixer_module._DEEP_LLM_MODEL
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
    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: _FakeLLM())
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

    monkeypatch.setattr(mixer_module, "_get_llm", lambda *args, **kwargs: _MissingCredentialLLM())

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
