"""W5-1 — EvaluatorAgent rule-based 4 metric 단위 테스트.

Phase 1 metric: context_hit_ratio / evidence_claim_ratio / specificity_score /
actionability_score.
"""

from __future__ import annotations

from src.agents.evaluator_agent import EvaluatorAgent
from src.analysis.models import (
    AnalysisContext,
    CapabilityWindow,
    SectorPulseRow,
    TimelineEntry,
)


_DEFAULT_ACTIONS = [
    "에이전틱AI 협업 모델 PoC 제안서 작성 추진",
    "MSP 입찰 사전 자격 점검 착수",
    "고객사 임원 인터뷰 통해 수요 검증 추진",
]


def _basic_implication(
    *,
    why_important: str = "삼성SDS 의 OpenAI 파트너십 확대는 SK AX 의 에이전틱AI 사업에 직접 영향.",
    actions: list[str] | None = None,
    used_layers: list[str] | None = None,
    confidence: float = 0.72,
) -> dict:
    actions_value = _DEFAULT_ACTIONS if actions is None else actions
    return {
        "is_valid_implication": True,
        "skax_implication": {
            "why_important": why_important,
            "potential_impact": "MSP 입찰 전략에 변동 가능.",
            "opportunities": ["에이전틱AI 협업 모델 PoC 추진"],
            "threats": ["삼성SDS 가 MSP 시장 잠식"],
            "recommended_actions": actions_value,
            "business_line_mapping": ["에이전틱AI", "MSP"],
        },
        "peer_implication": {
            "company_id": "samsung_sds",
            "company_name_ko": "삼성SDS",
            "peer_meaning": "삼성SDS 는 외부 AI 파트너십을 통한 역량 확보 가속.",
            "capability_change": "2026 분기 클라우드/AI 인력 +12%.",
            "sourced_evidence_ids": ["fact_001"],
        },
        "follow_up_questions": ["...", "...", "..."],
        "watch_points": ["1분기 매출"],
        "confidence": confidence,
        "evidence_label": "sufficient",
        "provenance": {
            "generator": "ImplicationAgent",
            "prompt_version": "implication-v5.0",
            "model": "gpt-4o",
            "used_context_layers": used_layers
            or ["peer_event_timeline_recent", "capability_evolution"],
        },
    }


def _basic_context(layer_count: int) -> AnalysisContext:
    ctx = AnalysisContext()
    if layer_count >= 1:
        ctx.peer_event_timeline_recent = [
            TimelineEntry(
                company_id="samsung_sds",
                event_date="2026-05-10",
                card_id="CN-20260510-0001",
                event_type="partnership",
                sector="ax",
                headline="OpenAI 협업 확대",
                importance="high",
                importance_score=0.84,
            )
        ]
    if layer_count >= 2:
        ctx.capability_evolution = {
            "samsung_sds": CapabilityWindow(
                period="2025Q4-2026Q1",
                business_area="Cloud",
                narrative="MSP 매출 +15%.",
                delta_intensity=0.6,
                confidence=0.7,
            )
        }
    if layer_count >= 3:
        ctx.sector_pulse_recent = [
            SectorPulseRow(
                sector="ax",
                week_start="2026-05-18",
                event_count=12,
                peer_event_count=8,
                general_event_count=4,
                intensity_avg=0.62,
            )
        ]
    return ctx


def test_context_hit_ratio_uses_available_layer_count():
    impl = _basic_implication(used_layers=["peer_event_timeline_recent", "capability_evolution"])
    ctx = _basic_context(layer_count=2)  # available = 2
    metrics = EvaluatorAgent().evaluate(implication=impl, analysis_context=ctx)
    assert metrics.context_hit_ratio == 1.0


def test_context_hit_ratio_new_peer_no_unfair_penalty():
    """available_layer_count=1 인 신규 peer 가 6 고정 페널티 받지 않도록."""
    impl = _basic_implication(used_layers=["peer_event_timeline_recent"])
    ctx = _basic_context(layer_count=1)
    metrics = EvaluatorAgent().evaluate(implication=impl, analysis_context=ctx)
    assert metrics.context_hit_ratio == 1.0


def test_actionability_korean_verb_suffix_pattern_matches():
    """한국어 동사형 어미 사전 매칭. v3.2.1 P5-LOG-1."""
    impl = _basic_implication(
        actions=[
            "디지털 전환을 가속화한다",
            "신규 사업 검토",
            "MSP 입찰 자격 점검 착수",
        ]
    )
    metrics = EvaluatorAgent().evaluate(implication=impl)
    # 모든 action 의 verb-suffix match + 일부 구체성 (회사명/시점/숫자 X) → 부분 점수.
    assert 0.0 < metrics.actionability_score <= 1.0


def test_actionability_pure_noun_form_gets_no_match():
    impl = _basic_implication(actions=["디지털 전환 가속화", "AI 도입 확대", "고객 경험 혁신"])
    # 모든 action 이 "확대 / 도입" 같은 동사형 어휘 일부 포함 — verb_hits 발생.
    # 그래도 구체성 0 이면 곱 결과 0.
    metrics = EvaluatorAgent().evaluate(implication=impl)
    # 구체성 0 (회사명·시점·숫자 모두 없음) → actionability 0.
    assert metrics.actionability_score == 0.0


def test_actionability_zero_when_no_actions():
    impl = _basic_implication(actions=[])
    metrics = EvaluatorAgent().evaluate(implication=impl)
    assert metrics.actionability_score == 0.0


def test_evidence_claim_ratio_grounded_when_facts_present():
    impl = _basic_implication(why_important="삼성SDS 매출 3.54조원, MSP 사업 비중 12% 증가.")
    integrated_issue = {
        "fact_basis": [
            {"fact_id": "fact_001", "evidence_text": "삼성SDS 매출 3.54조원"},
            {"fact_id": "fact_002", "evidence_text": "MSP 사업 비중 12%"},
        ],
        "key_numbers": [{"value": "3.54조원"}, {"value": "12%"}],
    }
    metrics = EvaluatorAgent().evaluate(implication=impl, integrated_issue=integrated_issue)
    assert metrics.evidence_claim_ratio >= 0.6


def test_specificity_with_peer_and_sector_match():
    impl = _basic_implication()
    integrated_issue = {"sectors": ["ax"]}
    impl["skax_implication"]["why_important"] = "삼성SDS 의 ax 영역 진출."
    metrics = EvaluatorAgent().evaluate(implication=impl, integrated_issue=integrated_issue)
    assert metrics.specificity_score > 0.0


def test_regression_drift_none_when_no_baseline():
    impl = _basic_implication()
    metrics = EvaluatorAgent().evaluate(implication=impl)
    assert metrics.regression_drift is None


def test_regression_drift_computed_when_baseline_present():
    impl = _basic_implication(confidence=0.8)
    metrics = EvaluatorAgent().evaluate(implication=impl, rolling_confidence_avg=0.5)
    assert metrics.regression_drift is not None
    assert abs(metrics.regression_drift - 0.6) < 0.01
