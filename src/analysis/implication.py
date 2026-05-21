"""Strategic implication generation — legacy heuristic fallback only.

W1-1 이후 LLM 기반 시사점은 `src.agents.implication_agent.ImplicationAgent` 가
담당한다. 본 모듈의 ``ImplicationGenerator`` 는 LLM 호출이 실패했거나 분석 입력이
부족할 때 항상 dict 응답을 반환해 카드 생성이 막히지 않도록 한다.

backward-compat:
- 기존 호출부 (`ImplicationGenerator().generate(summary=..., analysis=...)`) 가
  여전히 dict 응답을 받을 수 있도록 시그니처 보존.
- 출력 dict 도 v1 schema 와 동일 (`peer_implication`, `skax_implication`, ...) 한
  단순 텍스트 형식.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.analysis.models import AnalysisInputBundle


class ImplicationGenerator:
    """Heuristic fallback implication generator (LLM 미사용).

    W1-1 의 LLM 기반 ImplicationAgent 가 fail 했을 때 동일 dict 계약을
    유지하기 위해 호출된다.
    """

    def generate(
        self,
        *,
        summary: dict[str, Any],
        analysis: dict[str, Any],
        classification: dict[str, Any] | None = None,
        input_bundle: AnalysisInputBundle | dict[str, Any] | None = None,
        profile_context: dict[str, Any] | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Lightweight fallback — analysis 의 단순 요약을 그대로 노출."""
        classification = classification or {}
        profile_context = profile_context or {}
        peer_profile_context = peer_profile_context or {}
        skax_profile_context = skax_profile_context or {}
        if profile_context:
            peer_profile_context = peer_profile_context or profile_context.get("peer_profiles", {})
            skax_profile_context = skax_profile_context or profile_context.get("skax_profile", {})
        input_payload = (
            input_bundle.to_dict()
            if isinstance(input_bundle, AnalysisInputBundle)
            else input_bundle or {}
        )

        risk_or_opportunity = str(analysis.get("risk_or_opportunity") or "neutral")
        impact_level = str(
            analysis.get("impact_level") or classification.get("importance") or "low"
        )
        market_signal = str(analysis.get("market_signal") or "")
        analysis_summary = str(analysis.get("analysis_summary") or summary.get("main_event") or "")

        peer_meaning = _peer_implication(
            analysis_summary=analysis_summary,
            impact_level=impact_level,
            peer_profile_context=peer_profile_context,
        )
        skax_meaning = _skax_implication(
            risk_or_opportunity=risk_or_opportunity,
            market_signal=market_signal,
            skax_profile_context=skax_profile_context,
        )
        watch_points = _watch_points(
            summary=summary,
            analysis=analysis,
            classification=classification,
        )
        confidence = _confidence(summary, analysis)
        return {
            "is_valid_implication": bool(peer_meaning or skax_meaning),
            "implication_scope": "peer_and_skax",
            "peer_implication": {
                "company_id": _primary_peer_id(peer_profile_context, input_payload),
                "company_name_ko": _primary_peer_name(peer_profile_context),
                "peer_meaning": peer_meaning,
                "capability_change": None,
                "precedent_link": None,
                "sourced_evidence_ids": [],
            },
            "skax_implication": {
                "why_important": skax_meaning,
                "potential_impact": "",
                "opportunities": [],
                "threats": [],
                "recommended_actions": [],
                "business_line_mapping": [],
            },
            "follow_up_questions": [],
            "watch_points": watch_points,
            "confidence": confidence,
            "evidence_label": _evidence_label(confidence),
            "provenance": {
                "generator": "ImplicationGenerator",
                "prompt_version": "implication-v0-heuristic",
                "model": "heuristic",
                "uses_input_bundle": bool(input_payload),
                "uses_peer_profile_context": bool(peer_profile_context),
                "uses_skax_profile_context": bool(skax_profile_context),
                "bundle_id": input_payload.get("bundle_id") if input_payload else None,
                "used_context_layers": [],
                "run_at": datetime.now(UTC).isoformat(),
            },
            # Backward-compat flatten (CardNewsAgent / frontend 호환)
            "opportunities": [],
            "threats": [],
            "recommended_actions": [],
        }


def _peer_implication(
    *,
    analysis_summary: str,
    impact_level: str,
    peer_profile_context: dict[str, Any],
) -> str:
    peer_profile = _primary_peer_profile(peer_profile_context)
    peer_name = peer_profile.get("company_name") or peer_profile.get("peer_id")
    prefix = f"{peer_name} 관점에서는 " if peer_name else "피어사 관점에서는 "
    if analysis_summary:
        return f"{prefix}{analysis_summary} 신호를 {impact_level} 영향도로 추적할 필요가 있습니다."
    return f"{prefix}추가 근거가 쌓일 때까지 제한적으로 관찰하는 편이 적절합니다."


def _primary_peer_profile(peer_profile_context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(peer_profile_context, dict):
        return {}
    if peer_profile_context.get("company_name") or peer_profile_context.get("peer_id"):
        return peer_profile_context
    for value in peer_profile_context.values():
        if isinstance(value, dict) and (value.get("company_name") or value.get("peer_id")):
            return value
    return {}


def _primary_peer_id(peer_profile_context: dict[str, Any], input_payload: dict[str, Any]) -> str:
    profile = _primary_peer_profile(peer_profile_context)
    company_id = profile.get("peer_id") or profile.get("company_id")
    if company_id:
        return str(company_id)
    companies = (input_payload or {}).get("companies") or []
    if isinstance(companies, list) and companies:
        return str(companies[0])
    return ""


def _primary_peer_name(peer_profile_context: dict[str, Any]) -> str:
    profile = _primary_peer_profile(peer_profile_context)
    return str(profile.get("company_name") or profile.get("company_name_ko") or "")


def _skax_implication(
    *,
    risk_or_opportunity: str,
    market_signal: str,
    skax_profile_context: dict[str, Any],
) -> str:
    business_lines = skax_profile_context.get("business_lines") or []
    target = (
        ", ".join(str(item) for item in business_lines[:3])
        if isinstance(business_lines, list)
        else ""
    )
    base = "SK AX 관점에서는"
    if target:
        base = f"{base} {target} 영역과의 연결 가능성을 기준으로"
    if market_signal:
        return f"{base} {market_signal} 신호를 {risk_or_opportunity} 관점에서 검토해야 합니다."
    return f"{base} 기회/위협 여부를 후속 데이터와 함께 재평가해야 합니다."


def _watch_points(
    *,
    summary: dict[str, Any],
    analysis: dict[str, Any],
    classification: dict[str, Any],
) -> list[str]:
    points: list[str] = []
    event_type = classification.get("event_type")
    if event_type:
        points.append(f"event_type={event_type} 관련 후속 보도/공시 변화")
    if analysis.get("impact_level"):
        points.append(f"impact_level={analysis['impact_level']} 판단을 바꿀 추가 근거")
    if summary.get("source_count"):
        points.append("동일 이슈의 출처 확산 여부")
    return points[:3]


def _confidence(summary: dict[str, Any], analysis: dict[str, Any]) -> float:
    values: list[float] = []
    for payload in (summary, analysis):
        confidence = payload.get("confidence")
        if confidence is None:
            continue
        try:
            values.append(float(confidence))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return min(max(sum(values) / len(values), 0.0), 1.0)


def _evidence_label(confidence: float) -> str:
    if confidence < 0.6:
        return "insufficient"
    if confidence < 0.8:
        return "moderate"
    return "sufficient"
