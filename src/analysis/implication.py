"""Strategic implication generation component."""

from __future__ import annotations

from typing import Any


class ImplicationGenerator:
    """Generate reusable implication output from summary and analysis results."""

    def generate(
        self,
        *,
        summary: dict[str, Any],
        analysis: dict[str, Any],
        classification: dict[str, Any] | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a first-pass implication payload.

        This intentionally stays lightweight for now. The supervisor owns orchestration,
        while richer peer/SKAX perspective agents can later replace this component.
        """
        classification = classification or {}
        peer_profile_context = peer_profile_context or {}
        skax_profile_context = skax_profile_context or {}

        risk_or_opportunity = str(analysis.get("risk_or_opportunity") or "neutral")
        impact_level = str(analysis.get("impact_level") or classification.get("importance") or "low")
        market_signal = str(analysis.get("market_signal") or "")
        analysis_summary = str(analysis.get("analysis_summary") or summary.get("main_event") or "")

        return {
            "implication_scope": "peer_and_skax",
            "peer_implication": _peer_implication(
                analysis_summary=analysis_summary,
                impact_level=impact_level,
                peer_profile_context=peer_profile_context,
            ),
            "skax_implication": _skax_implication(
                risk_or_opportunity=risk_or_opportunity,
                market_signal=market_signal,
                skax_profile_context=skax_profile_context,
            ),
            "watch_points": _watch_points(
                summary=summary,
                analysis=analysis,
                classification=classification,
            ),
            "confidence": _confidence(summary, analysis),
            "provenance": {
                "generator": "ImplicationGenerator",
                "uses_peer_profile_context": bool(peer_profile_context),
                "uses_skax_profile_context": bool(skax_profile_context),
            },
        }


def _peer_implication(
    *,
    analysis_summary: str,
    impact_level: str,
    peer_profile_context: dict[str, Any],
) -> str:
    peer_name = peer_profile_context.get("company_name") or peer_profile_context.get("peer_id")
    prefix = f"{peer_name} 관점에서는 " if peer_name else "피어사 관점에서는 "
    if analysis_summary:
        return f"{prefix}{analysis_summary} 신호를 {impact_level} 영향도로 추적할 필요가 있습니다."
    return f"{prefix}추가 근거가 쌓일 때까지 제한적으로 관찰하는 편이 적절합니다."


def _skax_implication(
    *,
    risk_or_opportunity: str,
    market_signal: str,
    skax_profile_context: dict[str, Any],
) -> str:
    business_lines = skax_profile_context.get("business_lines") or []
    target = ", ".join(str(item) for item in business_lines[:3]) if isinstance(business_lines, list) else ""
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
    points = []
    event_type = classification.get("event_type")
    if event_type:
        points.append(f"event_type={event_type} 관련 후속 보도/공시 변화")
    if analysis.get("impact_level"):
        points.append(f"impact_level={analysis['impact_level']} 판단을 바꿀 추가 근거")
    if summary.get("source_count"):
        points.append("동일 이슈의 출처 확산 여부")
    return points[:3]


def _confidence(summary: dict[str, Any], analysis: dict[str, Any]) -> float:
    values = []
    for payload in (summary, analysis):
        try:
            values.append(float(payload.get("confidence")))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return min(max(sum(values) / len(values), 0.0), 1.0)
