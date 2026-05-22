"""Data hygiene 헬퍼 — W4-2 / W4-3 의 graceful fallback 로직.

cluster DB 의 실측 단점을 다른 모듈이 알 필요 없게 격리한다:
- period_quarter NULL → year 만으로도 그룹화 가능 여부 판단 (P3-DATA-1)
- card_news.source_raw_article_ids 빈 array → sources / evidence_payload 로 대체 (P3-CRIT-2)
- raw_article_business_signals.matched_companies 빈 array → peer_id 직접 사용 (P3-DATA-4)
"""

from __future__ import annotations

from typing import Any


def is_signal_well_grouped(signal: dict[str, Any]) -> bool:
    """period_quarter NULL 인 경우에도 (peer, business_area, year) 로 그룹화 가능?

    period_year 가 있으면 'unknown' fallback quarter 로도 의미 있는 그룹화 가능.
    """
    return signal is not None and signal.get("period_year") is not None


def is_card_provenance_traceable(card: dict[str, Any]) -> bool:
    """source_raw_article_ids 가 비어있어도 sources/evidence_payload 로 추적 가능한가?"""
    if not card:
        return False
    if card.get("source_raw_article_ids"):
        return True
    if card.get("sources"):
        return True
    evidence_payload = card.get("evidence_payload") or {}
    if isinstance(evidence_payload, dict) and evidence_payload.get("source_links"):
        return True
    evidence_chain = card.get("evidence_chain") or {}
    if isinstance(evidence_chain, dict) and evidence_chain.get("source_links"):
        return True
    return False


def signal_density_label(*, signal_count_4q: int, metric_count_4q: int) -> str:
    """peer 별 evidence 풍부도. P3-DATA-6 — ImplicationAgent confidence 조정 신호.

    Threshold 는 실측 (samsung_sds 8,235 signals / lg_cns 3,900 / 평균 5k) 기반.
    """
    total = signal_count_4q + metric_count_4q
    if total >= 600:
        return "rich"
    if total >= 200:
        return "moderate"
    return "sparse"


def confidence_floor_for_density(density_label: str) -> float:
    """density_label 별 confidence 상한. sparse 일 때 자동 강등."""
    if density_label == "sparse":
        return 0.6
    if density_label == "moderate":
        return 0.8
    return 1.0


__all__ = [
    "confidence_floor_for_density",
    "is_card_provenance_traceable",
    "is_signal_well_grouped",
    "signal_density_label",
]
