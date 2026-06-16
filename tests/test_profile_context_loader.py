"""profile_context_loader 단위 테스트 — 백필 point-in-time snapshot strip.

DB/LLM 미사용 (순수 헬퍼). load() 의 as_of 클램프(쿼리)는 live-DB smoke 로 별도 검증.
"""

from __future__ import annotations

from src.services.profile_context_loader import _strip_snapshot_time_sensitive


def test_strip_removes_time_sensitive_keeps_stable() -> None:
    """백필 시 snapshot 의 시점-민감/미래지향 필드만 제거, 안정 프로필은 유지."""
    snapshot = {
        "company_summary": "삼성SDS 클라우드·SI",
        "business_areas": [{"name": "클라우드"}],
        "core_capabilities": ["MSP", "AX"],
        "key_products_services": ["FabriX"],
        "execution_cases": [{"title": "..."}],
        # 시점-민감/미래지향 — 과거 카드엔 look-ahead → 제거 대상
        "financial_summary": {"2026Q1": {"매출": "17,424억원"}},
        "recent_changes": ["2026-05 ..."],
        "capability_evolution": [{"year": 2026}],
        "market_view": {"trend": "..."},
        "priority_initiatives": ["..."],
        "investment_roadmap": ["..."],
        "operational_highlights": ["..."],
    }
    out = _strip_snapshot_time_sensitive(snapshot)

    # 안정 프로필은 유지 (백필 카드도 일반 프로필 맥락은 받아야 함)
    for key in (
        "company_summary",
        "business_areas",
        "core_capabilities",
        "key_products_services",
        "execution_cases",
    ):
        assert key in out, f"안정 필드가 제거됨: {key}"

    # 시점-민감 필드는 제거 (룩어헤드 차단)
    for key in (
        "financial_summary",
        "recent_changes",
        "capability_evolution",
        "market_view",
        "priority_initiatives",
        "investment_roadmap",
        "operational_highlights",
    ):
        assert key not in out, f"시점-민감 필드가 남음: {key}"


def test_strip_is_noop_for_stable_only_snapshot() -> None:
    snapshot = {"company_summary": "x", "business_areas": []}
    assert _strip_snapshot_time_sensitive(snapshot) == snapshot
