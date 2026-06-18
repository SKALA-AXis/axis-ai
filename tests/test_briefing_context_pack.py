# 작성일: 2026-06-09
# 작성자: 최종민
# 변경이력:
#   2026-06-09 최종민 — ContextPackAssembler·주간 다이제스트 에이전트·백필 스크립트 추가에 따른 테스트 작성
from __future__ import annotations

from src.services.briefing_context_pack import merge_historical_context_into_basis


def test_merge_historical_context_enriches_comparison_point() -> None:
    basis = {
        "comparison_point": {
            "finding": "클라우드 파트너십 가속",
            "rationale": "이번 기간 카드 근거",
        },
    }
    historical = {
        "weekly_digest_by_peer": {
            "lg_cns": {
                "narrative": "클라우드 계약 확대",
                "delta_vs_prev": ["엔터프라이즈 AI 도입"],
            }
        },
        "executive_memory_recent": [
            {"report_date": "2026-06-08", "headline": "LG CNS AI 파트너십 검토"},
        ],
    }
    merged = merge_historical_context_into_basis(basis, historical)
    rationale = merged["comparison_point"]["rationale"]
    assert "지난주 대비" in rationale
    assert "임원 인사이트" in rationale
    assert merged["historical_context"] == historical
