from __future__ import annotations

from src.composers.card_news_composer import _plain_summary_lines


def test_card_summary_prefers_core_project_facts_over_market_reaction():
    summary = {
        "cluster_event_type": "contract",
        "fact_summary": [
            "A 컨소시엄이 국가 인프라 구축 사업자로 최종 선정됐다.",
            "사업 추진을 위한 실시협약과 SPC 설립을 위한 주주간계약을 체결했다.",
            "A사의 주가가 전 거래일 대비 3.22% 상승해 거래를 마쳤다.",
        ],
        "cluster_fact_intelligence": {
            "unique_facts": [
                {
                    "fact": (
                        "국가 인프라는 2028년까지 첨단 AI 반도체 1만 5000장 규모로 구축될 예정이다."
                    ),
                    "summary_role": "numeric_effect",
                }
            ]
        },
    }

    lines = _plain_summary_lines(summary)

    assert len(lines) == 3
    assert all("주가" not in line for line in lines)
    assert any("AI 반도체 1만 5000장" in line for line in lines)
