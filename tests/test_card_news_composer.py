from __future__ import annotations

import json

import src.composers.card_news_composer as composer
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


def test_card_summary_chooses_distinct_fact_axes_from_repeated_cluster_news(monkeypatch):
    summary = {
        "cluster_event_type": "contract",
        "headline": "삼성SDS, 정부 GPU 확보 사업 참여",
        "fact_summary": [
            (
                "과학기술정보통신부는 8일 첨단 GPU 확보·구축·운용지원 사업 "
                "참여 기업으로 이 3사를 최종 선정했다."
            ),
            (
                "과학기술정보통신부는 첨단 GPU 확보·구축·운용지원 사업 참여 "
                "기업으로 네이버클라우드, 삼성SDS, 엘리스그룹 등 3개 클라우드 "
                "서비스 기업을 선정했다."
            ),
            (
                "정부가 2조원 규모의 GPU 확보·구축·운용지원 사업에 "
                "네이버클라우드와 삼성SDS, 엘리스그룹을 참여 기업으로 선정했다."
            ),
        ],
        "consolidated_facts": [
            {
                "fact": (
                    "과학기술정보통신부는 네이버클라우드, 삼성SDS, 엘리스그룹을 통해 "
                    "베라루빈 2016장, B300 7688장 등 첨단 GPU 총 9704장을 확보한다고 밝혔다."
                )
            },
            {
                "fact": (
                    "확보 GPU는 독자 AI 파운데이션 모델 개발, 국가 AI 프로젝트, "
                    "산학연 AI 모델·서비스 개발 및 고도화 지원에 활용될 계획이다."
                )
            },
            {
                "fact": (
                    "정부는 GPU 구매 발주를 추진해 입고·구축이 완료되는 순서대로 "
                    "연내 B300 서비스를 개시하고, 베라루빈은 내년 상반기 안에 "
                    "순차적으로 서비스를 시작할 예정이다."
                )
            },
        ],
    }

    selected = [
        summary["fact_summary"][1],
        summary["consolidated_facts"][0]["fact"],
        summary["consolidated_facts"][1]["fact"],
    ]

    class FakeResponse:
        content = json.dumps({"summary_lines": selected}, ensure_ascii=False)

    class FakeLlm:
        def invoke(self, *_args, **_kwargs):
            return FakeResponse()

    old_llm = composer._llm
    composer._llm = FakeLlm()  # type: ignore[assignment]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    try:
        lines = _plain_summary_lines(summary, use_llm=True)
    finally:
        composer._llm = old_llm

    assert len(lines) == 3
    assert sum("선정" in line for line in lines) == 1
    assert any("9704장" in line for line in lines)
    assert any("활용" in line or "서비스" in line for line in lines)
