from __future__ import annotations

from src.composers.card_news_composer import CardNewsComposer, _plain_summary_lines


def test_card_summary_preserves_integration_fact_summary_verbatim():
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

    assert lines == summary["fact_summary"]


def test_card_summary_does_not_dedupe_or_backfill_integration_copy():
    summary = {
        "cluster_event_type": "contract",
        "fact_summary": [
            "정부가 GPU 확보 사업 참여 기업으로 네이버클라우드와 삼성SDS를 선정했다.",
            "과기정통부는 GPU 확보·구축 사업 참여 기업으로 삼성SDS를 최종 선정했다.",
            "삼성SDS가 정부 GPU 확보 사업 참여 기업으로 선정됐다.",
            "삼성SDS 주가가 전 거래일 대비 3.22% 상승했다.",
        ],
    }

    lines = _plain_summary_lines(summary)

    assert lines == summary["fact_summary"]


def test_card_summary_ignores_display_llm_selection_and_keeps_integrated_summary(monkeypatch):
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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    lines = _plain_summary_lines(summary, use_llm=True)

    assert lines == summary["fact_summary"]
    assert not any("9704장" in line for line in lines)


def test_card_news_from_analysis_package_adds_grounded_display_sections(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    package = {
        "bundle_id": "news:1",
        "input_bundle": {
            "items": [
                {
                    "id": 1,
                    "title": "테스트 기사",
                    "source_name": "news",
                    "url": "https://example.com/article",
                    "published_at": "2026-06-10T00:00:00+09:00",
                }
            ]
        },
        "integrated_issue": {
            "is_valid_summary": True,
            "cluster_id": 1,
            "main_company": "lg_cns",
            "headline": "LG CNS 금융 IT 현대화 신호",
            "source_article_ids": [1],
            "fact_summary": [
                "LG CNS가 금융 IT 현대화 계약 상대방으로 확인됐다.",
                "해당 사업은 코어뱅킹 웹단말 전환을 포함한다.",
                "계약 기간은 2026년부터 2027년까지다.",
            ],
        },
        "analysis": {
            "analysis_summary": "금융 IT 현대화 수요가 웹단말 전환까지 구체화됐다.",
            "strategic_meaning": ["표시되면 안 되는 미검증 시사점"],
            "market_signal": "금융권 핵심 시스템 전환 기준이 구체화되고 있다.",
            "confidence": 0.8,
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": "lg_cns",
                "company_name_ko": "LG CNS",
                "peer_meaning": (
                    "LG CNS는 금융권 핵심 시스템 전환 과제와 연결된 관찰 신호로 확인된다."
                ),
                "capability_change": "근거 없는 피어 역량 확장 문장",
            },
            "skax_implication": {
                "why_important": "SK AX 내부 대응 범위를 점검해야 한다.",
                "potential_impact": "유사 금융권 사업에서 비교 기준이 바뀔 수 있다.",
                "recommended_actions": [
                    (
                        "SK AX는 유사 금융권 전환 사업에서 대상 시스템 범위와 운영 책임, "
                        "전환 리스크를 내부 점검 기준으로 구조화해야 한다."
                    ),
                    "근거 없는 외부 제안 문장",
                ],
                "business_line_mapping": ["금융AX"],
            },
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
        "issue_understanding": {"peer_role_in_issue": "contract_counterparty"},
        "profile_linkage": {"linkage_level": "medium", "reason": "금융 IT 사업영역과 연결"},
        "skax_response_linkage": {
            "response_mode": "profile_based_action",
            "internal_checkpoints": ["금융권 전환 범위와 운영 책임을 비교 점검"],
            "monitoring_points": ["후속 금융권 전환 계약에서 피어 역할 변화 확인"],
        },
        "grounding_summary": {"used_fact_ids": ["fact:1"]},
        "sentence_grounding": {
            "entries": [
                {
                    "path": "peer_implication.peer_meaning",
                    "grounding_type": "fact+profile",
                    "needs_review": False,
                },
                {
                    "path": "peer_implication.capability_change",
                    "grounding_type": "ungrounded",
                    "needs_review": True,
                },
                {
                    "path": "skax_implication.recommended_actions[0]",
                    "grounding_type": "fact+profile",
                    "needs_review": False,
                },
                {
                    "path": "skax_implication.recommended_actions[1]",
                    "grounding_type": "ungrounded",
                    "needs_review": True,
                },
            ]
        },
        "validation": {"classification": {"event_type": "contract", "sector": "ax"}},
    }

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert list(sections) == ["summary", "insight", "action"]
    assert len(sections["summary"]["items"]) == 3
    insight_text = " ".join(sections["insight"]["items"])
    assert "LG CNS" in insight_text
    assert "금융권 핵심 시스템 전환 과제" in insight_text
    assert "비교 기준" in insight_text
    assert "후속 단계" in insight_text
    assert "핵심 변수" in insight_text
    assert "근거 없는 피어 역량 확장 문장" not in sections["insight"]["items"]
    action_text = " ".join(sections["action"]["items"])
    assert "SK AX" in action_text
    assert "금융 IT 현대화 신호와 유사한 사업" in action_text
    assert "어디까지 감당할 수 있는지" in action_text
    assert "직접 담당 가능한 범위" in action_text
    assert "외부 보완이 필요한 범위" in action_text
    assert "모니터링해야 합니다" in action_text
    assert "근거 없는 외부 제안 문장" not in sections["action"]["items"]
    assert [slide["layout_type"] for slide in card["slides"]] == ["summary", "insight", "action"]
    assert card["analysis_package"]["issue_understanding"] == package["issue_understanding"]
    assert card["analysis_package"]["profile_linkage"] == package["profile_linkage"]
    assert card["analysis_package"]["skax_response_linkage"] == package["skax_response_linkage"]
    assert card["analysis_package"]["grounding_summary"] == package["grounding_summary"]
