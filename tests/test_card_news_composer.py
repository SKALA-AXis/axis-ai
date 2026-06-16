from __future__ import annotations

from src.composers.card_news_composer import (
    CardNewsComposer,
    _card_from_summary,
    _looks_like_non_summary_line,
    _plain_summary_lines,
    _public_copy_cleanup,
)


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


def test_card_summary_cleans_truncated_display_fragments_without_rewriting_source():
    truncated = '젠슨 황 "韓, AI인프라 확장 필수"…AI팩토리 구축 협력[젠슨황 방한 4대...'
    summary = {
        "is_valid_summary": True,
        "cluster_event_type": "general_update",
        "main_company": "lg_cns",
        "headline": truncated,
        "fact_summary": [
            truncated,
            "황 CEO는 국내 주요 그룹 총수들과 회동하며 AI 인프라 구축 필요성을 강조했다.",
            "AI 팩토리와 데이터센터 구축 논의가 함께 언급됐다.",
        ],
    }
    articles = [
        {
            "id": 1,
            "title": truncated,
            "source_name": "news",
            "url": "https://example.com/news",
            "published_at": "2026-06-14T09:00:00+09:00",
        }
    ]

    card = _card_from_summary(
        summary=summary,
        articles=articles,
        company="lg_cns",
        cluster_id=48797,
        representative_id=1,
        classification={},
    )

    assert summary["headline"] == truncated
    assert card is not None
    visible_text = " ".join([card["title"], *card["summary_lines"]])
    assert "..." not in visible_text
    assert "…" not in visible_text
    assert "[젠슨황 방한" not in visible_text
    assert "..." not in str((card.get("db_record") or {}).get("title"))
    assert "…" not in str((card.get("db_record") or {}).get("title"))


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


def test_card_summary_filters_question_headline_from_display_copy():
    summary = {
        "cluster_event_type": "general_update",
        "fact_summary": [
            "LG CNS는 앤트로픽과 클로드 엔터프라이즈 통합 계약을 체결했다.",
            "이번 계약은 특정 계열사가 아니라 그룹 전반에 적용 가능한 형태로 알려졌다.",
            "재계, 내부 AI서 외부 AI 도입으로 전환 이유는?",
        ],
    }

    lines = _plain_summary_lines(summary)

    assert lines == [
        "LG CNS는 앤트로픽과 클로드 엔터프라이즈 통합 계약을 체결했다.",
        "이번 계약은 특정 계열사가 아니라 그룹 전반에 적용 가능한 형태로 알려졌다.",
    ]
    assert _looks_like_non_summary_line("재계, 내부 AI서 외부 AI 도입으로 전환 이유는?")


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
    import src.composers.card_news_composer as composer_module

    monkeypatch.setattr(composer_module, "_now_iso", lambda: "2026-06-15T09:30:00+09:00")
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

    assert card["id"] == "CN-20260610-0001"
    assert card["published_date"] == "2026-06-10"
    assert list(sections) == ["summary", "insight", "action"]
    assert len(sections["summary"]["items"]) == 3
    assert sections["insight"]["items"] == []
    assert sections["action"]["items"] == []
    assert card["needs_review"] is True
    assert [slide["layout_type"] for slide in card["slides"]] == ["summary"]
    assert card["analysis_package"]["issue_understanding"] == package["issue_understanding"]
    assert card["analysis_package"]["profile_linkage"] == package["profile_linkage"]
    assert card["analysis_package"]["skax_response_linkage"] == package["skax_response_linkage"]
    assert card["analysis_package"]["grounding_summary"] == package["grounding_summary"]


def test_card_news_prefers_frontend_ready_copy_without_rewriting(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    package = {
        "bundle_id": "news:7",
        "input_bundle": {
            "items": [
                {
                    "id": 7,
                    "title": "테스트 기사",
                    "source_name": "news",
                    "url": "https://example.com/article",
                    "published_at": "2026-06-12T09:00:00+09:00",
                }
            ]
        },
        "integrated_issue": {
            "is_valid_summary": True,
            "cluster_id": 7,
            "main_company": "lg_cns",
            "headline": "LG CNS, 물류 자동화 협약",
            "source_article_ids": [7],
            "fact_summary": [
                "LG CNS는 물류센터 로봇 자동화 협약을 체결했다.",
                "로봇 학습 플랫폼과 통합 관제 플랫폼을 활용한다.",
                "물류 현장 데이터를 기반으로 운영 체계를 구축할 예정이다.",
            ],
        },
        "analysis": {
            "analysis_summary": "이 문장이 frontend_ready를 이기면 안 된다.",
            "strategic_meaning": ["후보 점수화로 선택되면 안 되는 문장"],
            "market_signal": "템플릿형 시장 신호",
            "confidence": 0.8,
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": "lg_cns",
                "company_name_ko": "LG CNS",
                "peer_meaning": "후보 선택 경로로 쓰이면 안 되는 피어 문장",
                "capability_change": "후보 선택 경로로 쓰이면 안 되는 역량 문장",
            },
            "skax_implication": {
                "why_important": "후보 선택 경로로 쓰이면 안 되는 SK AX 문장",
                "potential_impact": "후보 선택 경로로 쓰이면 안 되는 영향 문장",
                "recommended_actions": ["후보 선택 경로로 쓰이면 안 되는 대응 문장"],
                "business_line_mapping": ["물류"],
            },
            "frontend_ready": {
                "source": "llm_direct",
                "key_implication": {
                    "source": "llm_direct",
                    "frame": "피어 물류 자동화 적용",
                    "event_anchor_terms": ["로봇 학습 플랫폼", "통합 관제 플랫폼"],
                    "profile_anchor_terms": ["물류센터", "관제"],
                    "sentence": (
                        "LG CNS는 물류센터를 로봇 학습·관제 플랫폼의 적용처로 "
                        "넓히는 흐름을 보여준다."
                    ),
                    "evidence_sentence": (
                        "기사에는 로봇 학습 플랫폼과 통합 관제 플랫폼을 활용해 "
                        "물류 현장 데이터를 기반으로 운영 체계를 구축한다는 사실이 제시됐다."
                    ),
                },
                "suggested_action": {
                    "source": "llm_direct",
                    "frame": "SK AX 물류 자동화 점검",
                    "event_anchor_terms": ["로봇 학습 플랫폼", "통합 관제 플랫폼"],
                    "skax_anchor_terms": ["제조·물류", "관제"],
                    "sentence": (
                        "SK AX는 제조·물류 고객군에서 운영 데이터와 관제 역량을 "
                        "묶어 볼 필요가 있다."
                    ),
                    "evidence_sentence": (
                        "이번 사례가 로봇 장비보다 학습·관제·현장 데이터 운영을 "
                        "함께 다루기 때문이다."
                    ),
                },
            },
            "confidence": 0.75,
            "evidence_label": "moderate",
        },
        "validation": {"classification": {"event_type": "partnership", "sector": "ax"}},
    }

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert sections["summary"]["items"] == package["integrated_issue"]["fact_summary"]
    assert sections["insight"]["items"] == [
        (
            "핵심 시사점: LG CNS는 물류센터를 로봇 학습·관제 플랫폼의 적용처로 "
            "넓히는 흐름을 보여준다.\n"
            "근거/설명: 기사에는 로봇 학습 플랫폼과 통합 관제 플랫폼을 활용해 "
            "물류 현장 데이터를 기반으로 운영 체계를 구축한다는 사실이 제시됐다."
        )
    ]
    assert sections["action"]["items"] == [
        (
            "핵심 대응: SK AX는 제조·물류 고객군에서 운영 데이터와 관제 역량을 "
            "묶어 볼 필요가 있다.\n"
            "근거/설명: 이번 사례가 로봇 장비보다 학습·관제·현장 데이터 운영을 "
            "함께 다루기 때문이다."
        )
    ]
    assert "후보 선택 경로" not in " ".join(
        sections["insight"]["items"] + sections["action"]["items"]
    )


def _package_for_frontend_ready_policy(frontend_ready: dict | None = None) -> dict:
    return {
        "input_bundle": {"items": [{"id": 1, "title": "정책 테스트"}]},
        "integrated_issue": {
            "is_valid_summary": True,
            "cluster_id": 1,
            "main_company": "lg_cns",
            "headline": "LG CNS, 물류 자동화 협약",
            "fact_summary": ["LG CNS는 물류 자동화 협약을 체결했다."],
        },
        "analysis": {
            "analysis_summary": "분석 fallback 문장",
            "strategic_meaning": ["분석 후보 문장"],
            "market_signal": "시장 후보 문장",
        },
        "implication": {
            "is_valid_implication": True,
            "peer_implication": {
                "company_id": "lg_cns",
                "company_name_ko": "LG CNS",
                "peer_meaning": "기존 implication 피어 문장",
                "capability_change": "기존 implication 역량 문장",
            },
            "skax_implication": {
                "why_important": "기존 implication SK AX 문장",
                "potential_impact": "기존 implication 영향 문장",
                "recommended_actions": ["기존 implication 대응 문장"],
                "business_line_mapping": ["물류"],
            },
            "confidence": 0.7,
            "evidence_label": "moderate",
            **({"frontend_ready": frontend_ready} if frontend_ready is not None else {}),
        },
        "validation": {"classification": {"event_type": "partnership", "sector": "ax"}},
    }


def test_card_news_does_not_show_derived_frontend_ready_source() -> None:
    package = _package_for_frontend_ready_policy(
        {
            "source": "derived_from_implication",
            "key_implication": {
                "source": "derived_from_implication",
                "frame": "피어 분석",
                "event_anchor_terms": ["물류 자동화"],
                "profile_anchor_terms": ["물류"],
                "sentence": "보이면 안 되는 시사점",
                "evidence_sentence": "보이면 안 되는 근거",
            },
            "suggested_action": {
                "source": "derived_from_implication",
                "frame": "대응 분석",
                "event_anchor_terms": ["물류 자동화"],
                "skax_anchor_terms": ["물류"],
                "sentence": "보이면 안 되는 대응",
                "evidence_sentence": "보이면 안 되는 대응 근거",
            },
        }
    )

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert sections["insight"]["items"] == []
    assert sections["action"]["items"] == []
    assert card["needs_review"] is True


def test_card_news_does_not_show_report_copy_repair_frontend_ready_source() -> None:
    package = _package_for_frontend_ready_policy(
        {
            "source": "report_copy_repair_direct",
            "key_implication": {
                "source": "report_copy_repair_direct",
                "frame": "피어 분석",
                "event_anchor_terms": ["물류 자동화"],
                "profile_anchor_terms": ["물류"],
                "sentence": "보이면 안 되는 report copy 시사점",
                "evidence_sentence": "보이면 안 되는 report copy 근거",
            },
            "suggested_action": {
                "source": "report_copy_repair_direct",
                "frame": "대응 분석",
                "event_anchor_terms": ["물류 자동화"],
                "skax_anchor_terms": ["물류"],
                "sentence": "보이면 안 되는 report copy 대응",
                "evidence_sentence": "보이면 안 되는 report copy 대응 근거",
            },
        }
    )

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert sections["insight"]["items"] == []
    assert sections["action"]["items"] == []
    assert card["needs_review"] is True


def test_card_news_shows_industry_frontend_ready_with_industry_labels() -> None:
    package = _package_for_frontend_ready_policy()
    package["implication"]["industry_frontend_ready"] = {
        "source": "industry_signal_direct",
        "signal_scope": "market_infra_signal",
        "display_policy": "industry_only",
        "items": [
            {
                "strategic_axis": "market_infra_capacity",
                "event_anchor_terms": ["AI 인프라", "데이터센터", "GPU"],
                "key_implication": {
                    "source": "industry_signal_direct",
                    "frame": "industry_signal",
                    "event_anchor_terms": ["AI 인프라", "데이터센터"],
                    "sentence": (
                        "AI 인프라와 데이터센터는 산업 경쟁 기준이 구체화되는 신호입니다."
                    ),
                    "evidence_sentence": (
                        "글로벌 벤더 중심으로 AI 인프라와 데이터센터 관련 내용이 제시되어 "
                        "산업 인프라 조건을 읽는 근거가 됩니다."
                    ),
                },
                "suggested_action": {
                    "source": "industry_signal_direct",
                    "frame": "industry_response_check",
                    "event_anchor_terms": ["AI 인프라", "데이터센터"],
                    "sentence": (
                        "SK AX는 AI 인프라와 데이터센터 시장을 볼 때 투자·운영 조건을 "
                        "분리해 모니터링해야 합니다."
                    ),
                    "evidence_sentence": (
                        "피어 직접 실행이 아니라 산업 인프라 신호이므로 투자 규모와 "
                        "참여 주체를 나눠 보는 기준이 필요합니다."
                    ),
                },
            }
        ],
    }

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert sections["insight"]["title"] == "시사점"
    assert sections["action"]["title"] == "대응방안"
    assert sections["insight"]["display_policy"] == "industry_only"
    assert sections["action"]["signal_scope"] == "market_infra_signal"
    assert sections["insight"]["items"]
    assert sections["action"]["items"]
    assert card["needs_review"] is False
    assert card["frontend_implication"]["display_policy"] == "industry_only"
    assert "display_label" not in card["frontend_implication"]
    assert "기존 implication" not in " ".join(
        sections["insight"]["items"] + sections["action"]["items"]
    )


def test_card_news_shows_frontend_repair_direct_frontend_ready_source() -> None:
    package = _package_for_frontend_ready_policy(
        {
            "source": "frontend_repair_direct",
            "key_implication": {
                "source": "frontend_repair_direct",
                "frame": "피어 분석",
                "claim_type": "event_based_signal",
                "claim_strength": "cautious",
                "evidence_mode": "event_based",
                "event_anchor_terms": ["물류 자동화"],
                "profile_anchor_terms": [],
                "sentence": "물류 자동화 협약은 현장 운영 방식 변화 신호입니다.",
                "evidence_sentence": "기사에는 물류 자동화 협약 체결 사실이 제시됐습니다.",
            },
            "suggested_action": {
                "source": "frontend_repair_direct",
                "frame": "대응 분석",
                "claim_type": "internal_strategy_check",
                "claim_strength": "cautious",
                "evidence_mode": "generic_monitoring",
                "event_anchor_terms": ["물류 자동화"],
                "skax_anchor_terms": [],
                "sentence": "SK AX는 물류 자동화 관련 내부 대응 범위를 점검해야 합니다.",
                "evidence_sentence": "이 점검은 유사 물류 자동화 흐름과 연결됩니다.",
            },
        }
    )

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}

    assert sections["insight"]["items"] == [
        "핵심 시사점: 물류 자동화 협약은 현장 운영 방식 변화 신호입니다.\n"
        "근거/설명: 기사에는 물류 자동화 협약 체결 사실이 제시됐습니다."
    ]
    assert sections["action"]["items"] == [
        "핵심 대응: SK AX는 물류 자동화 관련 내부 대응 범위를 점검해야 한다.\n"
        "근거/설명: 이 점검은 유사 물류 자동화 흐름과 연결된다."
    ]
    assert card["needs_review"] is False


def test_card_news_display_sync_does_not_overwrite_original_recommended_actions() -> None:
    package = _package_for_frontend_ready_policy(
        {
            "source": "frontend_repair_direct",
            "key_implication": {
                "source": "frontend_repair_direct",
                "frame": "피어 분석",
                "claim_type": "event_based_signal",
                "claim_strength": "cautious",
                "evidence_mode": "event_based",
                "event_anchor_terms": ["물류 자동화"],
                "profile_anchor_terms": [],
                "sentence": "물류 자동화 협약은 현장 운영 방식 변화 신호입니다.",
                "evidence_sentence": "기사에는 물류 자동화 협약 체결 사실이 제시됐습니다.",
            },
            "suggested_action": {
                "source": "frontend_repair_direct",
                "frame": "대응 분석",
                "claim_type": "internal_strategy_check",
                "claim_strength": "cautious",
                "evidence_mode": "generic_monitoring",
                "event_anchor_terms": ["물류 자동화"],
                "skax_anchor_terms": [],
                "sentence": "SK AX는 물류 자동화 관련 내부 대응 범위를 점검해야 합니다.",
                "evidence_sentence": "이 점검은 유사 물류 자동화 흐름과 연결됩니다.",
            },
        }
    )

    card = CardNewsComposer().generate_from_analysis_package(package)

    assert card["implication"]["skax_implication"]["recommended_actions"] == [
        "기존 implication 대응 문장"
    ]
    assert card["implication"]["frontend"]["suggested_actions"] == [
        "핵심 대응: SK AX는 물류 자동화 관련 내부 대응 범위를 점검해야 한다.\n"
        "근거/설명: 이 점검은 유사 물류 자동화 흐름과 연결된다."
    ]


def test_card_news_without_frontend_ready_does_not_run_editorial_fallback() -> None:
    package = _package_for_frontend_ready_policy()

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}
    visible = " ".join(sections["insight"]["items"] + sections["action"]["items"])

    assert sections["insight"]["items"] == []
    assert sections["action"]["items"] == []
    assert "분석 후보" not in visible
    assert "기존 implication" not in visible
    assert card["frontend_implication"]["key_implications"] == []
    assert card["frontend_implication"]["suggested_actions"] == []
    assert card["needs_review"] is True


def test_public_copy_cleanup_does_not_rewrite_semantic_terms() -> None:
    text = "피어 프로필과 프로필 접점은 피어사 분석에 남아야 한다."

    assert _public_copy_cleanup(text) == text


def test_card_news_action_section_filters_actions_without_issue_grounding() -> None:
    package = {
        "integrated_issue": {
            "cluster_id": 2,
            "main_company": "samsung_sds",
            "cluster_event_type": "partnership",
            "headline": "삼성SDS, 클라우드 보안 협력 강화",
            "main_event": "삼성SDS가 클라우드 보안 협력을 강화했다.",
            "fact_summary": [
                "삼성SDS가 클라우드 보안 협력을 강화했다.",
                "파트너 기술을 활용해 기업 고객의 보안 모니터링을 확대한다.",
                "클라우드 환경의 취약점 점검과 후속 조치 체계를 보강한다.",
            ],
            "fact_basis": [
                {
                    "summary_line_index": 1,
                    "fact_ids": ["fact:1"],
                    "source_article_ids": [10],
                }
            ],
            "representative_sources": [{"published_at": "2026-06-10T00:00:00+09:00"}],
        },
        "implication": {
            "frontend": {
                "suggested_actions": [
                    "제안서에는 모델 기능 비교와 별도로 전사 적용 범위표를 둡니다.",
                    "PoC 검증표는 답변 품질 중심이 아니라 업무 단위별 검증으로 바꿉니다.",
                    (
                        "SK AX는 클라우드 보안 협력 확대와 유사한 사업에서 "
                        "운영 책임과 보안 모니터링 범위를 비교 점검해야 합니다."
                    ),
                ],
            },
            "skax_implication": {
                "recommended_actions": [
                    "제안서에는 모델 기능 비교와 별도로 전사 적용 범위표를 둡니다.",
                    "PoC 검증표는 답변 품질 중심이 아니라 업무 단위별 검증으로 바꿉니다.",
                    (
                        "SK AX는 클라우드 보안 협력 확대와 유사한 사업에서 "
                        "운영 책임과 보안 모니터링 범위를 비교 점검해야 합니다."
                    ),
                ],
            },
            "peer_implication": {
                "peer_meaning": (
                    "클라우드 보안 협력 확대는 기업 고객 보안 운영 범위가 넓어지는 신호입니다."
                )
            },
        },
        "sentence_grounding": {
            "entries": [
                {
                    "path": "skax_implication.recommended_actions[0]",
                    "grounding_type": "fact",
                    "needs_review": False,
                },
                {
                    "path": "skax_implication.recommended_actions[1]",
                    "grounding_type": "fact",
                    "needs_review": False,
                },
                {
                    "path": "skax_implication.recommended_actions[2]",
                    "grounding_type": "fact",
                    "needs_review": False,
                },
            ]
        },
        "validation": {"classification": {"event_type": "partnership", "sector": "ax"}},
    }

    card = CardNewsComposer().generate_from_analysis_package(package)
    sections = {section["type"]: section for section in card["display_sections"]}
    action_text = " ".join(sections["action"]["items"])

    assert "제안서" not in action_text
    assert "PoC" not in action_text
    assert "검증표" not in action_text
    assert action_text == ""
    assert card["needs_review"] is True


def test_card_news_from_cluster_summary_uses_source_published_date(monkeypatch):
    import src.composers.card_news_composer as composer_module

    monkeypatch.setattr(composer_module, "_now_iso", lambda: "2026-06-15T09:30:00+09:00")
    monkeypatch.setattr(
        composer_module,
        "get_articles_by_ids",
        lambda article_ids: [
            {
                "id": 25771,
                "title": "포스코DX, AI와 사람의 협업 시대를 선도",
                "content": "포스코DX가 AI 협업 기술을 소개했다.",
                "source_name": "naver_news",
                "url": "https://example.com/25771",
                "published_at": "2026-03-04T09:00:00+09:00",
            }
        ],
    )

    card = CardNewsComposer().generate_from_cluster(
        cluster_id=25771,
        representative_id=25771,
        company="posco_dx",
        classification={"event_type": "tech_release", "sector": "ax"},
        summary={
            "is_valid_summary": True,
            "main_company": "posco_dx",
            "headline": "포스코DX, AI와 사람의 협업 시대를 선도",
            "fact_summary": ["포스코DX가 AI 협업 기술을 소개했다."],
        },
    )

    assert card["id"] == "CN-20260304-25771"
    assert card["published_date"] == "2026-03-04"
    assert card["created_at"] == "2026-03-04T09:00:00+09:00"


def test_card_news_compacts_sentence_like_cover_title(monkeypatch):
    import src.composers.card_news_composer as composer_module

    monkeypatch.setattr(composer_module, "_now_iso", lambda: "2026-06-15T12:00:00+09:00")
    card = CardNewsComposer().generate(
        {
            "cluster_id": 49206,
            "main_company": "samsung_sds",
            "headline": (
                "스마트테크 코리아 2026은 16개국의 620개사가 참가하여 AI와 로봇, "
                "스마트제조, 디지털 유통·물류 등 미래 산업에 쓰이는 기술을 소개했다"
            ),
            "fact_summary": [
                "스마트테크 코리아 2026은 16개국 620개사가 참가했다.",
                "AI와 로봇, 스마트제조, 디지털 유통·물류 기술이 소개됐다.",
                "행사는 서울 코엑스에서 열렸다.",
            ],
        },
        classification={"company": "samsung_sds", "title": ""},
        articles=[
            {
                "id": 49206,
                "title": "[현장] 막 내린 ‘스마트테크 코리아 2026’, AI로 달라진 미래 선보여",
                "url": "https://example.com/article",
                "published_at": "2026-06-15T02:36:00+00:00",
            }
        ],
    )

    assert card["title"] == "막 내린 ‘스마트테크 코리아 2026’, AI로 달라진 미래 선보여"
    assert len(card["title"]) <= 52


def test_card_news_from_cluster_summary_uses_earliest_source_date(monkeypatch):
    import src.composers.card_news_composer as composer_module

    monkeypatch.setattr(composer_module, "_now_iso", lambda: "2026-06-16T09:30:00+09:00")
    monkeypatch.setattr(
        composer_module,
        "get_articles_by_ids",
        lambda article_ids: [
            {
                "id": 2,
                "title": "후속 기사",
                "content": "후속 기사",
                "source_name": "naver_news",
                "url": "https://example.com/2",
                "published_at": "2026-06-15T09:00:00+09:00",
            },
            {
                "id": 1,
                "title": "최초 기사",
                "content": "최초 기사",
                "source_name": "naver_news",
                "url": "https://example.com/1",
                "published_at": "2026-06-12T09:00:00+09:00",
            },
        ],
    )

    card = CardNewsComposer().generate_from_cluster(
        cluster_id=48480,
        representative_id=2,
        cluster_article_ids=[2, 1],
        company="lg_cns",
        classification={"event_type": "tech_release", "sector": "ax"},
        summary={
            "is_valid_summary": True,
            "main_company": "lg_cns",
            "headline": "LG CNS, 서비스 출시",
            "fact_summary": ["LG CNS가 서비스를 출시했다."],
        },
    )

    assert card["id"] == "CN-20260612-48480"
    assert card["published_date"] == "2026-06-12"
    assert card["created_at"] == "2026-06-12T09:00:00+09:00"


def test_financial_only_card_title_adds_business_context():
    card = CardNewsComposer().generate(
        summary={
            "cluster_id": 36061,
            "main_company": "samsung_sds",
            "cluster_event_type": "earnings",
            "headline": "삼성SDS, 지난해 매출과 영업이익 증가",
            "fact_summary": [
                "삼성SDS는 지난해 매출 13조9299억원과 영업이익 9571억원을 기록했다.",
                "클라우드&AI 부문 매출은 3조5872억원으로 전년 대비 7% 증가했다.",
                "AI 솔루션 영역에서 생성형 AI 서비스 확산을 추진한다.",
            ],
            "source_article_ids": [36061],
            "is_valid_summary": True,
        },
        classification={"event_type": "earnings", "sector": "infra"},
        articles=[
            {
                "id": 36061,
                "title": "삼성SDS, 지난해 영업익 9571억",
                "published_at": "2026-01-22T00:00:00+09:00",
                "source_name": "news",
                "url": "https://example.com/36061",
            }
        ],
    )

    assert card["title"] == "삼성SDS, 클라우드·AI 인프라·AI·AX 사업 중심 실적 변화"
    assert card["display"]["background_asset_url"] == "/png.png"


def test_card_cover_uses_best_cluster_image_not_first_placeholder():
    card = CardNewsComposer().generate(
        summary={
            "cluster_id": 36301,
            "main_company": "samsung_sds",
            "headline": "삼성SDS, 구미에 AI 데이터센터 건립",
            "fact_summary": [
                "삼성SDS가 구미에 AI 데이터센터를 건립하기로 했다.",
                "구미 AI 데이터센터 건립을 위한 양해각서를 체결했다.",
                "AI 인프라 경쟁력을 강화할 계획이다.",
            ],
            "source_article_ids": [1, 2],
            "is_valid_summary": True,
        },
        classification={"event_type": "investment", "sector": "infra"},
        articles=[
            {
                "id": 1,
                "title": "삼성SDS, 신규 구미 AI 데이터센터 건립에 4273억원 투자",
                "published_at": "2026-01-02T00:00:00+09:00",
                "source_name": "news",
                "url": "https://example.com/a",
                "metadata": {"image_urls": ["https://static.sedaily.com/img/1X1.png"]},
            },
            {
                "id": 2,
                "title": "삼성SDS, 경상북도와 AI 데이터센터 건립 MOU 체결",
                "content": "삼성SDS와 경상북도, 구미시 관계자가 협약식 현장에서 기념 촬영했다.",
                "published_at": "2026-01-08T00:00:00+09:00",
                "source_name": "news",
                "url": "https://example.com/b",
                "metadata": {
                    "image_urls": ["https://cdn.example.com/news/photo/20260108/datacenter-mou.jpg"]
                },
            },
        ],
    )

    assert card["display"]["background_asset_url"].endswith("datacenter-mou.jpg")


def test_summary_fallback_uses_article_title_and_image_for_sentence_headline():
    card = _card_from_summary(
        summary={
            "is_valid_summary": True,
            "headline": (
                "포스코는 세일즈포스가 개최한 에이전트포스 월드투어 코리아 2026에서 "
                "AX 전략을 공개했다."
            ),
            "fact_summary": ["포스코DX가 3개월간 AI 에이전트 25개를 업무에 실험했다."],
        },
        articles=[
            {
                "id": 48508,
                "title": "포스코DX, AI 에이전트 기반 AX 전략 공개",
                "content": "포스코DX는 품질 점검과 영업 지원 업무에 AI 에이전트를 적용했다.",
                "published_at": "2026-06-12T09:00:00+09:00",
                "source_name": "news",
                "url": "https://example.com/posco-ax",
                "metadata": {"image_urls": ["https://cdn.example.com/news/posco-agent.jpg"]},
            }
        ],
        company="posco_dx",
        cluster_id=48508,
        representative_id=48508,
        classification={"event_type": "tech_release", "sector": "ax"},
    )

    assert card is not None
    assert card["title"] == "포스코DX, AI 에이전트 기반 AX 전략 공개"
    assert card["summary_lines"] == [
        "포스코DX가 3개월간 AI 에이전트 25개를 업무에 실험했다",
        "포스코DX, AI 에이전트 기반 AX 전략 공개",
        "포스코DX는 품질 점검과 영업 지원 업무에 AI 에이전트를 적용했다",
    ]
    assert card["image_assets"][0]["url"] == "https://cdn.example.com/news/posco-agent.jpg"
    assert (
        card["db_record"]["image_assets"][0]["url"]
        == "https://cdn.example.com/news/posco-agent.jpg"
    )


def test_literal_summary_lines_strip_number_prefixes():
    lines = _plain_summary_lines(
        {
            "fact_summary": [
                "1. LG CNS가 생성형 AI 서비스를 도입하기 위한 검증을 진행했다.",
                "2. 임직원 2500명을 대상으로 대표 서비스 3종을 선정했다.",
                "3. 특정 계열사가 아닌 그룹 전반 적용 가능한 계약으로 알려졌다.",
            ]
        }
    )

    assert lines == [
        "LG CNS가 생성형 AI 서비스를 도입하기 위한 검증을 진행했다.",
        "임직원 2500명을 대상으로 대표 서비스 3종을 선정했다.",
        "특정 계열사가 아닌 그룹 전반 적용 가능한 계약으로 알려졌다.",
    ]
