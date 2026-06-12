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

    assert card["id"] == "CN-20260610-0001"
    assert card["published_date"] == "2026-06-10"
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
    assert "SK AX" in action_text
    assert "운영 책임" in action_text


def test_card_news_from_cluster_summary_uses_source_published_date(monkeypatch):
    import src.composers.card_news_composer as composer_module

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
