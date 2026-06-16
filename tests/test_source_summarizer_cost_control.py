from __future__ import annotations

from src.analysis import summarizer


def test_format_articles_uses_rule_based_snippets_and_dedupes_repeated_content(
    monkeypatch,
) -> None:
    monkeypatch.setattr(summarizer, "_SNIPPETS_PER_ARTICLE", 3)
    monkeypatch.setattr(summarizer, "_SNIPPET_DEDUP_SIMILARITY", 0.80)

    duplicate_content = "삼성SDS가 생성형 AI 플랫폼 A를 출시했다. 주요 기능은 업무 자동화다."
    articles = [
        {
            "id": 101,
            "title": "삼성SDS, 생성형 AI 플랫폼 A 출시",
            "content": duplicate_content,
            "source_type": "news",
            "importance_score": 0.9,
            "relevance_score": 0.9,
            "is_representative": True,
            "matched_companies": ["samsung_sds"],
        },
        {
            "id": 102,
            "title": "삼성SDS, 생성형 AI 플랫폼 A 출시",
            "content": duplicate_content,
            "source_type": "news",
            "importance_score": 1.0,
            "relevance_score": 1.0,
            "matched_companies": ["samsung_sds"],
        },
        {
            "id": 103,
            "title": "삼성SDS AI 플랫폼, 금융권 PoC 착수",
            "content": "삼성SDS는 플랫폼 A를 금융권 고객 PoC에 적용한다고 밝혔다.",
            "source_type": "news",
            "importance_score": 0.7,
            "relevance_score": 0.7,
            "matched_companies": ["samsung_sds"],
        },
    ]

    text = summarizer._format_articles(
        articles=articles,
        target_companies=["samsung_sds"],
        representative_id=101,
    )

    article_101 = text.split("[1] article_id: 101", maxsplit=1)[1].split(
        "[2] article_id: 102",
        maxsplit=1,
    )[0]
    article_102 = text.split("[2] article_id: 102", maxsplit=1)[1].split(
        "[3] article_id: 103",
        maxsplit=1,
    )[0]
    article_103 = text.split("[3] article_id: 103", maxsplit=1)[1]

    assert "content_policy: 원문 전체 content는 LLM에 넣지 않습니다" in text
    assert "article_role: evidence_snippets" in article_101
    assert "content:" not in article_101
    assert 'evidence_snippets: ["삼성SDS, 생성형 AI 플랫폼 A 출시"]' in article_101
    assert 'evidence_snippets: ["주요 기능은 업무 자동화다"]' in article_102
    assert "article_role: evidence_snippets" in article_103
    assert "삼성SDS는 플랫폼 A를 금융권 고객 PoC에 적용" in article_103


def test_format_articles_drops_article_ui_boilerplate(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "_SNIPPETS_PER_ARTICLE", 4)

    text = summarizer._format_articles(
        articles=[
            {
                "id": 201,
                "title": "삼성SDS, 오픈AI와 챗GPT 리셀러 계약 체결",
                "content": (
                    "뉴스 듣기 글자 크기 가 보통 가 크게 기사 공유 페이스북 "
                    "카카오톡 이메일 주소복사 북마크 다크모드 프린트 네이버 채널구독. "
                    "삼성SDS는 오픈AI와 챗GPT 엔터프라이즈 리셀러 계약을 체결했다."
                ),
                "matched_companies": ["samsung_sds"],
            }
        ],
        target_companies=["samsung_sds"],
        representative_id=201,
    )

    assert "뉴스 듣기" not in text
    assert "주소복사" not in text
    assert "챗GPT 엔터프라이즈 리셀러 계약" in text


def test_candidate_peer_companies_uses_preprocessing_targets_not_body_mentions() -> None:
    articles = [
        {
            "id": 101,
            "company": ["samsung_sds"],
            "matched_companies": ["samsung_sds"],
            "title": "삼성SDS 주가 강세",
            "content": "기사 본문에는 LG CNS와 현대오토에버도 비교 대상으로 언급됐다.",
        }
    ]

    assert summarizer._candidate_peer_companies(articles) == ["samsung_sds"]


def test_candidate_peer_companies_preserves_explicit_peer_comparison_mentions() -> None:
    articles = [
        {
            "id": 48524,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "title": "SI업계 엇갈린 내부거래 의존...포스코DX↑·LG CNS↓",
            "content": (
                "LG CNS는 내부거래 비중을 47.1%까지 낮췄고 삼성SDS는 79.2%를 기록했다. "
                "반면 포스코DX는 96.4%, 현대오토에버는 94.6%로 대조를 이뤘다."
            ),
        }
    ]

    assert summarizer._candidate_peer_companies(articles) == [
        "lg_cns",
        "samsung_sds",
        "hyundai_autoever",
        "posco_dx",
    ]


def test_peer_comparison_enrichment_keeps_structural_facts() -> None:
    articles = [
        {
            "id": 48524,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "title": "SI업계 엇갈린 내부거래 의존...포스코DX↑·LG CNS↓",
            "content": (
                "LG CNS는 올해 1분기 전체 매출 중 내부거래 비중을 47.1%까지 낮췄다. "
                "삼성SDS는 79.2%에 달하지만 전년 동기 대비 감소했다. "
                "포스코DX의 특수관계자 거래 매출은 전체 매출의 96.4%를 차지했다. "
                "현대오토에버 역시 내부거래액이 전체 매출의 94.6%를 기록했다. "
                "공정위는 SI 업종을 지속적으로 감시해야 할 분야라고 지적했다. "
                "AI와 클라우드 사업을 강화한 전략이 주효했다는 평가다. "
                "그룹 신사업과 밀접하게 연관되어 내부거래 비중이 높다는 분석도 있다. "
                "신규 사업이 계열사 중심으로 전개되면 외부 고객 확보가 제한될 수 있다. "
                "클라우드&AI 부문의 매출을 분리하지 않아 AI 성과를 검증하기 어렵다."
            ),
        }
    ]
    summary = {
        "main_company": "lg_cns",
        "mentioned_peer_companies": ["lg_cns"],
        "target_peer_companies": ["lg_cns"],
        "headline": "LG CNS, 내부거래 비중 47.1%로 최저 기록",
        "one_line_summary": "LG CNS는 내부거래 비중을 47.1%까지 낮췄다.",
        "fact_summary": ["LG CNS는 내부거래 비중을 47.1%까지 낮췄다."],
    }

    enriched = summarizer._enrich_peer_comparison_issue(
        summary,
        articles=articles,
        target_companies=summarizer._candidate_peer_companies(articles),
    )

    assert enriched["issue_frame"]["frame_type"] == "peer_comparison"
    assert enriched["issue_frame"]["comparison_axis"] == "internal_transaction_ratio"
    assert "posco_dx" in enriched["mentioned_peer_companies"]
    assert "hyundai_autoever" in enriched["mentioned_peer_companies"]
    assert any(fact["value"] == "96.4%" for fact in enriched["comparison_facts"])
    assert enriched["risk_facts"]
    assert enriched["market_structure_facts"]
    inventory = enriched["strategic_evidence_inventory"]
    assert inventory["core_facts"]
    assert inventory["supporting_facts"]
    assert inventory["background_facts"]
    assert inventory["cause_or_driver_facts"]
    assert inventory["risk_facts"]
    assert inventory["uncertainty_or_limitation_facts"]
    assert inventory["strategic_tensions"]
    assert inventory["actionable_questions"]
    assert any("외부 고객 확보" in fact for fact in inventory["supporting_facts"])
    assert any("검증하기 어렵" in fact for fact in inventory["uncertainty_or_limitation_facts"])


def test_contract_summary_prefers_business_scope_over_numeric_only_facts() -> None:
    facts = [
        {
            "fact_id": "f1",
            "article_id": 1,
            "summary_role": "numeric_effect",
            "normalized_fact": "계약금액은 총 24억4674만원이다.",
            "evidence_text": "계약금액은 총 24억4674만원이며 최근 매출 대비 6.30%다.",
            "numbers": ["24억4674만원", "6.30%"],
            "dates": [],
            "entities": [],
            "event_verbs": [],
            "confidence": "high",
        },
        {
            "fact_id": "f2",
            "article_id": 1,
            "summary_role": "main_event",
            "normalized_fact": "플래티어는 현대오토에버와 차량 서비스 플랫폼 공급계약을 체결했다.",
            "evidence_text": "플래티어가 현대오토에버와 차량 서비스 플랫폼 공급계약을 체결했다.",
            "numbers": [],
            "dates": [],
            "entities": ["차량 서비스 플랫폼 공급계약"],
            "event_verbs": ["체결"],
            "confidence": "high",
        },
        {
            "fact_id": "f3",
            "article_id": 1,
            "summary_role": "service_function",
            "normalized_fact": "공급 범위는 고객 서비스 운영 시스템과 플랫폼 고도화 업무다.",
            "evidence_text": "공급 범위는 고객 서비스 운영 시스템과 플랫폼 고도화 업무다.",
            "numbers": [],
            "dates": [],
            "entities": ["고객 서비스 운영 시스템"],
            "event_verbs": [],
            "confidence": "medium",
        },
    ]

    selected = summarizer._select_fact_ids_for_summary_lines(
        extracted_facts=facts,
        cluster_event_type="contract",
    )

    assert selected["1"] == ["f2"]
    assert selected["2"] == ["f3"]
    assert selected["3"] == ["f1"]


def test_non_financial_summary_rejects_numeric_only_lines() -> None:
    facts = [
        {
            "fact_id": "f1",
            "article_id": 1,
            "summary_role": "numeric_effect",
            "fact_type": "numeric_fact",
            "normalized_fact": "계약금액은 총 24억4674만원이다.",
            "evidence_text": "계약금액은 총 24억4674만원이다.",
            "numbers": ["24억4674만원"],
        },
        {
            "fact_id": "f2",
            "article_id": 1,
            "summary_role": "numeric_effect",
            "fact_type": "numeric_fact",
            "normalized_fact": "최근 매출 대비 6.30%에 해당한다.",
            "evidence_text": "최근 매출 대비 6.30%에 해당한다.",
            "numbers": ["6.30%"],
        },
        {
            "fact_id": "f3",
            "article_id": 1,
            "summary_role": "numeric_effect",
            "fact_type": "numeric_fact",
            "normalized_fact": "연간 운영계약에 이은 추가 수주 건이다.",
            "evidence_text": "연간 운영계약에 이은 추가 수주 건이다.",
            "numbers": [],
        },
    ]
    result = {
        "is_valid_summary": True,
        "cluster_event_type": "contract",
        "fact_summary": [fact["normalized_fact"] for fact in facts],
        "summary_lines_with_fact_ids": [
            {"line_index": 1, "text": facts[0]["normalized_fact"], "fact_ids": ["f1"]},
            {"line_index": 2, "text": facts[1]["normalized_fact"], "fact_ids": ["f2"]},
            {"line_index": 3, "text": facts[2]["normalized_fact"], "fact_ids": ["f3"]},
        ],
    }

    checked = summarizer._validate_fact_id_summary(
        result=result,
        extracted_facts=facts,
        source_article_ids=[],
        main_company="hyundai_autoever",
    )

    assert checked["is_valid_summary"] is False
    assert "수치/시장반응 중심" in checked["reason"]


def test_validate_fact_id_summary_detects_cross_company_attribution() -> None:
    facts = [
        {
            "fact_id": "f1",
            "article_id": 1,
            "summary_role": "main_event",
            "fact_type": "application_fact",
            "normalized_fact": (
                "삼성전자는 임직원 2500여명을 대상으로 서비스 실효성 검증을 거쳐 "
                "생성형 AI 서비스 3종을 선정했다."
            ),
            "evidence_text": (
                "삼성전자는 임직원 2500여명을 대상으로 서비스 실효성 검증을 거쳐 "
                "생성형 AI 서비스 3종을 선정했다."
            ),
            "numbers": ["2500여명", "3종"],
        },
        {
            "fact_id": "f2",
            "article_id": 1,
            "summary_role": "service_function",
            "fact_type": "application_fact",
            "normalized_fact": "LG CNS는 앤트로픽과 클로드 엔터프라이즈 통합 계약을 체결했다.",
            "evidence_text": "LG CNS는 앤트로픽과 클로드 엔터프라이즈 통합 계약을 체결했다.",
            "numbers": [],
        },
        {
            "fact_id": "f3",
            "article_id": 1,
            "summary_role": "application_case",
            "fact_type": "application_fact",
            "normalized_fact": (
                "계약은 특정 계열사가 아니라 그룹 전반에 적용 가능한 형태로 알려졌다."
            ),
            "evidence_text": "계약은 특정 계열사가 아니라 그룹 전반에 적용 가능한 형태로 알려졌다.",
            "numbers": [],
        },
    ]
    result = {
        "is_valid_summary": True,
        "cluster_event_type": "general_update",
        "fact_summary": [
            (
                "LG CNS는 임직원 2500여명을 대상으로 서비스 실효성 검증을 거쳐 "
                "생성형 AI 서비스 3종을 선정했다."
            ),
            facts[1]["normalized_fact"],
            facts[2]["normalized_fact"],
        ],
        "summary_lines_with_fact_ids": [
            {"line_index": 1, "text": "", "fact_ids": ["f1"]},
            {"line_index": 2, "text": "", "fact_ids": ["f2"]},
            {"line_index": 3, "text": "", "fact_ids": ["f3"]},
        ],
    }

    checked = summarizer._validate_fact_id_summary(
        result=result,
        extracted_facts=facts,
        source_article_ids=[],
        main_company="lg_cns",
    )

    assert checked["is_valid_summary"] is False
    assert "회사 주체 귀속 불일치" in checked["reason"]


def test_extracted_facts_drop_article_body_noise_without_title_overlap() -> None:
    facts = summarizer._build_extracted_facts(
        cluster_id=46515,
        article_fact_notes=[
            {
                "article_id": 46223,
                "core_facts": [
                    {
                        "fact": (
                            "젠슨 황은 출국길에 한국 기술 없이는 "
                            "AI 슈퍼컴을 만들 수 없다고 언급했다."
                        ),
                        "evidence_text": (
                            "젠슨 황은 출국길에 한국 기술 없이는 AI 슈퍼컴을 "
                            "만들 수 없다고 언급했다."
                        ),
                        "activity_type": "general_update",
                        "summary_role": "main_event",
                    },
                    {
                        "fact": (
                            "네이버클라우드, 삼성SDS, 엘리스그룹이 GPU 확보·구축 사업자로 선정됐다."
                        ),
                        "evidence_text": (
                            "정부는 네이버클라우드, 삼성SDS, 엘리스그룹을 "
                            "GPU 확보·구축 사업자로 선정했다."
                        ),
                        "activity_type": "contract",
                        "summary_role": "main_event",
                    },
                ],
            }
        ],
        articles=[
            {
                "id": 46223,
                "title": "[속보]정부, 네이버클라우드·삼성SDS·엘리스그룹에 2조 규모 GPU 지원",
                "matched_companies": ["samsung_sds"],
            }
        ],
        cluster_event_type="contract",
    )

    texts = [fact["normalized_fact"] for fact in facts]

    assert any("GPU 확보" in text for text in texts)
    assert all("젠슨 황" not in text for text in texts)


def test_contract_summary_drops_embedded_stock_market_fact() -> None:
    facts = summarizer._build_extracted_facts(
        cluster_id=36453,
        article_fact_notes=[
            {
                "article_id": 36453,
                "core_facts": [
                    {
                        "fact": (
                            "삼성SDS가 오픈AI와 챗GPT 엔터프라이즈 리셀러 파트너 계약을 체결했다."
                        ),
                        "evidence_text": (
                            "삼성SDS가 오픈AI와 챗GPT 엔터프라이즈 리셀러 파트너 계약을 체결했다."
                        ),
                        "activity_type": "contract",
                        "summary_role": "main_event",
                    },
                    {
                        "fact": "삼성SDS의 주가는 164,500원으로 전일 대비 6,500원 하락했다.",
                        "evidence_text": (
                            "삼성SDS KOSPI 현재가 164,500 전일대비 6,500 "
                            "등락률 -3.80% 거래량 258,315."
                        ),
                        "activity_type": "contract",
                        "summary_role": "numeric_effect",
                    },
                ],
            }
        ],
        articles=[
            {
                "id": 36453,
                "title": "삼성SDS, 오픈AI와 챗GPT 엔터프라이즈 리셀러 파트너 계약",
                "matched_companies": ["samsung_sds"],
            }
        ],
        cluster_event_type="contract",
    )

    texts = [fact["normalized_fact"] for fact in facts]

    assert any("리셀러 파트너 계약" in text for text in texts)
    assert all("주가" not in text for text in texts)
    assert all("등락률" not in text for text in texts)


def test_large_cluster_selects_diverse_articles_from_majority_event(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "_MIN_ANALYZED_ARTICLES", 8)
    monkeypatch.setattr(summarizer, "_MAX_ANALYZED_ARTICLES", 20)
    monkeypatch.setattr(summarizer, "_MAJORITY_THRESHOLD", 0.70)
    monkeypatch.setattr(summarizer, "_MIXED_THRESHOLD", 0.50)

    majority_titles = [
        "LG CNS, 클로드 엔터프라이즈 도입 계약 체결",
        "LG CNS, 클로드 도입 계약으로 그룹 AX 확산",
        "LG CNS, 클로드 엔터프라이즈 전사 활용 계약",
        "LG CNS, 기업용 클로드 계약 체결",
    ]
    articles = [
        {
            "id": article_id,
            "title": majority_titles[article_id % len(majority_titles)],
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "content": "LG CNS가 클로드 엔터프라이즈 도입 계약을 체결했다.",
            "relevance_score": 0.8,
            "importance_score": 0.8,
            "is_representative": article_id == 1,
        }
        for article_id in range(1, 66)
    ]
    articles.extend(
        {
            "id": article_id,
            "title": f"LG CNS, 다른 주제 플랫폼 출시 {article_id}",
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "content": "별도 플랫폼 출시 기사다.",
            "relevance_score": 0.8,
            "importance_score": 0.8,
        }
        for article_id in range(66, 77)
    )

    selection = summarizer._select_analysis_articles(articles=articles, representative_id=1)

    assert selection["status"] == "sampled_majority_group"
    assert len(selection["selected_article_ids"]) == 20
    assert selection["majority_ratio"] >= 0.70
    assert 1 in selection["selected_article_ids"]
    assert set(selection["selected_article_ids"]).issubset(set(selection["majority_article_ids"]))
    assert selection["outlier_article_ids"]


def test_large_mixed_cluster_without_majority_is_blocked(monkeypatch) -> None:
    monkeypatch.setattr(summarizer, "_MIN_ANALYZED_ARTICLES", 8)
    monkeypatch.setattr(summarizer, "_MAX_ANALYZED_ARTICLES", 20)
    monkeypatch.setattr(summarizer, "_MIXED_THRESHOLD", 0.50)

    groups = [
        ("계약 체결", "계약을 체결했다."),
        ("플랫폼 출시", "플랫폼을 출시했다."),
        ("주가 강세", "주가가 강세를 보였다."),
    ]
    articles = []
    article_id = 1
    for title_suffix, content in groups:
        for _ in range(10):
            articles.append(
                {
                    "id": article_id,
                    "title": f"LG CNS, {title_suffix} {article_id}",
                    "company": ["lg_cns"],
                    "matched_companies": ["lg_cns"],
                    "content": content,
                    "relevance_score": 0.8,
                    "importance_score": 0.8,
                }
            )
            article_id += 1

    selection = summarizer._select_analysis_articles(articles=articles, representative_id=1)

    assert selection["status"] == "mixed_cluster_no_majority"
    assert selection["selected_article_ids"] == []
    assert selection["excluded_article_ids"] == list(range(1, 31))
