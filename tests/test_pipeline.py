# 작성일: 2026-04-21
# 작성자: 최종민
# 변경이력:
#   2026-04-21 최종민 — axis-ai 베이스라인 구축, issue_cards→card_news 리네임
#   2026-04-28 박지원 — 크롤러 구현 및 전처리·클러스터링 파이프라인 개선
#   2026-06-18 최종민 — 코드 변경
"""파이프라인 단위 테스트"""

import numpy as np

from scripts import reprocess_news_clusters
from scripts.reprocess_news_clusters import _params, _target_where_sql
from src.preprocessing import dedup
from src.preprocessing.dedup import (
    _cluster,
    _company_presence_score,
    _event_bucket,
    _event_signature,
    _existing_cluster_candidate_window,
    _rule_prefilter_key,
    _same_issue,
    _should_merge_articles,
)
from src.preprocessing.preprocessing import (
    LinkCheckResult,
    PreprocessingResult,
    PreprocessingService,
)
from src.preprocessing.relevance import (
    _core_company_role_reject_result,
    _fast_pass_result,
    _noise_reject_result,
    analyze_relevance_article,
)


def test_ingestion_state_structure():
    state: PreprocessingResult = {
        "company": ["samsung_sds"],
        "trigger_type": "scheduled",
        "collected_since": None,
        "crawl_run_id": None,
        "raw_article_ids": [],
        "relevant_ids": [],
        "official_document_ids": [],
        "parsed_document_ids": [],
        "industry_document_ids": [],
        "structured_signal_ids": [],
        "skipped_preprocess_ids": [],
        "cluster_map": {},
        "representative_ids": [],
        "classified_clusters": [],
        "errors": [],
        "human_review_flags": [],
    }
    assert state["company"] == ["samsung_sds"]


def test_reprocess_news_clusters_supports_published_until_filter():
    params = _params(
        ["news"],
        [],
        ["RAW"],
        "2026-05-31T00:00:00+00:00",
        "2026-06-01T00:00:00+00:00",
    )
    where_sql = _target_where_sql()

    assert params["published_since"] == "2026-05-31T00:00:00+00:00"
    assert params["published_until"] == "2026-06-01T00:00:00+00:00"
    assert "published_at >= CAST(:published_since AS timestamptz)" in where_sql
    assert "published_at < CAST(:published_until AS timestamptz)" in where_sql


def test_preprocessing_skips_dead_news_links_before_relevance(monkeypatch):
    marked: list[tuple[int, str, int | None]] = []

    def fake_mark(article_id, url, result):  # noqa: ANN001
        marked.append((article_id, url, result.http_code))

    monkeypatch.setattr(
        "src.preprocessing.preprocessing._mark_dead_link_article",
        fake_mark,
    )

    def fake_link_checker(url: str) -> LinkCheckResult:
        if "dead.example" in url:
            return LinkCheckResult(status="dead", http_code=404, final_url=url)
        return LinkCheckResult(status="live", http_code=200, final_url=url)

    service = PreprocessingService(
        verify_news_links=True,
        link_checker=fake_link_checker,
    )
    skipped = service.skip_dead_news_links(
        [
            {
                "id": 1,
                "source_type": "news",
                "url": "https://dead.example/news/1",
            },
            {
                "id": 2,
                "source_type": "news",
                "url": "https://live.example/news/2",
            },
        ],
        [1, 2],
    )

    assert skipped == [1]
    assert marked == [(1, "https://dead.example/news/1", 404)]


def test_industry_news_relevance_fast_pass_without_company_mention():
    article = {
        "id": 51009,
        "company": ["industry_trend"],
        "source_type": "news",
        "source_name": "naver_industry_news",
        "title": "AI 경쟁 승부처는 전력·데이터·제도…산업현장 확산이 성패 가른다",
        "content": "AI 경쟁이 모델 성능을 넘어 산업 현장과 전력, 데이터 인프라로 확장되고 있다.",
        "metadata": {"topic_scope": "industry_trend", "matched_sectors": ["ax", "infra"]},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is True
    assert enriched["relevance_label"] == "relevant"
    assert enriched["metadata"]["topic_scope"] == "industry_trend"


def test_industry_news_relevance_blocks_single_company_certification_noise():
    article = {
        "id": 51010,
        "company": ["industry_trend"],
        "source_type": "news",
        "source_name": "naver_industry_news",
        "title": "영풍, 정보보호 국제표준 'ISO 27001' 인증 획득",
        "content": "개별 기업의 인증 획득 소식이다.",
        "metadata": {"topic_scope": "industry_trend", "matched_sectors": ["security"]},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is False
    assert enriched["relevance_label"] == "irrelevant"
    assert "인증 획득 단신" in enriched["relevance_reason"]


def test_industry_news_relevance_keeps_broad_security_framework_anchor():
    article = {
        "id": 51012,
        "company": ["industry_trend"],
        "source_type": "news",
        "source_name": "naver_industry_news",
        "title": "N2SF 국가 망 보안체제 시대 공공 보안시장 공략",
        "content": "국가 망 보안체제와 제로트러스트 전환이 공공 보안시장 변화로 제시됐다.",
        "metadata": {"topic_scope": "industry_trend", "matched_sectors": ["security"]},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is True
    assert enriched["relevance_label"] == "relevant"


def test_industry_news_relevance_keeps_ai_adoption_infrastructure_signal():
    article = {
        "id": 51011,
        "company": ["industry_trend"],
        "source_type": "news",
        "source_name": "naver_industry_news",
        "title": "대학 63% 생성형 AI 도입했지만 인프라·인력·예산 미비",
        "content": "교육 분야 생성형 AI 도입이 확산됐지만 인프라와 인력 부족이 과제로 제시됐다.",
        "metadata": {"topic_scope": "industry_trend", "matched_sectors": ["ax", "infra"]},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is True
    assert enriched["relevance_label"] == "relevant"


def test_news_relevance_blocks_title_company_missing_from_body_parser_noise():
    article = {
        "id": 1,
        "company": ["lg_cns"],
        "source_type": "news",
        "source_name": "naver_news",
        "title": "LG CNS, 공공 AI 플랫폼 구축 협력",
        "content": (
            "법원은 민사 사건에서 손해배상 판결을 선고했다. 원고와 피고의 "
            "항소심 판단, 소송 절차, 증거 제출 기한에 대한 내용만 포함됐다. "
            "재판부는 청구 원인과 변론 경과를 검토한 뒤 일부 청구를 기각하고 "
            "판결문 송달 절차와 불복 기간을 안내했다."
        ),
        "metadata": {},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is False
    assert enriched["relevance_label"] == "irrelevant"
    assert "파싱 품질 문제" in enriched["relevance_reason"]


def test_news_relevance_blocks_peer_only_participant_listing():
    article = {
        "id": 2,
        "company": ["posco_dx"],
        "source_type": "news",
        "source_name": "naver_news",
        "title": "AI 보안 협력체 출범, 공익 인프라 강화 추진",
        "content": (
            "국내 기업과 기관들이 AI 보안 협력체를 출범했다. "
            "A사, 포스코DX, B사 등 5개 기업은 핵심 운영 주체 그룹으로 참여하며, "
            "공공기관과 연구기관도 파트너사로 이름을 올렸다. "
            "협력체는 취약점 탐지 기술을 민생 인프라 전반으로 확산하는 것을 목표로 한다."
        ),
        "metadata": {},
        "crawl_status": "success",
    }

    enriched, is_relevant = analyze_relevance_article(article)

    assert is_relevant is False
    assert enriched["relevance_label"] == "irrelevant"
    assert "참여사 목록" in enriched["relevance_reason"]


def test_same_issue_does_not_merge_on_customer_name_only():
    left = {
        "company": ["lg_cns"],
        "title": "요금 심사부터 설비 운영까지…LG CNS, AI로 한전 시스템 전환",
        "content": "LG CNS가 한전 전력관리 시스템을 AI로 재설계한다.",
    }
    right = {
        "company": ["lg_cns"],
        "title": "LG CNS, 한국전력 차세대 영업배전시스템 구축 위한 ISP 컨설팅 사업",
        "content": "한국전력 차세대 영업배전시스템 ISP 컨설팅 사업을 수주했다.",
    }

    assert _same_issue(left, right) is False


def test_same_issue_groups_shared_specific_terms_and_numbers():
    left = {
        "company": ["lg_cns"],
        "matched_sectors": ["infra"],
        "title": "LG CNS, 차세대 영업배전시스템 2500만 고객 서비스 전환",
        "content": "차세대 영업배전시스템을 AI 기반으로 고도화한다.",
    }
    right = {
        "company": ["lg_cns"],
        "matched_sectors": ["infra"],
        "title": "LG CNS, 차세대 영업배전시스템 전환 프로젝트 착수",
        "content": "2500만 고객 대상 차세대 영업배전시스템 구축 사업이다.",
    }

    assert _same_issue(left, right) is True


def test_same_contract_domain_uses_general_signature_without_event_hardcoding(monkeypatch):
    left = {
        "id": 47000,
        "company": ["hyundai_autoever"],
        "title": "[특징주] 플래티어, 현대오토에버와 24억 규모 추가 공급계약 체결",
        "content": "플래티어가 현대오토에버와 인증중고차 관련 공급계약을 체결했다.",
        "published_at": "2026-06-09T11:30:00+09:00",
    }
    right = {
        "id": 46955,
        "company": ["hyundai_autoever"],
        "title": '플래티어, 현대오토에버와 연속 수주..."45억원 확보"',
        "content": "플래티어가 현대오토에버 인증중고차 플랫폼 운영 관련 연속 수주 성과를 냈다.",
        "published_at": "2026-06-09T10:51:00+09:00",
    }

    assert _event_signature(left).startswith("contract_deal:")
    assert _event_signature(right).startswith("contract_deal:")
    assert _should_merge_articles(left, right, similarity=0.82, threshold=0.8) is True


def test_same_company_security_action_articles_merge_across_bucket_noise():
    left = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, AI 보안 스타트업 손잡고 클라우드 보안 강화",
        "content": "",
        "published_at": "2026-06-10T13:48:00+09:00",
    }
    right = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, AI 보안 전선 넓힌다…국내외 전문기업과 맞손",
        "content": "",
        "published_at": "2026-06-10T14:36:00+09:00",
    }

    assert _rule_prefilter_key(left) == _rule_prefilter_key(right)
    assert _event_bucket(left) != _event_bucket(right)
    assert _should_merge_articles(left, right, similarity=0.82, threshold=0.8) is True


def test_same_company_product_access_articles_merge_with_particle_normalization():
    left = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, OpenAI와 손잡고 챗GPT 에듀 확대",
        "content": "",
        "published_at": "2026-04-27T08:00:00+09:00",
    }
    right = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 교육기관 대상 '챗GPT 에듀' 판매권 확보",
        "content": "",
        "published_at": "2026-04-27T08:10:00+09:00",
    }

    assert "openai" in dedup._title_event_tokens(left)
    assert _should_merge_articles(left, right, similarity=0.72, threshold=0.8) is True


def test_title_llm_post_merge_allows_concrete_product_overlap():
    article_by_id = {
        1: {
            "id": 1,
            "title": "삼성SDS, ‘챗GPT 에듀’ 판매 확대…교육용 생성형 AI 시장 공략",
            "matched_companies": ["samsung_sds"],
            "published_at": "2026-04-27T10:00:00+09:00",
        },
        2: {
            "id": 2,
            "title": "삼성SDS, '챗GPT 에듀' 리셀러 권한 추가 확보…OpenAI와 협력 강화",
            "matched_companies": ["samsung_sds"],
            "published_at": "2026-04-27T10:10:00+09:00",
        },
    }

    left = reprocess_news_clusters._title_cluster_features([1], article_by_id)
    right = reprocess_news_clusters._title_cluster_features([2], article_by_id)

    assert "openai" in reprocess_news_clusters._title_merge_tokens(article_by_id[2]["title"])
    assert reprocess_news_clusters._title_clusters_related(left, right) is True


def test_security_action_articles_do_not_merge_on_security_only():
    left = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 보안 솔루션 출시로 고객 대응 강화",
        "content": "",
        "published_at": "2026-06-10T09:00:00+09:00",
    }
    right = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 보안 기업과 협력 확대",
        "content": "",
        "published_at": "2026-06-10T10:00:00+09:00",
    }

    assert _should_merge_articles(left, right, similarity=0.82, threshold=0.8) is False


def test_representative_prefers_article_with_company_in_title():
    title_article = {
        "company": ["lg_cns"],
        "title": "LG CNS, 한전 차세대 시스템 구축",
        "content": "한국전력 시스템 구축 사업을 수주했다.",
    }
    no_title_article = {
        "company": ["lg_cns"],
        "title": "한국전력, 2500만 이용 전력 서비스 AI로 재설계",
        "content": "LG CNS가 한국전력 시스템 구축 사업을 맡았다.",
    }

    assert _company_presence_score(title_article) > _company_presence_score(no_title_article)


def test_cluster_merge_blocks_different_event_buckets_even_with_high_similarity():
    ax_strategy = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성전자 AI 자율공장 전환 가속…수혜주는 삼성SDS",
        "content": "삼성그룹의 AX 확산에 따라 삼성SDS의 스마트팩토리 사업 기회가 커졌다.",
        "published_at": "2026-05-29T06:39:00+00:00",
    }
    market_reaction = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성에스디에스, 장중 24% 급등…두나무 투자·AX 기대감",
        "content": "삼성에스디에스 주가가 두나무 지분 취득과 AX 기대감에 급등했다.",
        "published_at": "2026-05-29T04:56:00+00:00",
    }

    assert _should_merge_articles(ax_strategy, market_reaction, 0.97, 0.80) is False


def test_cluster_merge_keeps_different_event_buckets_separate_even_with_llm_env(monkeypatch):
    monkeypatch.setenv("ENABLE_OPENAI_CALLS", "true")
    monkeypatch.setenv("ENABLE_CLUSTER_LLM_JUDGE", "true")
    ax_strategy = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성전자 AI 자율공장 전환 가속…수혜주는 삼성SDS",
        "content": "삼성그룹의 AX 확산에 따라 삼성SDS의 스마트팩토리 사업 기회가 커졌다.",
        "published_at": "2026-05-29T06:39:00+00:00",
    }
    market_reaction = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성에스디에스, 장중 24% 급등…두나무 투자·AX 기대감",
        "content": "삼성에스디에스 주가가 두나무 지분 취득과 AX 기대감에 급등했다.",
        "published_at": "2026-05-29T04:56:00+00:00",
    }

    assert _should_merge_articles(ax_strategy, market_reaction, 0.97, 0.80) is False


def test_dunamu_stake_articles_use_investment_bucket_before_market_reaction():
    left = {
        "id": 43264,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "matched_sectors": ["security", "infra", "deal"],
        "title": "삼성증권·삼성SDS·삼성카드, 두나무 지분 4% 공동 인수",
        "content": "",
        "published_at": "2026-05-28T08:52:00+00:00",
    }
    right = {
        "id": 43202,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "matched_sectors": ["security", "infra", "deal"],
        "title": (
            "'오픈AI 협력' 판매 채널 우선하는 삼성SDS-LG CNS·운영 역량 내세워…두나무 지분 투자"
        ),
        "content": "",
        "published_at": "2026-05-28T02:10:00+00:00",
    }

    assert _event_bucket(left) == "investment_deal"
    assert _event_bucket(right) == "investment_deal"
    assert _event_signature(left) == "investment_deal:dunamu"
    assert _event_signature(right) == "investment_deal:dunamu"
    assert _should_merge_articles(left, right, 0.80, 0.80) is True


def test_event_signature_splits_market_reaction_by_day():
    first_day = {
        "title": "삼성에스디에스 주가 장중 급등",
        "content": "주가가 급등했다.",
        "published_at": "2026-05-27T05:56:00+00:00",
    }
    second_day = {
        "title": "삼성에스디에스 주가 장중 하락",
        "content": "주가가 하락했다.",
        "published_at": "2026-06-02T02:12:00+00:00",
    }

    assert _event_signature(first_day) == "market_reaction:2026-05-27"
    assert _event_signature(second_day) == "market_reaction:2026-06-02"


def test_cluster_splits_union_bridge_by_event_bucket(monkeypatch):
    monkeypatch.setattr(dedup, "_should_merge_articles", lambda *args, **kwargs: True)
    articles = [
        {
            "id": 1,
            "company": ["samsung_sds"],
            "title": "삼성SDS AX 서밋서 AI 네이티브 전환 전략 공개",
            "content": "AX 로드맵을 공개했다.",
        },
        {
            "id": 2,
            "company": ["samsung_sds"],
            "title": "삼성SDS 두나무 지분 투자 결정",
            "content": "두나무 지분을 취득했다.",
        },
        {
            "id": 3,
            "company": ["samsung_sds"],
            "title": "삼성에스디에스 주가 장중 급등",
            "content": "주가가 급등했다.",
        },
    ]
    embeddings = np.ones((3, 2), dtype=np.float32)

    cluster_map = _cluster(articles=articles, embeddings=embeddings, threshold=0.8)

    assert sorted(len(ids) for ids in cluster_map.values()) == [1, 1, 1]


def test_fast_pass_is_limited_to_core_monitoring_companies():
    result = _fast_pass_result(
        title="NC AI, 한화오션 자율용접 로봇 AI 두뇌 개발",
        content="NC AI가 한화오션과 자율용접 로봇 AI 두뇌를 개발했다.",
        source_type="news",
        matched_companies=["nc_ai"],
        matched_sectors=["ax"],
    )

    assert result is None


def test_relevance_rejects_pure_market_price_article():
    result = _noise_reject_result(
        title="삼성에스디에스 주가, 6월 4일 장중 261,750원 12.91% 하락",
        content="삼성에스디에스 주가가 장중 하락했다.",
        source_type="news",
        matched_companies=["samsung_sds"],
        matched_sectors=["other"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_rejects_multi_company_roundup_news_title():
    titles = [
        "[#시큐리티 포커스] 유락 '디파스 프로 맥' 출시·삼성SDS 'AI 클라우드 ...",
        "[전자·IT 레이더] 삼성SDS·한컴·카페24, 보안·AI·커머스 핵심 사업",
        "[민주 IT] LG CNS·LG유플러스·KT",
    ]

    for title in titles:
        result = _noise_reject_result(
            title=title,
            content="여러 보안 기업과 IT 기업의 소식을 묶어 전한다.",
            source_type="news",
            matched_companies=["samsung_sds"],
            matched_sectors=["security"],
        )

        assert result is not None
        assert result["relevance_label"] == "irrelevant"
        assert "섹션형" in result["reason"]


def test_relevance_rejects_operational_campaign_news_title():
    titles = [
        "현대오토에버, 차량 5부제 확대 시행…에너지 절약 동참",
        "A그룹, 차량 5부제 확대 시행…에너지 절약 정책 동참",
    ]

    for title in titles:
        result = _noise_reject_result(
            title=title,
            content="그룹 차원의 에너지 절약 캠페인에 참여한다.",
            source_type="news",
            matched_companies=["hyundai_autoever"],
            matched_sectors=["ax"],
        )

        assert result is not None
        assert result["relevance_label"] == "irrelevant"


def test_relevance_keeps_event_driven_market_article_for_analysis():
    result = _noise_reject_result(
        title="[특징주] 삼성SDS, AI 데이터센터 수혜 기대감에 28%대 급등",
        content="삼성SDS가 AI 데이터센터와 공공 AI 수혜 기대감에 급등했다.",
        source_type="news",
        matched_companies=["samsung_sds"],
        matched_sectors=["ax"],
    )

    assert result is None


def test_relevance_keeps_external_supplier_contract_for_downstream_ranking():
    result = _noise_reject_result(
        title="인스웨이브, LG CNS와 99억원 규모 계약 체결",
        content="인스웨이브가 LG CNS와 코어뱅킹 현대화 웹단말 전환 계약을 체결했다.",
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["deal"],
    )

    assert result is None


def test_relevance_keeps_external_partner_agreement_for_downstream_ranking():
    result = _noise_reject_result(
        title="NC AI, 포스코DX와 로봇 파운데이션 모델 공동 개발 업무협약 체결",
        content="NC AI가 포스코DX와 산업현장용 로봇 파운데이션 모델을 공동 개발한다.",
        source_type="news",
        matched_companies=["posco_dx"],
        matched_sectors=["ax"],
    )

    assert result is None


def test_relevance_fast_pass_keeps_peer_customer_robot_poc_news():
    title = "LG CNS-컬리, 물류센터 휴머노이드 PoC…'피지컬웍스' 현장 검증"
    content = "LG CNS와 컬리가 물류센터에 휴머노이드 로봇을 도입하고 자동화 실증을 추진한다."

    role_reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["infra", "deal"],
    )
    result = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["infra", "deal"],
    )

    assert role_reject is None
    assert result is not None
    assert result["relevance_label"] == "relevant"


def test_relevance_fast_pass_keeps_peer_partner_robot_adoption_news():
    title = "LG CNS, 컬리 물류센터에 휴머노이드 로봇 도입 맞손"
    content = "LG CNS가 컬리와 손잡고 물류센터 휴머노이드 로봇 실증과 자동화 협력을 진행한다."

    role_reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["infra", "deal"],
    )
    result = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["infra", "deal"],
    )

    assert role_reject is None
    assert result is not None
    assert result["relevance_label"] == "relevant"


def test_relevance_fast_pass_keeps_peer_executive_strategy_article():
    title = '이주평 삼성SDS 상무 "제조AI 핵심 데이터는 시계열"'
    content = (
        "이주평 삼성SDS 상무가 M.AX 컨퍼런스에서 제조 AX 전환과 "
        "제조 데이터, AI 인프라 경쟁력을 발표했다."
    )

    role_reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["samsung_sds"],
        matched_sectors=["ax", "infra"],
    )
    result = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["samsung_sds"],
        matched_sectors=["ax", "infra"],
    )

    assert role_reject is None
    assert result is not None
    assert result["relevance_label"] == "relevant"


def test_core_role_defers_title_peer_sector_trend_article_to_llm():
    title = "[이슈딜] 메모리 다음은 AX…네이버·LG CNS 뜬다"
    content = "AX 시장 확대 속에서 네이버와 LG CNS의 클라우드·AI 사업 성장성이 주목된다."

    role_reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )
    fast_pass = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )

    assert role_reject is None
    assert fast_pass is None


def test_relevance_fast_pass_keeps_lg_cns_agentic_ai_platform_release():
    title = "LG CNS, 에이전틱 AI 개발 플랫폼 출시…대규모 시스템 구축 자동화"
    content = (
        "LG CNS가 데브온 에이전틱 AIND를 출시하고 IT 시스템 구축 전 과정과 "
        "운영 전 과정을 자동화한다고 밝혔다."
    )

    role_reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax", "security", "deal"],
    )
    result = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax", "security", "deal"],
    )

    assert role_reject is None
    assert result is not None
    assert result["relevance_label"] == "relevant"


def test_relevance_keeps_peer_subject_with_external_counterparty():
    result = _noise_reject_result(
        title="포스코DX, NC AI와 손잡고 산업현장용 피지컬AI 개발",
        content="포스코DX가 NC AI와 로봇 파운데이션 모델을 공동 개발한다.",
        source_type="news",
        matched_companies=["posco_dx"],
        matched_sectors=["ax"],
    )

    assert result is None


def test_cluster_merge_allows_same_business_issue_with_different_titles(monkeypatch):
    left = {
        "company": ["nc_ai"],
        "matched_companies": ["nc_ai"],
        "matched_sectors": ["ax"],
        "title": "NC AI, 한화오션 자율용접 로봇 AI 두뇌 개발",
        "content": "NC AI가 한화오션과 자율용접 로봇 AI 두뇌를 개발했다.",
        "published_at": "2026-06-03T23:52:00+00:00",
    }
    right = {
        "company": ["nc_ai"],
        "matched_companies": ["nc_ai"],
        "matched_sectors": ["ax"],
        "title": "NC AI가 조선소 용접에 AI 두뇌 심는 이유는?",
        "content": "한화오션 자율용접 로봇에 NC AI 기술을 적용하는 내용이다.",
        "published_at": "2026-06-03T23:06:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.83, 0.80) is True


def test_cluster_merge_groups_same_contract_articles():
    left = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["deal"],
        "title": "인스웨이브, LG CNS와 99억원 규모 계약 체결",
        "content": "코어뱅킹 현대화 웹단말 전환 계약을 체결했다.",
        "published_at": "2026-06-02T08:20:00+00:00",
    }
    right = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["deal"],
        "title": "인스웨이브, LG CNS와 99억 규모 공급 계약",
        "content": "LG CNS와 코어뱅킹 현대화 웹단말 전환 사업 공급계약을 맺었다.",
        "published_at": "2026-06-02T05:33:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.80, 0.80) is True


def test_event_driven_contract_market_article_uses_llm_to_merge_with_contract_cluster(
    monkeypatch,
):
    contract = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["deal"],
        "title": "인스웨이브, LG CNS와 99억원 규모 계약 체결",
        "content": "코어뱅킹 현대화 웹단말 전환 계약을 체결했다.",
        "published_at": "2026-06-02T08:20:00+00:00",
    }
    market = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["deal"],
        "title": "[특징주] 인스웨이브, LG CNS와 98억 규모 웹단말 공급계약",
        "content": "인스웨이브 주가가 LG CNS 공급계약 소식에 급등했다.",
        "published_at": "2026-06-02T06:34:00+00:00",
    }

    assert _event_bucket(contract) == "contract_deal"
    assert _event_bucket(market) == "market_reaction"
    assert _rule_prefilter_key(contract) == _rule_prefilter_key(market)
    assert _should_merge_articles(contract, market, 0.95, 0.80) is True


def test_contract_key_groups_same_deal_with_amount_or_service_title(monkeypatch):
    amount_title = {
        "company": ["hyundai_autoever"],
        "matched_companies": ["hyundai_autoever"],
        "matched_sectors": ["deal"],
        "title": "플래티어, 현대오토에버와 21억 6천만 원대 공급계약",
        "content": "플래티어가 현대오토에버와 인증중고차 플랫폼 운영 계약을 체결했다.",
        "published_at": "2026-06-01T05:34:00+00:00",
    }
    service_title = {
        "company": ["hyundai_autoever"],
        "matched_companies": ["hyundai_autoever"],
        "matched_sectors": ["deal"],
        "title": "플래티어, 현대오토에버와 CPO 플랫폼 운영 계약",
        "content": "인증중고차 플랫폼 운영 사업을 장기 운영한다.",
        "published_at": "2026-06-01T07:34:00+00:00",
    }

    assert _rule_prefilter_key(amount_title) == _rule_prefilter_key(service_title)
    assert _should_merge_articles(amount_title, service_title, 0.82, 0.80) is True


def test_logistics_robotics_partner_articles_can_merge_with_llm(monkeypatch):
    first = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["infra", "deal"],
        "title": "LG CNS-컬리, 물류센터 휴머노이드 PoC…'피지컬웍스' 현장 검증",
        "content": "LG CNS와 컬리가 물류센터에 휴머노이드 로봇을 도입하고 자동화 실증을 추진한다.",
        "published_at": "2026-05-18T06:40:00+00:00",
    }
    second = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["infra", "deal"],
        "title": "컬리 물류센터에 휴머노이드 뜬다…LG CNS와 자동화 협력",
        "content": "LG CNS와 컬리가 휴머노이드 로봇 기반 물류 자동화 협력에 나섰다.",
        "published_at": "2026-05-18T01:01:00+00:00",
    }

    assert _rule_prefilter_key(first) == _rule_prefilter_key(second)
    assert _should_merge_articles(first, second, 0.80, 0.80) is True


def test_title_similarity_rule_can_merge_across_company_and_sector_labels(monkeypatch):
    left = {
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["infra"],
        "title": "LG CNS·LX판토스, 물류센터 자동화 로봇 도입 협력",
        "content": "",
        "published_at": "2026-06-11T01:10:00+00:00",
    }
    right = {
        "company": ["lx_pantos"],
        "matched_companies": ["lx_pantos"],
        "matched_sectors": ["logistics"],
        "title": "LX판토스 물류 거점에 자동화 로봇 배치…LG CNS와 실증",
        "content": "",
        "published_at": "2026-06-11T01:20:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.62, 0.80) is True


def test_prefilter_groups_same_day_event_without_company_split(monkeypatch):
    left = {
        "company": ["nvidia"],
        "matched_companies": ["nvidia"],
        "title": "NC AI, 한화오션 자율 용접 로봇 AI 모델 개발",
        "content": "NC AI와 한화오션의 자율 용접 로봇 AI 모델 개발 기사.",
        "published_at": "2026-06-04T01:42:00+00:00",
    }
    right = {
        "company": ["nc_ai"],
        "matched_companies": ["nc_ai"],
        "title": "NC AI, 한화오션 상선에 자율용접로봇 AI 두뇌 공급",
        "content": "한화오션 선박에 자율용접로봇 AI 두뇌를 공급한다.",
        "published_at": "2026-06-04T01:30:00+00:00",
    }

    assert _rule_prefilter_key(left) == _rule_prefilter_key(right)
    assert _should_merge_articles(left, right, 0.82, 0.80) is True


def test_prefilter_allows_same_ax_event_across_adjacent_dates(monkeypatch):
    may_article = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "포스코DX, NC AI와 ‘산업 현장 로봇’ 지능화 모델 개발",
        "content": "포스코DX와 NC AI가 로봇 파운데이션 모델을 공동 개발한다.",
        "published_at": "2026-05-31T04:01:00+00:00",
    }
    june_article = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "NC AI, 포스코DX와 로봇 파운데이션 모델 공동 개발 업무협약 체결",
        "content": "NC AI와 포스코DX가 산업현장용 피지컬AI 기반 로봇 지능화 기술을 공동 개발한다.",
        "published_at": "2026-06-01T09:00:00+00:00",
    }

    assert _rule_prefilter_key(may_article) == _rule_prefilter_key(june_article)
    assert _should_merge_articles(may_article, june_article, 0.82, 0.80) is True


def test_prefilter_merges_same_ax_event_with_loose_title_with_llm(monkeypatch):
    loose_title = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "'로봇의 두뇌' 만든다…NC AI·포스코DX 협력 선언",
        "content": "NC AI와 포스코DX가 로봇 파운데이션 모델 공동 개발에 협력한다.",
        "published_at": "2026-05-31T08:22:00+00:00",
    }
    foundation_model = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "NC AI, 포스코DX와 로봇 파운데이션 모델 공동 개발 업무협약 체결",
        "content": "NC AI와 포스코DX가 산업현장용 피지컬AI 기반 로봇 지능화 기술을 공동 개발한다.",
        "published_at": "2026-06-01T09:00:00+00:00",
    }

    assert _rule_prefilter_key(loose_title) == _rule_prefilter_key(foundation_model)
    assert _should_merge_articles(loose_title, foundation_model, 0.82, 0.80) is True


def test_existing_cluster_candidate_window_uses_article_published_dates():
    window = _existing_cluster_candidate_window(
        [
            {"published_at": "2026-05-18T01:00:00+00:00"},
            {"published_at": "2026-05-18T23:20:00+00:00"},
        ]
    )

    assert window is not None
    assert window[0].startswith("2026-05-13T01:00:00+00:00")
    assert window[1].startswith("2026-05-24T23:20:00+00:00")


def test_prefilter_allows_same_cloud_infra_event_across_adjacent_dates():
    center_award = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "국가 AI컴퓨팅센터, 삼성SDS가 만든다... 올해 3분기 착공",
        "content": "GPU 1.5만장 규모 국가 AI컴퓨팅센터 구축 사업을 삼성SDS가 맡는다.",
        "published_at": "2026-05-11T08:33:00+00:00",
    }
    center_context = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "'AI 고속도로 심장부' 국가AI컴퓨팅센터, 삼성SDS가 맡는다",
        "content": "삼성SDS가 국가 AI 컴퓨팅 인프라 구축을 맡는다는 내용이다.",
        "published_at": "2026-05-11T08:01:00+00:00",
    }

    assert _rule_prefilter_key(center_award) == _rule_prefilter_key(center_context)
    assert _should_merge_articles(center_award, center_context, 0.82, 0.80) is True


def test_prefilter_groups_ai_center_variants_before_similarity():
    infrastructure = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "2조5천억원 'AI 고속도로' 시동…정부·삼성SDS 국가 인프라 구축",
        "content": "삼성SDS가 국가 AI컴퓨팅센터 구축을 맡는다는 내용이다.",
        "published_at": "2026-05-11T09:06:00+00:00",
    }
    center = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "국가 AI컴퓨팅센터, 삼성SDS가 만든다... 올해 3분기 착공",
        "content": "GPU 1.5만장 규모 국가 AI컴퓨팅센터 구축 사업을 삼성SDS가 맡는다.",
        "published_at": "2026-05-11T08:33:00+00:00",
    }

    assert _rule_prefilter_key(infrastructure) == _rule_prefilter_key(center)
    assert _should_merge_articles(infrastructure, center, 0.82, 0.80) is True


def test_prefilter_allows_same_contract_event_across_adjacent_dates(monkeypatch):
    sto_award = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 예탁결제원 STO 플랫폼 구축 사업 수주",
        "content": "삼성SDS가 예탁결제원 토큰증권 플랫폼 구축 사업을 수주했다.",
        "published_at": "2026-05-06T00:53:00+00:00",
    }
    sto_context = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 예탁원 ‘토큰증권 플랫폼’ 수주…디지털 금융 표준 세운다",
        "content": "예탁원 토큰증권 플랫폼 구축 사업을 삼성SDS가 맡는다.",
        "published_at": "2026-05-05T23:34:00+00:00",
    }

    assert _rule_prefilter_key(sto_award) == _rule_prefilter_key(sto_context)
    assert _should_merge_articles(sto_award, sto_context, 0.82, 0.80) is True


def test_prefilter_groups_contract_title_without_company_subject(monkeypatch):
    sto_award = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 예탁결제원 STO 플랫폼 구축 사업 수주",
        "content": "삼성SDS가 예탁결제원 토큰증권 플랫폼 구축 사업을 수주했다.",
        "published_at": "2026-05-06T00:53:00+00:00",
    }
    no_company_subject = {
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "예탁원 토큰증권 플랫폼, 삼성SDS가 구축한다",
        "content": "예탁원 토큰증권 플랫폼 구축 사업을 삼성SDS가 맡는다.",
        "published_at": "2026-05-06T01:19:00+00:00",
    }

    assert _rule_prefilter_key(sto_award) == _rule_prefilter_key(no_company_subject)
    assert _should_merge_articles(sto_award, no_company_subject, 0.82, 0.80) is True


def test_robot_partnership_uses_ax_bucket_before_contract_bucket(monkeypatch):
    agreement = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "NC AI, 포스코DX와 로봇 파운데이션 모델 공동 개발 업무협약 체결",
        "content": "NC AI와 포스코DX가 산업현장용 피지컬AI 기반 로봇 지능화 기술을 공동 개발한다.",
        "published_at": "2026-06-01T09:00:00+00:00",
    }
    development = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "포스코DX-NC AI, '피지컬AI 기반 로봇 지능화 기술' 공동개발",
        "content": "포스코DX와 NC AI가 로봇 파운데이션 모델을 공동 개발한다.",
        "published_at": "2026-06-01T05:46:00+00:00",
    }

    assert _event_bucket(agreement) == "ax_strategy"
    assert _rule_prefilter_key(agreement) == _rule_prefilter_key(development)
    assert _should_merge_articles(agreement, development, 0.82, 0.80) is True


def test_nc_physical_ai_subissues_do_not_collapse_into_one_cluster():
    posco_robot = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "NC AI, 포스코DX와 로봇 파운데이션 모델 공동 개발 업무협약 체결",
        "content": "NC AI와 포스코DX가 산업현장용 피지컬AI 기반 로봇 지능화 기술을 공동 개발한다.",
        "published_at": "2026-06-01T09:00:00+00:00",
    }
    hanwha_welding = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "NC AI, 한화오션 자율용접 로봇 AI 두뇌 개발…피지컬AI 영토 확장",
        "content": "NC AI가 한화오션 자율용접 로봇 AI 두뇌를 개발한다.",
        "published_at": "2026-06-04T01:42:00+00:00",
    }
    jensen_meeting = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "title": "젠슨 황, 엔씨 김택진 대표 만난다…피지컬 AI 협력 논의 가능성",
        "content": "젠슨 황과 김택진 대표가 피지컬 AI 협력 가능성을 논의한다.",
        "published_at": "2026-06-02T08:18:00+00:00",
    }

    assert _rule_prefilter_key(posco_robot) == _rule_prefilter_key(hanwha_welding)
    assert _should_merge_articles(posco_robot, hanwha_welding, 0.99, 0.80) is False
    assert _should_merge_articles(posco_robot, jensen_meeting, 0.99, 0.80) is False


def test_same_company_ax_event_variants_merge_by_generic_content_features():
    agent_push = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax"],
        "title": "포스코DX, ‘1인 N에이전트’ 시대 대비…전사 AX 혁신 가속",
        "content": (
            "전사 AX 혁신 프로그램의 일환으로 임직원 대상 해커톤을 열고 "
            "업무별 AI 에이전트 활용 역량을 강화했다."
        ),
        "published_at": "2026-06-18T00:38:00+00:00",
    }
    event_angle = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax"],
        "title": "포스코DX, ‘AX 해커톤’ 개최해 AI 역량 강화",
        "content": (
            "전사 AX 혁신 프로그램에서 임직원이 업무별 AI 에이전트 활용 과제를 "
            "해커톤 형태로 발표했다."
        ),
        "published_at": "2026-06-18T02:06:00+00:00",
    }
    broad_title = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax"],
        "title": "포스코DX, '1인 N에이전트 시대' 연다…전사 AI 역량 강화",
        "content": (
            "임직원 해커톤과 업무별 AI 에이전트 활용을 통해 전사 AX 혁신을 확대하는 프로그램이다."
        ),
        "published_at": "2026-06-18T00:29:00+00:00",
    }

    assert _should_merge_articles(agent_push, event_angle, 0.74, 0.80) is True
    assert _should_merge_articles(agent_push, broad_title, 0.74, 0.80) is True
    assert _should_merge_articles(event_angle, broad_title, 0.74, 0.80) is True


def test_same_company_generic_ax_theme_does_not_merge_when_object_features_differ():
    event_program = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax"],
        "title": "포스코DX, 전사 AX 역량 강화 프로그램 확대",
        "content": "임직원 대상 업무 자동화 교육과 실습 프로그램을 확대했다.",
        "published_at": "2026-06-18T00:38:00+00:00",
    }
    product_release = {
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax"],
        "title": "포스코DX, 제조 현장 데이터 분석 플랫폼 공개",
        "content": "제조 현장의 품질 데이터를 분석하는 신규 플랫폼을 공개했다.",
        "published_at": "2026-06-18T02:06:00+00:00",
    }

    assert _should_merge_articles(event_program, product_release, 0.90, 0.80) is False


def test_core_role_rejects_alumni_personnel_article():
    result = _noise_reject_result(
        title="중고나라 LG CNS 출신 CTO 선임, AI로 '사기 거래와의 전쟁' 나선다",
        content="중고나라가 LG CNS 출신 CTO를 선임하고 AI 전환을 추진한다.",
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )
    if result is None:
        from src.preprocessing.relevance import _core_company_role_reject_result

        result = _core_company_role_reject_result(
            title="중고나라 LG CNS 출신 CTO 선임, AI로 '사기 거래와의 전쟁' 나선다",
            content="중고나라가 LG CNS 출신 CTO를 선임하고 AI 전환을 추진한다.",
            source_type="news",
            matched_companies=["lg_cns"],
            matched_sectors=["ax"],
        )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_fast_passes_company_market_expansion_title():
    title = "LG CNS, 북미 제조 AX 시장 공략 본격화 … 중소·중견 제조기업 공장 지능화 지원"
    content = "LG CNS가 북미 제조 AX 시장을 확대하고 AI 스마트팩토리 솔루션을 지원한다."

    fast_pass = _fast_pass_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )
    reject = _core_company_role_reject_result(
        title=title,
        content=content,
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )

    assert fast_pass is not None
    assert fast_pass["relevance_label"] == "relevant"
    assert reject is None


def test_dedup_merges_company_manufacturing_ax_market_expansion(monkeypatch):
    left = {
        "id": 38657,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS, 북미 제조 AX 시장 공략 본격화 … 중소·중견 제조기업 공장 지능화 지원",
        "content": "LG CNS가 북미 제조 AX 시장 공략을 확대하고 공장 지능화를 지원한다.",
        "published_at": "2026-05-20T08:52:00+00:00",
    }
    right = {
        "id": 38665,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS, AI 스마트팩토리 앞세워 북미 제조 AX 시장 정조준",
        "content": "LG CNS가 AI 스마트팩토리 솔루션으로 북미 제조 AX 시장 확대에 나선다.",
        "published_at": "2026-05-20T08:06:00+00:00",
    }

    assert _event_bucket(left) == "ax_strategy"
    assert _event_bucket(right) == "ax_strategy"
    assert _should_merge_articles(left, right, 0.82, 0.80) is True


def test_cluster_keeps_manufacturing_ax_market_articles_together_after_split(monkeypatch):
    articles = [
        {
            "id": 38657,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "matched_sectors": ["ax"],
            "title": "LG CNS, 북미 제조 AX 시장 공략 본격화 … 중소·중견 제조기업 공장 지능화 지원",
            "content": "LG CNS가 북미 제조 AX 시장 공략을 확대하고 공장 지능화를 지원한다.",
            "published_at": "2026-05-20T08:52:00+00:00",
        },
        {
            "id": 38664,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "matched_sectors": ["ax"],
            "title": "LG CNS, 북미 제조 AX 시장 확대…중소·중견 제조기업 공장 지능화 지원",
            "content": "LG CNS가 제조기업 대상 AI 스마트팩토리 솔루션을 확대한다.",
            "published_at": "2026-05-20T08:12:00+00:00",
        },
        {
            "id": 38665,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "matched_sectors": ["ax"],
            "title": "LG CNS, AI 스마트팩토리 앞세워 북미 제조 AX 시장 정조준",
            "content": "LG CNS가 AI 스마트팩토리 솔루션으로 북미 제조 AX 시장 확대에 나선다.",
            "published_at": "2026-05-20T08:06:00+00:00",
        },
        {
            "id": 38564,
            "company": ["lg_cns"],
            "matched_companies": ["lg_cns"],
            "matched_sectors": ["ax"],
            "title": "LG CNS, 제조특화 플랫폼 선봬…북미 제조 RX 공략 속도",
            "content": "LG CNS가 제조특화 플랫폼으로 북미 제조 시장 공략을 강화한다.",
            "published_at": "2026-05-20T07:16:00+00:00",
        },
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.97, 0.03, 0.0],
        ],
        dtype=np.float32,
    )

    cluster_map = _cluster(articles, embeddings, threshold=0.80)

    assert list(cluster_map.values()) == [[38657, 38664, 38665, 38564]]


def test_contract_deal_does_not_mix_logistics_robotics_and_smart_infra_lidar():
    logistics = {
        "id": 14918,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS-컬리, 물류센터 휴머노이드 로봇 도입 맞손",
        "content": "LG CNS와 컬리가 물류센터 휴머노이드 로봇 실증을 추진한다.",
        "published_at": "2026-05-18T01:02:00+00:00",
    }
    smart_infra = {
        "id": 29875,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["infra"],
        "title": "에스오에스랩, LG CNS와 협약…북미 스마트 인프라 시장 공략 가속",
        "content": "에스오에스랩과 LG CNS가 라이다 기반 북미 스마트 인프라 시장 공략에 나선다.",
        "published_at": "2026-05-19T02:22:00+00:00",
    }
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32)

    assert _should_merge_articles(logistics, smart_infra, 0.99, 0.80) is False
    assert list(_cluster([logistics, smart_infra], embeddings, threshold=0.80).values()) == [
        [14918],
        [29875],
    ]


def test_openai_enterprise_ai_articles_can_merge_with_llm(monkeypatch):
    left = {
        "id": 16029,
        "company": ["sk_ax"],
        "matched_companies": ["sk_ax"],
        "matched_sectors": ["ax"],
        "title": "SK AX, 오픈AI와 협력…엔터프라이즈 AI 사업 확대",
        "content": "SK AX가 오픈AI와 챗GPT 엔터프라이즈 기반 기업용 AI 사업을 확대한다.",
        "published_at": "2026-05-14T02:22:00+00:00",
    }
    right = {
        "id": 16017,
        "company": ["sk_ax"],
        "matched_companies": ["sk_ax"],
        "matched_sectors": ["ax"],
        "title": "SK AX, 오픈AI와 손잡고 기업용 AI 시장 공략",
        "content": "SK AX가 OpenAI와 협력해 기업용 생성형 AI 시장을 공략한다.",
        "published_at": "2026-05-14T00:07:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.84, 0.80) is True


def test_same_company_partner_product_articles_merge_across_contract_and_ax_buckets():
    existing = {
        "id": 46656,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS·앤트로픽 맞손…클로드 기반 AX 시장 공략",
        "content": "",
        "published_at": "2026-06-09T01:00:00+00:00",
    }
    direct_product = {
        "id": 47126,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS, 앤트로픽 ‘클로드 엔터프라이즈’ 도입",
        "content": "",
        "published_at": "2026-06-09T09:38:00+00:00",
    }
    partner_action = {
        "id": 47125,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS, 오픈AI 이어 앤트로픽과 맞손…“기업 AX 사업 확대”",
        "content": "",
        "published_at": "2026-06-09T09:50:00+00:00",
    }

    assert _should_merge_articles(direct_product, existing, 0.70, 0.80) is True
    assert _should_merge_articles(partner_action, existing, 0.70, 0.80) is True


def test_same_day_action_prefilter_lets_llm_review_cross_bucket_security_news(monkeypatch):
    cloud_title = {
        "id": 47168,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, AI 해커·클라우드 보안 스타트업 손잡았다",
        "published_at": "2026-06-10T08:24:00+09:00",
    }
    broad_partner_title = {
        "id": 47166,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS, 국내외 보안기업과 협력 확대…AI 보안 역량 강화",
        "published_at": "2026-06-10T08:44:00+09:00",
    }
    named_partner_title = {
        "id": 47169,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": "삼성SDS 엑스보우·테이텀 시큐리티 손잡고 AI 보안 삼각편대 구축",
        "published_at": "2026-06-10T08:12:00+09:00",
    }
    capability_title = {
        "id": 47212,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "title": '"취약점 탐지부터 사고 복구까지"…삼성SDS, AI 보안체계 강화',
        "published_at": "2026-06-10T09:02:00+09:00",
    }

    assert _event_bucket(cloud_title) == "cloud_infra"
    assert _event_bucket(broad_partner_title) == "contract_deal"
    assert _event_bucket(named_partner_title) == "security"
    assert _event_bucket(capability_title) == "security"
    assert _rule_prefilter_key(cloud_title) == _rule_prefilter_key(broad_partner_title)
    assert _rule_prefilter_key(cloud_title) == _rule_prefilter_key(named_partner_title)
    assert _rule_prefilter_key(capability_title) != _rule_prefilter_key(named_partner_title)

    assert _should_merge_articles(cloud_title, broad_partner_title, 0.82, 0.80) is True
    assert _should_merge_articles(cloud_title, named_partner_title, 0.82, 0.80) is True
    assert _should_merge_articles(named_partner_title, capability_title, 0.82, 0.80) is False


def test_same_company_title_fallback_merges_skala_training_variants():
    left = {
        "id": 1724,
        "company": ["sk_ax"],
        "matched_companies": ["sk_ax"],
        "matched_sectors": ["ax"],
        "title": "[산업소식] SK AX, 채용 연계형 AI 교육 프로그램 SKALA 4기 모집",
        "content": "",
        "published_at": "2026-05-07T06:00:00+00:00",
    }
    right = {
        "id": 1731,
        "company": ["sk_ax"],
        "matched_companies": ["sk_ax"],
        "matched_sectors": ["ax"],
        "title": "SK AX, AI교육 ‘스칼라’ 광주·울산으로 확대",
        "content": "",
        "published_at": "2026-05-07T05:10:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.70, 0.80) is False


def test_physicalworks_rx_platform_articles_can_merge_with_llm(monkeypatch):
    left = {
        "id": 1325,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "LG CNS, 로봇 학습·운영 플랫폼 '피지컬웍스' 공개",
        "content": "LG CNS가 RX 플랫폼 피지컬웍스를 공개했다.",
        "published_at": "2026-05-07T01:00:00+00:00",
    }
    right = {
        "id": 1450,
        "company": ["lg_cns"],
        "matched_companies": ["lg_cns"],
        "matched_sectors": ["ax"],
        "title": "로봇 현장 안착 수개월→1~2개월로 단축..LG CNS, '피지컬웍스' 공개",
        "content": "LG CNS가 로봇 학습과 운영을 통합하는 RX 플랫폼을 공개했다.",
        "published_at": "2026-05-07T10:16:00+00:00",
    }

    assert _should_merge_articles(left, right, 0.83, 0.80) is True


def test_nc_posco_robot_ai_title_variants_can_merge_with_llm(monkeypatch):
    base = {
        "id": 44134,
        "company": ["posco_dx"],
        "matched_companies": ["posco_dx"],
        "matched_sectors": ["ax", "deal"],
        "title": "NC AI-포스코DX, 로봇 AI 브레인 개발 협력…범용 로봇 지능화 기술 공동 개발",
        "content": "",
        "published_at": "2026-05-31T23:30:00+00:00",
    }
    variants = [
        "NC AI·포스코DX, AI 로봇 개발 맞손",
        "포스코DX-NC AI, '피지컬AI 지능화' 기술 공동 개발",
        "포스코DX-NC AI, 로봇 AI 개발 협력… 지능화 기술 연구",
        "‘리니지’ 기술로 로봇 학습한다…NC AI-포스코DX ‘피지컬 AI’ 맞손",
        "NC AI-포스코DX, 로봇 AI브레인 공동 개발 맞손",
        "포스코DX·NC AI 산업용 로봇 제어 모델 개발 추진",
        "NC AI, 포스코DX와 로봇 지능 개발 나서",
    ]

    for idx, title in enumerate(variants, start=1):
        article = {**base, "id": 44134 + idx, "title": title}
        assert _should_merge_articles(base, article, 0.80, 0.80) is True


def test_national_ai_computing_center_uses_generic_signature():
    article = {
        "id": 933,
        "company": ["samsung_sds"],
        "matched_companies": ["samsung_sds"],
        "matched_sectors": ["infra"],
        "title": "국가 AI컴퓨팅센터 민간참여자로 '삼성SDS 컨소시엄' 확정",
        "content": "삼성SDS 컨소시엄이 국가 AI 컴퓨팅 센터 사업자로 최종 확정됐다.",
        "published_at": "2026-05-12T04:30:00+00:00",
    }

    assert _event_signature(article).startswith(("cloud_infra:", "general:"))


def test_industry_trend_ai_competition_titles_share_signal_key():
    left = {
        "id": 51001,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "[경제 브리핑] AI 경쟁 승부처, 인프라·데이터·제도 기반이 가른다",
        "content": "",
        "published_at": "2026-06-16T11:34:00+09:00",
    }
    right = {
        "id": 51002,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "\"AI 경쟁, '모델 성능' 넘어 '전력 인프라·데이터·제도'로 전선 확장\"",
        "content": "",
        "published_at": "2026-06-16T10:50:00+09:00",
    }

    assert _event_signature(left) == "industry:ai_competition_infra_data_policy"
    assert _event_signature(left) == _event_signature(right)
    assert _should_merge_articles(left, right, 0.62, 0.80) is True


def test_industry_trend_6g_security_standard_titles_share_signal_key():
    left = {
        "id": 51003,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "韓 AI·6G 보안 국제표준 개발 착수···ITU-T서 신규과제 14건 승인",
        "content": "",
        "published_at": "2026-06-14T14:19:00+09:00",
    }
    right = {
        "id": 51004,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "과기정통부, AI·6G 보안 국제표준 개발 착수",
        "content": "",
        "published_at": "2026-06-12T16:34:00+09:00",
    }

    assert _event_signature(left) == "industry:ai_6g_security_standard"
    assert _event_signature(left) == _event_signature(right)
    assert _should_merge_articles(left, right, 0.62, 0.80) is True


def test_industry_trend_supply_chain_threat_titles_share_signal_key():
    left = {
        "id": 51007,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "협력사 통해 뚫린다… 공급망 공격, 사이버 위협 1위",
        "content": "",
        "published_at": "2026-06-16T12:10:00+09:00",
    }
    right = {
        "id": 51008,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": '"작년 기업 31% 공급망 위협 경험 ... 위협으로 꼽은 기업 9% 불과"',
        "content": "",
        "published_at": "2026-06-16T11:26:00+09:00",
    }

    assert _event_signature(left) == "industry:supply_chain_attack_risk"
    assert _event_signature(left) == _event_signature(right)
    assert _should_merge_articles(left, right, 0.62, 0.80) is True


def test_industry_trend_signal_key_prefers_title_over_broad_body_context():
    article = {
        "id": 51010,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "“믿었던 거래처가 구멍”…기업 3곳 중 1곳, 공급망 공격 당했다",
        "content": "본문 배경에는 AI 경쟁이 전력·데이터·제도 싸움으로 확장된다는 설명도 포함됐다.",
        "published_at": "2026-06-16T09:42:00+09:00",
    }

    assert _event_signature(article) == "industry:supply_chain_attack_risk"


def test_industry_trend_keeps_different_signal_topics_separate():
    ai_infra = {
        "id": 51005,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "AI 승부처 바뀌었다 “모델 성능 아닌 전력·법·데이터 싸움”",
        "content": "",
        "published_at": "2026-06-16T09:24:00+09:00",
    }
    security_standard = {
        "id": 51006,
        "company": ["industry_trend"],
        "source_name": "naver_industry_news",
        "title": "과기정통부, AI·6G 보안 국제표준 개발 착수",
        "content": "",
        "published_at": "2026-06-12T16:34:00+09:00",
    }

    assert _event_signature(ai_infra) != _event_signature(security_standard)
    assert _should_merge_articles(ai_infra, security_standard, 0.95, 0.80) is False


def test_relevance_rejects_financial_theme_without_peer_in_title():
    result = _noise_reject_result(
        title="증권사들, 코인원·두나무·코빗 찍었다..거래소 지분 확보",
        content="금융권이 가상자산 거래소 지분 확보에 나섰다.",
        source_type="news",
        matched_companies=["samsung_sds"],
        matched_sectors=["infra", "deal"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_rejects_vague_stock_reaction_title():
    result = _noise_reject_result(
        title="LG씨엔에스 주가, 급등세... 왜?",
        content="LG씨엔에스 주가가 급등세를 보이고 있다.",
        source_type="news",
        matched_companies=["lg_cns"],
        matched_sectors=["ax"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_rejects_group_ipo_context_without_peer_in_title():
    result = _noise_reject_result(
        title="현대차, '보스턴다이나믹스 카드' 꺼낸다···IPO로 지배구조 개편 실탄",
        content="현대차그룹이 로보틱스 IPO를 검토한다. 현대오토에버는 관련 계열사로 언급됐다.",
        source_type="news",
        matched_companies=["hyundai_autoever"],
        matched_sectors=["infra", "deal"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"


def test_relevance_rejects_nc_stock_promotion_context_without_peer_in_title():
    result = _noise_reject_result(
        title="NC AI, 'RFM 레퍼런스'도 없는데 잇단 피지컬AI 협력 채찍질…주가부양",
        content="NC AI의 피지컬AI 행보를 다루며 과거 포스코DX 협력을 배경으로 언급했다.",
        source_type="news",
        matched_companies=["posco_dx"],
        matched_sectors=["ax", "deal"],
    )

    assert result is not None
    assert result["relevance_label"] == "irrelevant"
