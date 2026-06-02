"""파이프라인 단위 테스트"""

import numpy as np

from src.preprocessing import dedup
from src.preprocessing.dedup import (
    _cluster,
    _company_presence_score,
    _event_signature,
    _same_issue,
    _should_merge_articles,
)
from src.preprocessing.preprocessing import PreprocessingResult


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
