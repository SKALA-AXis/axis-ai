"""파이프라인 단위 테스트"""

from src.preprocessing.dedup import _company_presence_score, _same_issue
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
