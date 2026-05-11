"""파이프라인 단위 테스트"""

from src.agents.dedup_agent import _company_presence_score, _same_issue
from src.pipeline.ingestion_graph import IngestionState


def test_ingestion_state_structure():
    state: IngestionState = {
        "company": ["samsung_sds"],
        "trigger_type": "scheduled",
        "raw_article_ids": [],
        "credible_ids": [],
        "cluster_map": {},
        "representative_ids": [],
        "classified_clusters": [],
        "issue_cards": [],
        "implications": [],
        "validation_results": [],
        "errors": [],
        "human_review_flags": [],
    }
    assert state["company"] == ["samsung_sds"]


def test_same_issue_groups_company_customer_business_overlap():
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
