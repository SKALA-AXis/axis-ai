"""제목 키 문장부호 정규화 회귀 테스트.

2026-06-11 클러스터 DB 검증에서 동일 제목이 따옴표 문자(' vs ')만 달라
다른 클러스터로 갈라진 사례 11쌍 발견 — _fallback_dedup_key 가
유니코드 문장부호를 정규화하지 않던 것이 원인.
"""

import os

os.environ.setdefault("DEDUP_CLUSTER_LLM_JUDGE_ENABLED", "false")

from src.preprocessing.dedup import _fallback_dedup_key, _normalize_title_key


def test_curly_and_straight_quotes_produce_same_key():
    # 실제 DB에서 갈라졌던 쌍 (LG CNS 챗GPT 에듀 — sim=1.00, 다른 cluster_id)
    straight = {"id": 1, "title": "LG CNS, '챗GPT 에듀' 리셀러 파트너 선정"}
    curly = {"id": 2, "title": "LG CNS, ‘챗GPT 에듀’ 리셀러 파트너 선정"}
    assert _fallback_dedup_key(straight) == _fallback_dedup_key(curly)


def test_double_quote_variants_produce_same_key():
    a = {"id": 1, "title": '삼성SDS, "양자내성암호" 국내 기술 표준 이끈다'}
    b = {"id": 2, "title": "삼성SDS, “양자내성암호” 국내 기술 표준 이끈다"}
    c = {"id": 3, "title": "삼성SDS, 「양자내성암호」 국내 기술 표준 이끈다"}
    keys = {_fallback_dedup_key(a), _fallback_dedup_key(b), _fallback_dedup_key(c)}
    assert len(keys) == 1


def test_middle_dot_and_dash_variants_produce_same_key():
    a = {"id": 1, "title": "포스코DX, AI·로봇 사업 확대 — 스마트팩토리"}
    b = {"id": 2, "title": "포스코DX, AI‧로봇 사업 확대 – 스마트팩토리"}
    assert _fallback_dedup_key(a) == _fallback_dedup_key(b)


def test_different_titles_keep_different_keys():
    a = {"id": 1, "title": "SK AX, 김완종 CCO 사장 승진 선임"}
    b = {"id": 2, "title": "SK브로드밴드 김성수 CEO 선임"}
    assert _fallback_dedup_key(a) != _fallback_dedup_key(b)


def test_normalize_title_key_collapses_whitespace_and_case():
    assert _normalize_title_key("LG  CNS,   AWS와\t협력") == _normalize_title_key("lg cns, aws와 협력")


def test_empty_title_falls_back_to_url_hash():
    article = {"id": 7, "title": "", "url_hash": "abc123"}
    assert _fallback_dedup_key(article) == "abc123"


def test_regression_same_group_different_company_personnel_not_merged():
    """과거 데이터에서 SK브로드밴드 CEO 선임이 SK AX 사장 승진 클러스터(26108)에
    혼입됐던 사례 — #119 개편 이후 결정적 경로(LLM off)가 병합을 거부해야 한다."""
    from src.preprocessing.dedup import _should_merge_articles

    left = {
        "id": 3,
        "title": "SK AX, 김완종 CCO 사장 승진 선임",
        "company": "sk_ax",
        "published_at": "2025-10-30",
        "content": "SK AX는 김완종 최고고객책임자를 사장으로 승진 선임했다",
    }
    right = {
        "id": 4,
        "title": "SK브로드밴드 김성수 CEO 선임",
        "company": "sk_ax",
        "published_at": "2025-10-30",
        "content": "SK브로드밴드가 김성수 신임 CEO를 선임했다",
    }
    assert _should_merge_articles(left, right, 0.85, 0.80) is False
    assert _should_merge_articles(left, right, 0.95, 0.80) is False
