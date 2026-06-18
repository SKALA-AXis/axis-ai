# 작성일: 2026-06-15
# 작성자: 최종민
# 변경이력:
#   2026-06-15 최종민 — 단일 기사 텍스트 분류 엔드포인트 추가, 합성 article 에 source_name 보정
#   2026-06-18 박지원 — 전처리 LLM 폴백 제거
"""classify_article_text — 단일 기사 텍스트 분류 (운영 로직 재사용) 단위 테스트."""

from __future__ import annotations

from src.preprocessing.classification import classify_article_text


def test_contract_article_classified_as_contract():
    r = classify_article_text(
        "LG CNS, 삼송 데이터센터에서 약 1조 원 규모 사업 수주",
        "LG CNS가 약 1조 원 규모의 데이터센터 구축 사업을 수주했다.",
        company="lg_cns",
    )
    assert r["event_type"] == "contract"
    # 대형 수주 → impact 보정으로 importance 가 게이트 임계(0.65) 이상.
    assert r["importance_score"] >= 0.65


def test_ma_article_classified_as_ma():
    r = classify_article_text(
        "삼성SDS, 두나무 지분 4% 인수로 디지털 자산 생태계 확장",
        "삼성SDS가 두나무 지분 4%를 인수했다.",
        company="samsung_sds",
    )
    assert r["event_type"] == "ma"
    assert r["importance_score"] >= 0.65


def test_partnership_article_classified_as_partnership():
    r = classify_article_text(
        "포스코DX와 NC AI, AI 로봇 자율작업 업무협약 체결",
        "포스코DX가 NC AI와 업무협약을 체결했다.",
        company="posco_dx",
    )
    assert r["event_type"] == "partnership"


def test_personnel_article_not_in_alert_set():
    r = classify_article_text(
        "삼성SDS, 신임 임원 인사 단행",
        "삼성SDS가 임원 인사와 조직 개편을 단행했다.",
        company="samsung_sds",
    )
    # 인사 = 대형 이벤트 게이트 대상 아님 → 알림 안 감을 시연하는 케이스.
    assert r["event_type"] not in {"ma", "contract", "partnership"}


def test_result_shape_has_required_fields():
    r = classify_article_text("테스트 제목", "본문", company="lg_cns")
    for key in (
        "event_type",
        "sector",
        "sectors",
        "exposure_score",
        "impact_score",
        "importance_score",
        "importance",
        "signals",
    ):
        assert key in r
    assert isinstance(r["sectors"], list)
    assert r["signals"]["cluster_size"] == 1


def test_rule_miss_uses_company_default(monkeypatch):
    """규칙 미매칭은 LLM fallback 없이 company 기본값을 반환한다."""
    import src.preprocessing.classification as clf

    monkeypatch.setattr(clf, "_classify_event_type_rule_based", lambda **kw: (None, ""))

    r = clf.classify_article_text(
        "삼성SDS 관련 동향",
        "본문 내용입니다.",
        company="samsung_sds",
    )
    assert r["event_type"] == "company"
    assert r["reasoning"] == "규칙 매칭 없음, company 기본값"


def test_classifier_does_not_expose_llm_constructor(monkeypatch):
    import src.preprocessing.classification as clf

    monkeypatch.setattr(clf, "_classify_event_type_rule_based", lambda **kw: (None, ""))

    assert not hasattr(clf, "_get_llm")
    r = clf.classify_article_text("삼성SDS 관련 동향", "본문 내용입니다.", company="samsung_sds")
    assert r["event_type"] == "company"
