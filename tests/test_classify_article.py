"""classify_article_text — 단일 기사 텍스트 분류 (운영 로직 재사용) 단위 테스트.

enable_llm=False 로 규칙 기반만 사용해 결정적으로 검증한다 (OpenAI 미호출).
"""

from __future__ import annotations

from src.preprocessing.classification import classify_article_text


def test_contract_article_classified_as_contract():
    r = classify_article_text(
        "LG CNS, 삼송 데이터센터에서 약 1조 원 규모 사업 수주",
        "LG CNS가 약 1조 원 규모의 데이터센터 구축 사업을 수주했다.",
        company="lg_cns",
        enable_llm=False,
    )
    assert r["event_type"] == "contract"
    # 대형 수주 → impact 보정으로 importance 가 게이트 임계(0.65) 이상.
    assert r["importance_score"] >= 0.65


def test_ma_article_classified_as_ma():
    r = classify_article_text(
        "삼성SDS, 두나무 지분 4% 인수로 디지털 자산 생태계 확장",
        "삼성SDS가 두나무 지분 4%를 인수했다.",
        company="samsung_sds",
        enable_llm=False,
    )
    assert r["event_type"] == "ma"
    assert r["importance_score"] >= 0.65


def test_partnership_article_classified_as_partnership():
    r = classify_article_text(
        "포스코DX와 NC AI, AI 로봇 자율작업 업무협약 체결",
        "포스코DX가 NC AI와 업무협약을 체결했다.",
        company="posco_dx",
        enable_llm=False,
    )
    assert r["event_type"] == "partnership"


def test_personnel_article_not_in_alert_set():
    r = classify_article_text(
        "삼성SDS, 신임 임원 인사 단행",
        "삼성SDS가 임원 인사와 조직 개편을 단행했다.",
        company="samsung_sds",
        enable_llm=False,
    )
    # 인사 = 대형 이벤트 게이트 대상 아님 → 알림 안 감을 시연하는 케이스.
    assert r["event_type"] not in {"ma", "contract", "partnership"}


def test_result_shape_has_required_fields():
    r = classify_article_text("테스트 제목", "본문", company="lg_cns", enable_llm=False)
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


def test_llm_fallback_path_no_keyerror(monkeypatch):
    """규칙 미매칭 → GPT-4o 폴백(_format_articles) 에서 source_name KeyError 안 나는지 회귀 가드.

    rule 을 강제로 미매칭시키고 LLM 을 스텁으로 갈아끼워, 합성 article 이 LLM 경로를
    통과(_format_articles 가 a["title"]·a["source_name"] bracket 접근)하는지 확인한다.
    """
    import src.observability
    import src.preprocessing.classification as clf

    class _Resp:
        content = '{"event_type": "financial", "reasoning": "stub"}'

    class _FakeLLM:
        def invoke(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(clf, "_classify_event_type_rule_based", lambda **kw: (None, ""))
    monkeypatch.setattr(clf, "openai_calls_enabled", lambda: True)
    monkeypatch.setattr(clf, "_get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(src.observability, "tracing_config", lambda **kw: None)

    r = clf.classify_article_text(
        "삼성SDS 관련 동향",  # 규칙 키워드 없음(게다가 강제 미매칭) → LLM 폴백
        "본문 내용입니다.",
        company="samsung_sds",
        enable_llm=True,
    )
    assert r["event_type"] == "financial"
