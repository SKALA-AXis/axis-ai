# 작성일: 2026-06-12
# 작성자: 최종민
# 변경이력:
#   2026-06-12 최종민 — 브리핑 read-through 재사용으로 페이지 로딩마다의 LLM 4회 호출 제거에 대한 회귀 테스트 추가
"""브리핑 read-through 재사용(briefing_reports 캐시) 회귀 테스트.

브리핑 페이지가 매 로딩마다 LLM 정제(최대 4회 GPT 호출)를 반복하던 문제의
해결 경로를 잠근다: 기본형 요청은 저장본을 재사용하고, 생성 시 캐시를 채운다.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from src.agents import briefing_generation_agent as bga
from src.agents.briefing import data_layer as briefing_data_layer
from src.agents.briefing_generation_agent import (
    BriefingGenerationAgent,
    _saved_report_is_fresh,
)

_PAST_PERIOD = {
    "date_from": datetime(2026, 6, 4).date(),
    "date_to": datetime(2026, 6, 4).date(),
}


def _current_period() -> dict:
    today = datetime.now(bga.KST).date()
    return {"date_from": today, "date_to": today}


def _saved_row(payload: dict, *, age: timedelta = timedelta(minutes=1)) -> dict:
    return {"payload": payload, "completed_at": datetime.now(UTC) - age}


# ---------------------------------------------------------------- freshness


def test_past_period_report_is_always_fresh():
    ancient = datetime(2026, 1, 1, tzinfo=UTC)
    assert _saved_report_is_fresh(ancient, _PAST_PERIOD) is True


def test_current_period_report_fresh_within_ttl():
    completed = datetime.now(UTC) - timedelta(minutes=5)
    assert _saved_report_is_fresh(completed, _current_period()) is True


def test_current_period_report_stale_beyond_ttl():
    completed = datetime.now(UTC) - timedelta(hours=2)
    assert _saved_report_is_fresh(completed, _current_period()) is False


def test_naive_completed_at_is_treated_as_utc():
    naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
    assert _saved_report_is_fresh(naive, _current_period()) is True


def test_missing_completed_at_is_stale_for_current_period():
    assert _saved_report_is_fresh(None, _current_period()) is False


# ------------------------------------------------------------ generate 경로


def test_default_request_reuses_saved_report(monkeypatch):
    sentinel = {"id": "BR-DAILY-20260604", "status": "completed", "cached": True}
    monkeypatch.setattr(bga, "load_briefing_report", lambda report_id: _saved_row(sentinel))

    def explode(**_kwargs):
        raise AssertionError("저장본 재사용 시 DB 페치가 실행되면 안 된다")

    monkeypatch.setattr(briefing_data_layer, "_fetch_period_integrated_issue_rows", explode)
    monkeypatch.setattr(bga, "_fetch_period_analysis_units", explode)

    result = asyncio.run(
        BriefingGenerationAgent().generate(briefing_type="daily", anchor_date="2026-06-04")
    )

    assert result is sentinel


def test_reuse_saved_false_forces_regeneration(monkeypatch):
    def explode(report_id):
        raise AssertionError("reuse_saved=False 면 저장본 조회도 하지 않는다")

    monkeypatch.setattr(bga, "load_briefing_report", explode)
    monkeypatch.setattr(bga, "_fetch_period_analysis_units", lambda **_kwargs: [])

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily", anchor_date="2026-06-04", reuse_saved=False
        )
    )

    assert result["id"] == "BR-DAILY-20260604"
    assert not result.get("cached")


def test_filtered_request_bypasses_cache(monkeypatch):
    def explode(report_id):
        raise AssertionError("필터 요청은 저장본과 내용이 달라 캐시를 보지 않는다")

    monkeypatch.setattr(bga, "load_briefing_report", explode)
    monkeypatch.setattr(bga, "_fetch_period_analysis_units", lambda **_kwargs: [])

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily",
            anchor_date="2026-06-04",
            peer_ids=["samsung_sds"],
        )
    )

    assert result["id"] == "BR-DAILY-20260604"


def test_stale_saved_report_triggers_regeneration(monkeypatch):
    stale = _saved_row({"id": "stale", "status": "completed"}, age=timedelta(hours=3))
    monkeypatch.setattr(bga, "load_briefing_report", lambda report_id: stale)
    monkeypatch.setattr(bga, "_fetch_period_analysis_units", lambda **_kwargs: [])

    # anchor 미지정 = 오늘(진행 중 기간) → 3시간 전 저장본은 TTL 초과로 재생성
    result = asyncio.run(BriefingGenerationAgent().generate(briefing_type="daily"))

    assert result["id"] != "stale"


def test_cacheable_generation_fills_cache_even_without_save_flag(monkeypatch):
    saved_calls: list[str] = []

    monkeypatch.setattr(bga, "load_briefing_report", lambda report_id: None)
    monkeypatch.setattr(
        bga,
        "save_briefing_report",
        lambda report, *, selected_cards: saved_calls.append(report["id"]),
    )
    monkeypatch.setattr(
        bga, "_refine_briefing_basis_with_llm", lambda **kwargs: kwargs["briefing_basis"]
    )
    monkeypatch.setattr(bga, "_refine_display_copy_with_llm", lambda **kwargs: kwargs["report"])
    monkeypatch.setattr(
        briefing_data_layer, "_fetch_period_integrated_issue_rows", _fake_issue_rows
    )

    result = asyncio.run(
        BriefingGenerationAgent().generate(
            briefing_type="daily", anchor_date="2026-06-04", save=False
        )
    )

    assert result["id"] == "BR-DAILY-20260604"
    assert saved_calls == ["BR-DAILY-20260604"]


def _fake_issue_rows(**_kwargs):
    return [
        {
            "integrated_issue_id": "11111111-1111-1111-1111-111111111111",
            "issue_key": "issue-key",
            "cluster_id": "cluster-1",
            "representative_raw_article_id": 1,
            "main_company": "samsung_sds",
            "event_type": "new_biz",
            "is_valid": True,
            "confidence": 0.82,
            "headline": "삼성SDS 생성형 AI 운영 플랫폼 확대",
            "one_line_summary": "운영 플랫폼 레퍼런스 확대",
            "source_ids": [1],
            "analyzed_source_ids": [1],
            "sectors": ["ax"],
            "mentioned_peer_companies": ["samsung_sds"],
            "content_summary": "운영 플랫폼 레퍼런스가 확대되고 있습니다.",
            "content_detailed_explanation": "제조와 금융 고객군에서 운영형 AI 신호가 확인됩니다.",
            "issue_brief": {"headline": "운영형 AI 레퍼런스 확대"},
            "content_digest": {
                "summary": "운영형 AI 수요가 확인됩니다.",
                "detailed_explanation": "고객 평가 기준이 운영 책임으로 이동합니다.",
            },
            "issue_frame": {"frame": "운영형 AI"},
            "issue_payload": {
                "is_valid_summary": True,
                "main_issue": "통합 이슈 원문",
                "integrated_text": "통합 이슈 상세",
                "source_article_ids": [1],
                "business_signals": [{"signal": "운영형 AI 레퍼런스"}],
            },
            "source_links": [
                {
                    "source_name": "연합뉴스",
                    "published_at": "2026-06-04T09:00:00+09:00",
                    "title": "기사",
                    "url": "https://example.com/news",
                }
            ],
            "source_names": ["연합뉴스"],
            "evidence_refs": [
                {"evidence_ref_id": "fact-1", "text": "통합 근거", "source_ids": [1]}
            ],
            "content_sections": [],
            "anchor_card_id": "CN-1",
            "anchor_peer_id": "samsung_sds",
            "anchor_importance": "high",
            "anchor_importance_score": 0.9,
            "anchor_evidence_payload": {
                "analysis_package": {
                    "integrated_issue": {"main_issue": "통합 이슈 원문"},
                    "analysis": {
                        "analysis_summary": "분석 요약",
                        "market_signal": "시장 신호",
                        "confidence": 0.8,
                    },
                    "implication": {
                        "skax_implication": {
                            "why_important": "운영 책임 기준을 확인해야 합니다.",
                            "potential_impact": "운영형 AI 수요 대응이 필요합니다.",
                            "recommended_actions": ["액션"],
                        },
                        "confidence": 0.8,
                    },
                    "classification": {"sector": "ax", "sectors": ["ax"]},
                    "validation": {"pass": True, "sc_score": 0.8},
                }
            },
            "basis_at": "2026-06-04T09:00:00+09:00",
            "issue_created_at": "2026-06-04T09:00:00+09:00",
        }
    ]
