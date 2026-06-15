"""api/router.py 엔드포인트 happy-path 회귀 (PROJECT_STRUCTURE_PLAN 2-A5).

에이전트/서비스 경계를 mock 으로 끊고 라우터 자체의 검증·매핑·응답 모델 계약만
잠근다. TestClient 를 context manager 없이 사용해 lifespan(모델 preload)을
건너뛴다. /mixer/analyze/stream 은 SSE+스레드 구조라 본 happy-path 범위에서 제외.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.api import router as router_module
from src.api.router import app

client = TestClient(app, raise_server_exceptions=False)


def _fake_async_agent(method_name: str, result: dict) -> type:
    """인자 무시하고 고정 결과를 반환하는 async 메서드 1개짜리 fake 클래스."""

    async def _method(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return result

    return type("FakeAgent", (), {"__init__": lambda self, *a, **k: None, method_name: _method})


# ------------------------------------------------------------------- health


def test_healthz_returns_ok_without_dependencies():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reports_ok_when_db_and_qdrant_up(monkeypatch):
    monkeypatch.setattr(router_module, "_check_db", lambda: True)
    monkeypatch.setattr(router_module, "_check_qdrant", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
    assert body["qdrant_connected"] is True


def test_health_degraded_when_db_down(monkeypatch):
    monkeypatch.setattr(router_module, "_check_db", lambda: False)
    monkeypatch.setattr(router_module, "_check_qdrant", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


# ----------------------------------------------------------------- pipeline


def test_pipeline_run_accepts_and_queues_track(monkeypatch):
    captured: list[tuple] = []

    async def fake_track(*args):
        captured.append(args)

    monkeypatch.setattr(router_module, "_run_collection_track", fake_track)

    response = client.post("/pipeline/run", json={"company": ["samsung_sds"]})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert body["task_id"]
    # TestClient 는 응답 후 background task 를 실행한다
    assert len(captured) == 1


def test_pipeline_run_rejects_unknown_track():
    response = client.post("/pipeline/run", json={"company": [], "track": "zzz"})
    assert response.status_code == 400


def test_pipeline_delivery_builds_briefing_content(monkeypatch):
    class FakeGraph:
        def invoke(self, state):
            return {
                "subject": "[AXIS] 데일리 브리핑",
                "html": "<html></html>",
                "text": "본문",
                "errors": [],
            }

    monkeypatch.setattr("src.pipeline.delivery_graph.delivery_graph", FakeGraph())

    response = client.post("/pipeline/delivery", json={"cards": []})

    assert response.status_code == 200
    assert response.json()["subject"] == "[AXIS] 데일리 브리핑"


# ----------------------------------------------------------------- briefing


def test_briefing_generate_passes_through_agent_payload(monkeypatch):
    payload = {"id": "BR-DAILY-20260612", "status": "completed", "title": "브리핑"}
    monkeypatch.setattr(
        "src.agents.briefing_generation_agent.BriefingGenerationAgent",
        _fake_async_agent("generate", payload),
    )

    response = client.post("/briefing/generate", json={"briefing_type": "daily"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "BR-DAILY-20260612"
    assert body["status"] == "completed"


def test_briefing_generate_maps_value_error_to_400(monkeypatch):
    class FakeAgent:
        async def generate(self, **kwargs):
            raise ValueError("anchor_date 형식 오류")

    monkeypatch.setattr("src.agents.briefing_generation_agent.BriefingGenerationAgent", FakeAgent)

    response = client.post("/briefing/generate", json={"briefing_type": "daily"})

    assert response.status_code == 400


# -------------------------------------------------------------------- cards


def test_list_cards_paginates_items(monkeypatch):
    items = [{"id": f"CN-{i}"} for i in range(5)]
    monkeypatch.setattr(
        router_module, "_build_card_news_items", lambda limit, today_only: items[:limit]
    )

    response = client.get("/api/cards", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    data = response.json()["data"]
    assert [item["id"] for item in data["items"]] == ["CN-1", "CN-2"]
    assert data["limit"] == 2 and data["offset"] == 1


def test_list_today_cards_wraps_api_response(monkeypatch):
    monkeypatch.setattr(router_module, "_build_card_news_items", lambda limit, today_only: [])

    response = client.get("/api/cards/today")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total"] == 0


def test_saved_card_news_item_uses_stored_created_at_as_card_date():
    item = router_module._saved_card_news_item_from_row(
        {
            "id": "CN-20260612-48480",
            "company": "lg_cns",
            "peer_company_id": "lg_cns",
            "cluster_id": 48480,
            "title": "LG CNS, AI 에이전트 및 로봇 플랫폼 통합 추진",
            "summary_lines": ["요약"],
            "event_type": "tech",
            "importance": "medium",
            "importance_score": 0.7,
            "implication": {
                "frontend": {"why_important": "시사점", "recommended_actions": ["대응"]}
            },
            "sources": [],
            "validation_pass": True,
            "validation_sc_score": 0.8,
            "primary_keyword_category": "ax",
            "source_raw_article_ids": [48480],
            "source_articles": [],
            "image_assets": [],
            "created_at": datetime(2026, 6, 12, 6, 11, 37, tzinfo=UTC),
        }
    )

    assert item["published_date"] == "2026-06-12"
    assert item["created_at"] == "2026-06-12T06:11:37+00:00"
    assert item["display_sections"][1]["items"] == ["시사점"]


# ------------------------------------------------------------------- search


def test_search_returns_hits_total(monkeypatch):
    monkeypatch.setattr(router_module, "_run_search_pipeline", lambda request: ([], {}))

    response = client.post("/search", json={"query": "LG CNS 전략"})

    assert response.status_code == 200
    assert response.json() == {"hits": [], "total": 0}


def test_gen_search_empty_hits_returns_guidance(monkeypatch):
    monkeypatch.setattr(router_module, "_run_search_pipeline", lambda request: ([], {}))

    response = client.post("/gen-search", json={"query": "약한 신호"})

    assert response.status_code == 200
    body = response.json()
    assert body["sc_passed"] is False
    assert body["sources"] == []


# --------------------------------------------------------------------- chat


def test_chat_validates_orchestrator_response(monkeypatch):
    result = {
        "conversation_id": "conv-1",
        "session_id": "sess-1",
        "message_id": "msg-1",
        "reply": "안녕하세요",
        "intent": "smalltalk",
        "scope": "page",
    }
    monkeypatch.setattr(
        "src.agents.chat_orchestrator_agent.ChatOrchestratorAgent",
        _fake_async_agent("answer", result),
    )

    response = client.post("/chat", json={"message": "안녕"})

    assert response.status_code == 200
    assert response.json()["reply"] == "안녕하세요"


def test_chat_pdf_rejects_invalid_base64():
    response = client.post(
        "/chat/pdf",
        json={"request": {"message": "분석해줘"}, "file_name": "a.pdf", "pdf_base64": "!!!"},
    )
    assert response.status_code == 400


# ----------------------------------------------------------------- insights


def test_today_insight_generate_happy_path(monkeypatch):
    monkeypatch.setattr(
        "src.agents.today_insight_agent.TodayInsightAgent",
        _fake_async_agent("generate", {"status": "success"}),
    )

    response = client.post("/today-insight/generate", json={})

    assert response.status_code == 200


def test_insight_generate_happy_path(monkeypatch):
    monkeypatch.setattr(
        "src.agents.insight_cascade_agent.InsightCascadeAgent",
        _fake_async_agent("generate", {"status": "success"}),
    )

    response = client.post("/insight/generate", json={"card_ids": ["CN-1"]})

    assert response.status_code == 200


def test_insight_generate_agent_failure_maps_to_502(monkeypatch):
    monkeypatch.setattr(
        "src.agents.insight_cascade_agent.InsightCascadeAgent",
        _fake_async_agent("generate", {"status": "failed", "error": "LLM 호출 실패"}),
    )

    response = client.post("/insight/generate", json={"card_ids": ["CN-1"]})

    assert response.status_code == 502


# -------------------------------------------------------------------- mixer


def test_mixer_analyze_happy_path(monkeypatch):
    monkeypatch.setattr(
        "src.agents.mixer_analysis_agent.MixerAnalysisAgent",
        _fake_async_agent("analyze", {"mix_id": "MX-1", "status": "success"}),
    )

    response = client.post("/mixer/analyze", json={"card_ids": ["CN-1"]})

    assert response.status_code == 200
    assert response.json()["mix_id"] == "MX-1"


# ------------------------------------------------------------ global trends


def test_global_trends_run_happy_path(monkeypatch):
    class FakeTrendAgent:
        def generate(self, trend_input):
            return {"status": "success", "analysis_id": "GT-1", "confidence": 0.8}

    monkeypatch.setattr("src.agents.it_trend_agent.ITTrendAgent", FakeTrendAgent)

    response = client.post("/global/trends/run", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == "GT-1"


# --------------------------------------------------------------- link/weak


def test_link_verify_happy_path(monkeypatch):
    monkeypatch.setattr(
        "src.services.link_verification.LinkVerificationService",
        _fake_async_agent(
            "verify", {"card_id": "CN-1", "verified_at": "2026-06-12T10:00:00+09:00"}
        ),
    )

    response = client.post("/link/verify", json={"card_id": "CN-1"})

    assert response.status_code == 200
    assert response.json()["card_id"] == "CN-1"


def test_weak_signal_run_returns_501():
    response = client.post("/weak-signal/run")
    assert response.status_code == 501
