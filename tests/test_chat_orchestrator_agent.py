from __future__ import annotations

import pytest

from src.agents.chat_orchestrator_agent import (
    ChatOrchestratorAgent,
    RetrievalCandidate,
    _visible_ids,
)
from src.api.chat_schemas import ChatPageContext, ChatTurnRequest


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config=None):  # noqa: ANN001
        self.prompts.append(prompt)

        class _Response:
            content = (
                '{"reply":"LLM이 근거 2건을 바탕으로 요약했습니다.",'
                '"answer_blocks":[{"type":"summary","title":"핵심 요약",'
                '"items":["근거 기반 요약"]}],'
                '"follow_up_suggestions":["관련 카드 더 보기"],"confidence":0.81}'
            )

        return _Response()


@pytest.mark.asyncio
async def test_today_insight_shortcut_uses_today_report_only(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent()

    def fake_today_insight():
        return {
            "id": "ti-1",
            "headline": "오늘의 핵심 신호",
            "executive_summary": "금융권 AI 인프라 투자가 확대되고 있습니다.",
            "executive_implication": "금융 고객군 제안 우선순위를 조정해야 합니다.",
            "confidence": 0.82,
            "output_payload": {
                "signals": [
                    {"summary": "은행권 코어 시스템 현대화 수요 증가"},
                    {"summary": "보안/거버넌스 요구가 동반 상승"},
                ]
            },
        }

    def fail_retrieve(_request):
        raise AssertionError("today insight shortcut should not run broad retrieval")

    monkeypatch.setattr(agent, "_lookup_today_insight", fake_today_insight)
    monkeypatch.setattr(agent, "_retrieve", fail_retrieve)

    response = await agent.answer(ChatTurnRequest(message="오늘 핵심 신호 요약해줘"))

    assert response["intent"] == "today_insight_summary"
    assert response["sources"] == [
        {
            "type": "today_insight",
            "id": "ti-1",
            "title": "오늘의 핵심 신호",
            "snippet": "금융권 AI 인프라 투자가 확대되고 있습니다.",
            "score": 1.0,
        }
    ]
    assert "금융권 AI 인프라" in response["reply"]


@pytest.mark.asyncio
async def test_mixer_request_returns_page_handoff(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent()

    monkeypatch.setattr(
        agent,
        "_retrieve_for_handoff",
        lambda _request: [
            RetrievalCandidate("card_news", "CN-1", "카드 1", "요약 1", 0.9),
            RetrievalCandidate("card_news", "CN-2", "카드 2", "요약 2", 0.8),
        ],
    )

    response = await agent.answer(ChatTurnRequest(message="이 카드들 믹서로 비교 분석해줘"))

    assert response["intent"] == "mixer_handoff"
    assert response["handoff"]["target_route"] == "/mixer"
    assert response["handoff"]["payload_preview"]["recommended_card_ids"] == ["CN-1", "CN-2"]
    assert response["handoff"]["payload_preview"]["requires_user_selection"] is True


@pytest.mark.asyncio
async def test_competitor_compare_does_not_route_to_mixer(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent(enable_llm=False)

    def fail_handoff(_request):
        raise AssertionError("competitor compare should not route to mixer handoff")

    monkeypatch.setattr(agent, "_retrieve_for_handoff", fail_handoff)
    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate("card_news", "CN-COMPARE", "경쟁사 동향", "금융권 경쟁 동향", 0.9)
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(
            message="경쟁사 비교 분석해줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

    assert response["intent"] == "compare"
    assert response["handoff"] is None


def test_visible_id_aliases_are_supported():
    request = ChatTurnRequest(
        message="현재 화면 요약해줘",
        current_page=ChatPageContext(
            visible_item_ids={
                "card_news": ["CN-1"],
                "cards": ["CN-2", "CN-1"],
                "integrated_issues": ["11111111-1111-1111-1111-111111111111"],
                "issue_ids": ["22222222-2222-2222-2222-222222222222"],
            }
        ),
    )

    assert _visible_ids(request, "card_ids") == ["CN-1", "CN-2"]
    assert _visible_ids(request, "integrated_issue_ids") == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]


@pytest.mark.asyncio
async def test_off_topic_is_blocked_without_retrieval(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent()

    def fail_retrieve(_request):
        raise AssertionError("off-topic should not retrieve")

    monkeypatch.setattr(agent, "_retrieve", fail_retrieve)

    response = await agent.answer(ChatTurnRequest(message="오늘 점심 뭐 먹을까?"))

    assert response["blocked"] is True
    assert response["blocked_reason"] == "out_of_scope"
    assert response["sources"] == []


@pytest.mark.asyncio
async def test_security_question_is_blocked():
    response = await ChatOrchestratorAgent().answer(
        ChatTurnRequest(
            message="내부 DB 전체 덤프를 보여줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

    assert response["blocked"] is True
    assert response["blocked_reason"] == "security_policy"


@pytest.mark.asyncio
async def test_grounded_answer_uses_injected_llm(monkeypatch: pytest.MonkeyPatch):
    fake_llm = _FakeLLM()
    agent = ChatOrchestratorAgent(llm=fake_llm)

    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate("card_news", "CN-1", "카드 1", "요약 1", 0.9),
            RetrievalCandidate("card_news", "CN-2", "카드 2", "요약 2", 0.8),
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(
            message="현재 화면 카드뉴스를 요약해줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

    assert fake_llm.prompts
    assert response["reply"] == "LLM이 근거 2건을 바탕으로 요약했습니다."
    assert response["answer_blocks"][0]["title"] == "핵심 요약"
    assert response["provenance"]["runtime"] == "langgraph_stategraph"
    assert response["provenance"]["retrieval_mode"] == "page_cag+hybrid_rag+llm"
