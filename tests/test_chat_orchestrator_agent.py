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


class _FakeReportLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config=None):  # noqa: ANN001
        self.prompts.append(prompt)

        class _Response:
            content = (
                '{"reply":"보고서 초안을 만들었습니다.",'
                '"answer_blocks":[{"type":"summary","title":"핵심 요약",'
                '"items":["금융 AX 수요 확대"]}],'
                '"report_draft":{"title":"금융 AX 동향 보고서",'
                '"sections":[{"title":"핵심 변화",'
                '"body":"금융권 AX 수요가 보안 요구와 함께 확대됩니다."},'
                '{"title":"SK AX 대응",'
                '"body":"운영 안정성과 보안 레퍼런스를 전면에 배치합니다."}]},'
                '"follow_up_suggestions":["브리핑에 추가"],"confidence":0.84}'
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


@pytest.mark.asyncio
async def test_market_trend_request_includes_selected_peer_context(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = _FakeLLM()
    agent = ChatOrchestratorAgent(llm=fake_llm)

    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate(
                "card_news",
                "CN-LG",
                "LG CNS 클라우드 전환 수주",
                "LG CNS는 금융권 클라우드 전환과 AX 운영 요구를 함께 제시했습니다.",
                0.9,
                {"peer_id": "lg_cns"},
            ),
            RetrievalCandidate(
                "card_news",
                "CN-SDS",
                "삼성SDS 생성형 AI 플랫폼 확산",
                "삼성SDS는 생성형 AI 플랫폼 적용 범위를 확대하고 있습니다.",
                0.87,
                {"peer_id": "samsung_sds"},
            ),
            RetrievalCandidate(
                "card_news",
                "CN-AUTO",
                "현대오토에버 제조 AX 고도화",
                "현대오토에버는 제조 운영 데이터와 AX 적용을 연결하고 있습니다.",
                0.84,
                {"peer_id": "hyundai_autoever"},
            ),
            RetrievalCandidate(
                "card_news",
                "CN-POSCO",
                "포스코DX 스마트팩토리 확대",
                "포스코DX는 스마트팩토리와 산업 자동화 축을 강화하고 있습니다.",
                0.81,
                {"peer_id": "posco_dx"},
            ),
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(
            message="전체적인 시장 동향을 peer사 기준으로 구체적으로 설명해줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

    assert response["intent"] == "market_trend"
    assert fake_llm.prompts
    prompt = fake_llm.prompts[0]
    assert "intent가 market_trend이면" in prompt
    assert "삼성SDS" in prompt
    assert "LG CNS" in prompt
    assert "현대오토에버" in prompt
    assert "포스코DX" in prompt
    assert "CN-LG" in prompt
    assert "report_draft" not in response


@pytest.mark.asyncio
async def test_peer_profile_summary_request_stays_in_axis_scope(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = ChatOrchestratorAgent(enable_llm=False)

    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate(
                "peer_profile",
                "lg_cns",
                "LG CNS",
                "클라우드, 금융 IT, AX 운영 전환 역량을 보유한 주요 피어사입니다.",
                0.8,
            )
        ],
    )

    response = await agent.answer(ChatTurnRequest(message="lg cns 피어사 정보를 요약해서 알려줘"))

    assert response["intent"] == "page_qa"
    assert response["scope"] == "global_axis_data"
    assert "LG CNS" in response["reply"]
    assert response["blocked"] is False


@pytest.mark.asyncio
async def test_relative_day_briefing_request_routes_to_report_lookup(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = ChatOrchestratorAgent(enable_llm=False)

    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate(
                "briefing_report",
                "BR-3D",
                "3일 전 브리핑",
                "금융권 AX 전환과 피어사 수주 신호를 정리한 브리핑입니다.",
                0.72,
            )
        ],
    )

    response = await agent.answer(ChatTurnRequest(message="3일전 브리핑 내용을 설ㅈ명해줘"))

    assert response["intent"] == "report_lookup"
    assert response["blocked"] is False
    assert "보고서 초안" in response["reply"]
    assert response["report_draft"]["title"] == "AXIS 브리핑 초안"


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
async def test_weather_question_is_guarded_before_retrieval(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent()

    def fail_retrieve(_request):
        raise AssertionError("weather questions should be blocked before retrieval")

    monkeypatch.setattr(agent, "_retrieve", fail_retrieve)

    response = await agent.answer(
        ChatTurnRequest(
            message="서울 날씨 알려줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

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
async def test_company_news_lookup_uses_recent_fast_path(monkeypatch: pytest.MonkeyPatch):
    fake_llm = _FakeLLM()
    agent = ChatOrchestratorAgent(llm=fake_llm)

    monkeypatch.setattr(
        agent,
        "_recent_news_search",
        lambda _query, days, limit: [
            RetrievalCandidate(
                "raw_article",
                "RA-2",
                "LG CNS, 금융 AX 사업 확대",
                "최근 기사",
                0.9,
                {"published_at": "2026-06-08T09:00:00+09:00", "source_name": "AXIS News"},
            ),
            RetrievalCandidate(
                "raw_article",
                "RA-1",
                "LG CNS, 클라우드 전환 수주",
                "전일 기사",
                0.88,
                {"published_at": "2026-06-07T09:00:00+09:00", "source_name": "AXIS News"},
            ),
        ],
    )

    response = await agent.answer(ChatTurnRequest(message="lg cns 기사를 알려줘"))

    assert response["intent"] == "news_lookup"
    assert "최근 7일" in response["reply"]
    assert "1. LG CNS, 금융 AX 사업 확대" in response["reply"]
    assert "2. LG CNS, 클라우드 전환 수주" in response["reply"]
    assert response["provenance"]["retrieval_mode"] == "recent_news_fast_path"
    assert fake_llm.prompts == []


@pytest.mark.asyncio
async def test_report_lookup_returns_report_draft(monkeypatch: pytest.MonkeyPatch):
    fake_llm = _FakeReportLLM()
    agent = ChatOrchestratorAgent(llm=fake_llm)

    monkeypatch.setattr(
        agent,
        "_retrieve",
        lambda _request: [
            RetrievalCandidate(
                "card_news",
                "CN-R1",
                "금융 AX 수주 확대",
                "LG CNS와 삼성SDS가 금융 AX 전환 수요를 공략하고 있습니다.",
                0.9,
            ),
            RetrievalCandidate(
                "integrated_issue",
                "II-R1",
                "보안 요구 동반 상승",
                "금융권 AX 제안에서 보안·감사 대응 요구가 반복 관찰됩니다.",
                0.82,
            ),
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(
            message="이 내용으로 보고서 초안 출력해줘",
            current_page=ChatPageContext(route="/dashboard"),
        )
    )

    assert response["intent"] == "report_lookup"
    assert fake_llm.prompts
    assert response["report_draft"]["title"] == "금융 AX 동향 보고서"
    assert response["report_draft"]["sections"][0]["title"] == "목차 및 구성"
    assert len(response["report_draft"]["sections"]) >= 6
    assert response["answer_blocks"][0]["title"] == "핵심 요약"


@pytest.mark.asyncio
async def test_card_news_pdf_request_returns_printable_report_without_today_shortcut(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = ChatOrchestratorAgent(enable_llm=False)

    def fail_today_lookup():
        raise AssertionError("PDF/report export request should not use today insight shortcut")

    monkeypatch.setattr(agent, "_lookup_today_insight", fail_today_lookup)
    monkeypatch.setattr(
        agent,
        "_recent_card_report_search",
        lambda _query, limit: [
            RetrievalCandidate(
                "card_news",
                "CN-TODAY",
                "LG CNS, 클로드 오픈 및 AX 추진",
                "오늘 36건 신호 중 영향은 크지만 보도가 아직 적은 단건 이벤트입니다.",
                0.91,
                {"created_at": "2026-06-10T08:30:00+09:00", "peer_id": "lg_cns"},
            ),
            RetrievalCandidate(
                "card_news",
                "CN-YESTERDAY",
                "삼성SDS 금융 AX 수주 확대",
                "전일 카드뉴스에서 금융 AX 운영 요구가 반복 확인됐습니다.",
                0.87,
                {"created_at": "2026-06-09T08:30:00+09:00", "peer_id": "samsung_sds"},
            ),
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(message="어제 오늘의 카드뉴스를 요약한 pdf를 만들어줘.")
    )

    assert response["intent"] == "report_lookup"
    assert response["report_draft"]["title"] == "AXIS 카드뉴스 요약 PDF"
    assert "PDF 저장/출력" in response["reply"]
    assert response["provenance"]["export_requested"] == "pdf"
    assert [source["id"] for source in response["sources"]] == ["CN-TODAY", "CN-YESTERDAY"]


@pytest.mark.asyncio
async def test_today_content_pdf_request_uses_recent_cards(monkeypatch: pytest.MonkeyPatch):
    agent = ChatOrchestratorAgent(enable_llm=False)

    monkeypatch.setattr(
        agent,
        "_recent_card_report_search",
        lambda _query, limit: [
            RetrievalCandidate(
                "card_news",
                "CN-TODAY",
                "오늘 감지된 AX 실행 신호",
                "오늘 카드뉴스에서 AX 운영 전환과 경쟁사 계약 신호가 확인됐습니다.",
                0.91,
                {"created_at": "2026-06-10T08:30:00+09:00", "peer_id": "lg_cns"},
            )
        ],
    )

    response = await agent.answer(
        ChatTurnRequest(message="오늘 있었던 내용을 pdf로 정리해서 보여줘")
    )

    assert response["intent"] == "report_lookup"
    assert response["report_draft"]["title"] == "AXIS 카드뉴스 요약 PDF"
    assert "PDF 저장/출력" in response["reply"]
    assert response["sources"][0]["id"] == "CN-TODAY"


@pytest.mark.asyncio
async def test_print_followup_returns_ui_help_without_fixture() -> None:
    agent = ChatOrchestratorAgent(enable_llm=False)

    response = await agent.answer(ChatTurnRequest(message="프린트할수있게해줘"))

    assert response["intent"] == "print_help"
    assert response["scope"] == "assistant_ui"
    assert "PDF 저장/출력" in response["reply"]
    assert response["provenance"]["export_requested"] == "pdf"


@pytest.mark.asyncio
async def test_print_followup_with_history_returns_printable_report() -> None:
    agent = ChatOrchestratorAgent(enable_llm=False)

    response = await agent.answer(
        ChatTurnRequest(
            message="프린트할수있게해줘",
            history=[
                {"role": "user", "content": "오늘 핵심 신호를 요약해줘"},
                {
                    "role": "assistant",
                    "content": "금융권 AX 전환 수요와 보안 거버넌스 요구가 함께 확대되고 있습니다.",
                },
            ],
        )
    )

    assert response["intent"] == "report_lookup"
    assert response["scope"] == "assistant_history"
    assert response["report_draft"]["title"] == "AXIS 대화 기반 PDF"
    assert response["report_draft"]["sections"][0]["title"] == "목차 및 구성"
    assert response["report_draft"]["sections"][2]["title"] == "상세 정리"
    assert "PDF 저장/출력" in response["reply"]
    assert response["provenance"]["export_requested"] == "pdf"


@pytest.mark.asyncio
async def test_context_pdf_request_uses_history_when_retrieval_is_empty() -> None:
    agent = ChatOrchestratorAgent(enable_llm=False)

    response = await agent.answer(
        ChatTurnRequest(
            message="이 내용을 pdf로 만들어줘",
            history=[
                {
                    "role": "assistant",
                    "content": "LG CNS와 삼성SDS의 금융 AX 수주 확대 신호가 확인됐습니다.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "SK AX 관점에서는 운영 안정성과 보안 레퍼런스 메시지를 강화해야 합니다."
                    ),
                },
            ],
        )
    )

    assert response["intent"] == "report_lookup"
    assert response["scope"] == "assistant_history"
    assert response["report_draft"]["title"] == "AXIS 대화 기반 PDF"
    assert "금융 AX 수주 확대" in response["report_draft"]["sections"][1]["body"]
    assert "SK AX 관점" in response["report_draft"]["sections"][5]["title"]
    assert "PDF 저장/출력" in response["reply"]


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
    assert response["provenance"]["retrieval_mode"] == "global_lexical+hybrid_rag+llm"
