"""공용 LLM 팩토리 동작 잠금 (refactoring-architecture R1).

ChatOpenAI 를 직접 만들지 않고 kwargs 구성만 검증하기 위해 생성자를 가로채
전달된 인자를 캡처한다. 동작 불변 이행(briefing/today_insight 등)의 안전망.
"""

from __future__ import annotations

import pytest

from src.llm import LLMSpec, build_chat_llm, is_reasoning_model


@pytest.fixture
def capture_chat_openai(monkeypatch):
    """build_chat_llm 내부 lazy import 대상(langchain_openai.ChatOpenAI)을 가로챈다."""
    captured: dict = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChatOpenAI)
    return captured


# ---------------------------------------------------------------- is_reasoning


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.5", True),
        ("gpt-5", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


# ------------------------------------------------------------- non-reasoning


def test_gpt4o_omits_reasoning_effort(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-4o", temperature=0.1, max_tokens=2400))
    assert capture_chat_openai["model"] == "gpt-4o"
    assert capture_chat_openai["temperature"] == 0.1
    assert capture_chat_openai["max_completion_tokens"] == 2400
    # 비추론 모델엔 reasoning_effort 전달 금지 (gpt-4o 는 거부)
    assert "reasoning_effort" not in capture_chat_openai


def test_json_object_wraps_model_kwargs(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-4o", json_object=True))
    assert capture_chat_openai["model_kwargs"] == {"response_format": {"type": "json_object"}}


def test_no_json_object_omits_model_kwargs(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-4o", json_object=False))
    assert "model_kwargs" not in capture_chat_openai


# ----------------------------------------------------------------- reasoning


def test_gpt5_applies_reasoning_effort(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-5.5", reasoning_effort="medium"))
    assert capture_chat_openai["reasoning_effort"] == "medium"


def test_gpt5_uses_reasoning_cap_when_provided(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-5.5", max_tokens=2400, max_tokens_reasoning=8000))
    assert capture_chat_openai["max_completion_tokens"] == 8000


def test_gpt5_falls_back_to_base_cap_when_no_reasoning_cap(capture_chat_openai):
    build_chat_llm(LLMSpec(model="gpt-5.5", max_tokens=2400))
    assert capture_chat_openai["max_completion_tokens"] == 2400


def test_gpt4o_ignores_reasoning_cap(capture_chat_openai):
    # 비추론 모델은 max_tokens_reasoning 가 있어도 base 캡 사용
    build_chat_llm(LLMSpec(model="gpt-4o", max_tokens=2400, max_tokens_reasoning=8000))
    assert capture_chat_openai["max_completion_tokens"] == 2400


# ------------------------------------------ 이행 사이트 동작 동등성 (회귀)


def test_briefing_site_equivalence(capture_chat_openai):
    """briefing/basis_builder 의 기존 kwargs 와 동일한지 (gpt-5.5 경로)."""
    build_chat_llm(
        LLMSpec(
            model="gpt-5.5",
            temperature=0.1,
            max_tokens=8000,
            json_object=True,
            reasoning_effort="low",
        )
    )
    assert capture_chat_openai == {
        "model": "gpt-5.5",
        "temperature": 0.1,
        "max_completion_tokens": 8000,
        "model_kwargs": {"response_format": {"type": "json_object"}},
        "reasoning_effort": "low",
    }


def test_today_insight_site_equivalence_gpt4o(capture_chat_openai):
    """today_insight 가 gpt-4o 로 떨어졌을 때 reasoning_effort 미전달 확인."""
    build_chat_llm(
        LLMSpec(
            model="gpt-4o",
            temperature=0.18,
            max_tokens=3200,
            json_object=True,
            reasoning_effort="low",
        )
    )
    assert capture_chat_openai == {
        "model": "gpt-4o",
        "temperature": 0.18,
        "max_completion_tokens": 3200,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
