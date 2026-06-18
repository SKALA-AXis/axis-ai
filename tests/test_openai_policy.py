# 작성일: 2026-05-26
# 작성자: 박지원
# 변경이력:
#   2026-05-26 박지원 — 전처리 수정 및 전처리 LLM 폴백 제거
from __future__ import annotations

import pytest

from src.config.openai_policy import (
    openai_calls_enabled,
    openai_disabled_reason,
)


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on"])
def test_openai_calls_enabled_only_when_global_flag_is_enabled(monkeypatch, flag):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_OPENAI_CALLS", flag)

    assert openai_calls_enabled() is True


def test_openai_calls_disabled_by_default_even_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ENABLE_OPENAI_CALLS", raising=False)

    assert openai_calls_enabled() is False
    assert openai_disabled_reason() == "ENABLE_OPENAI_CALLS is not enabled"


@pytest.mark.parametrize("flag", ["0", "false", "no", "off"])
def test_openai_calls_can_be_explicitly_disabled(monkeypatch, flag):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_OPENAI_CALLS", flag)

    assert openai_calls_enabled() is False
    assert openai_disabled_reason() == "ENABLE_OPENAI_CALLS explicitly disabled OpenAI calls"


def test_openai_calls_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_OPENAI_CALLS", raising=False)

    assert openai_calls_enabled() is False
    assert openai_disabled_reason() == "OPENAI_API_KEY is not configured"
