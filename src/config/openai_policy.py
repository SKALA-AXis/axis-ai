"""Runtime guard for explicit OpenAI calls."""

from __future__ import annotations

import os

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def openai_calls_enabled() -> bool:
    flag = os.getenv("ENABLE_OPENAI_CALLS")
    return _flag_enabled(flag) and _api_key_configured()


def openai_disabled_reason() -> str:
    flag = os.getenv("ENABLE_OPENAI_CALLS")
    if flag is not None and flag.strip().lower() in _FALSE_VALUES:
        return "ENABLE_OPENAI_CALLS explicitly disabled OpenAI calls"
    if not _api_key_configured():
        return "OPENAI_API_KEY is not configured"
    return "ENABLE_OPENAI_CALLS is not enabled"


def _api_key_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _flag_enabled(flag: str | None) -> bool:
    if flag is None:
        return False
    return flag.strip().lower() in _TRUE_VALUES
