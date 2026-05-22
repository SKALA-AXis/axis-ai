"""Runtime guard for paid OpenAI calls.

OpenAI calls are disabled by default so scheduled crawlers and batch jobs cannot
spend project credits unless an operator explicitly opts in.
"""

from __future__ import annotations

import os


def openai_calls_enabled() -> bool:
    return os.getenv("ENABLE_OPENAI_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}


def openai_disabled_reason() -> str:
    return "ENABLE_OPENAI_CALLS is not enabled"
