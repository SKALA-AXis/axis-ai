"""today_insight shared config — extracted from facade (move-only)."""

from __future__ import annotations

import os

_LLM_MODEL = os.getenv("TODAY_INSIGHT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
_PROMPT_VERSION = "today-insight-v1.4-qualitative-signals"
