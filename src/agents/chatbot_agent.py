"""Chatbot agent structure.

ChatbotAgent belongs to the 2단계 data-usage pipeline. It answers user questions
from stored raw/clean data, card news, implications, evidence chains, and
profile context. Detailed RAG/search orchestration is left for a later pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ChatbotAgent:
    """Repository-grounded Q&A agent contract."""

    prompt_version = "chatbot-agent-structure-v0"

    def answer(
        self,
        *,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "answer": "",
            "sources": [],
            "related_cards": [],
            "confidence": "low",
            "validation": {
                "pass": False,
                "reason": "structure_only",
            },
            "context": context or {},
        }


__all__ = ["ChatbotAgent"]
