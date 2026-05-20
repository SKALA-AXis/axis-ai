"""Report agent structure.

ReportAgent belongs to the 2단계 data-usage pipeline. It will generate daily,
weekly, monthly, executive, company, and sector reports from stored card news,
implications, integrated issues, and trend outputs. The detailed generation
logic is intentionally left for a later implementation pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ReportAgent:
    """Generate report-shaped outputs from stored analysis results."""

    prompt_version = "report-agent-structure-v0"

    def generate(
        self,
        *,
        report_type: str,
        items: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "report_type": report_type,
            "source_count": len(items or []),
            "summary": "",
            "sections": [],
            "sources": [],
            "validation": {
                "pass": False,
                "reason": "structure_only",
            },
            "context": context or {},
        }


__all__ = ["ReportAgent"]
