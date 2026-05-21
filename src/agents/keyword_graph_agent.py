"""Keyword graph agent structure.

KeywordGraphAgent belongs to the 2단계 data-usage pipeline. It will generate
company/sector/keyword nodes, edges, and time-series signals from stored
classified data and card news. The graph-building logic is left for a later
implementation pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class KeywordGraphAgent:
    """Generate keyword graph-shaped outputs from stored monitoring data."""

    prompt_version = "keyword-graph-agent-structure-v0"

    def build(
        self,
        *,
        items: list[dict[str, Any]],
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nodes": [],
            "edges": [],
            "time_series": {},
            "source_count": len(items or []),
            "validation": {
                "pass": False,
                "reason": "structure_only",
            },
            "filters": filters or {},
        }


__all__ = ["KeywordGraphAgent"]
