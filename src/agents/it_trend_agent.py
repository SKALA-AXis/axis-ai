"""IT trend agent structure.

This agent is the future entry point for trend briefs built from normalized
external trend sources such as research institutes, consulting reports, global
company newsroom items, and global peer/company news. The implementation is
intentionally skeletal:
collection/parsing stays outside the agent, and this class will consume already
normalized items when the trend logic is defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ITTrendInput:
    """Runtime DTO for IT trend generation."""

    items: list[dict[str, Any]]
    period: str | None = None
    source_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ITTrendAgent:
    """Generate current IT trend views from normalized trend-source data."""

    prompt_version = "it-trend-structure-v0"

    def build_input(
        self,
        *,
        items: list[dict[str, Any]],
        period: str | None = None,
        source_groups: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ITTrendInput:
        return ITTrendInput(
            items=list(items or []),
            period=period,
            source_groups=list(source_groups or []),
            metadata=dict(metadata or {}),
        )

    def generate(self, trend_input: ITTrendInput) -> dict[str, Any]:
        """Return the stable output contract until the trend logic is added."""
        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "period": trend_input.period,
            "source_groups": trend_input.source_groups,
            "source_count": len(trend_input.items),
            "trend_summary": "",
            "trend_lines": [],
            "signals": [],
            "sources": self._sources_from_items(trend_input.items),
            "validation": {
                "pass": False,
                "reason": "structure_only",
            },
            "metadata": trend_input.metadata,
        }

    def _sources_from_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            sources.append(
                {
                    "index": index,
                    "source_id": item.get("source_id") or item.get("id"),
                    "source_type": item.get("source_type"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                }
            )
        return sources


__all__ = ["ITTrendAgent", "ITTrendInput"]
