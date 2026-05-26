"""EventChainDiscoveryJob placeholder.

추후 개발: 저장된 card_news와 embedding 유사도를 이용해 후속/반응/연쇄 이벤트 후보를
탐색한다.
"""

from __future__ import annotations

from typing import Any


class EventChainDiscoveryJob:
    """Planned event-chain discovery batch job."""

    status = "planned"
    note = "추후 개발"

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("EventChainDiscoveryJob은 추후 개발 예정입니다.")


class EventChainDiscoveryService(EventChainDiscoveryJob):
    """Planned service alias until final boundary is decided."""


__all__ = ["EventChainDiscoveryJob", "EventChainDiscoveryService"]

