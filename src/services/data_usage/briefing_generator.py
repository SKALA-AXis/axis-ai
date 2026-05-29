"""BriefingGenerator placeholder.

추후 개발: 저장된 카드/인사이트/profile/trend context를 기반으로 briefing을 생성한다.
"""

from __future__ import annotations

from typing import Any


class BriefingGenerator:
    """Planned briefing generation component."""

    status = "planned"
    note = "추후 개발"

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("BriefingGenerator는 추후 개발 예정입니다.")


class BriefingGenerationService(BriefingGenerator):
    """Planned service alias until final naming is decided."""


__all__ = ["BriefingGenerationService", "BriefingGenerator"]
