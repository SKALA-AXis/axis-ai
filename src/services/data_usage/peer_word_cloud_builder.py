"""PeerWordCloudBuilder placeholder.

추후 개발: peer별 키워드/빈도/카테고리를 word cloud artifact로 조립한다.
"""

from __future__ import annotations

from typing import Any


class PeerWordCloudBuilder:
    """Planned peer word-cloud builder."""

    status = "planned"
    note = "추후 개발"

    def build(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("PeerWordCloudBuilder는 추후 개발 예정입니다.")


__all__ = ["PeerWordCloudBuilder"]
