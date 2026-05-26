"""EvidenceBuilder / EvidenceService placeholder.

추후 개발: source links, provenance, financial refs, evidence chain payload를 조립한다.
"""

from __future__ import annotations

from typing import Any


class EvidenceBuilder:
    """Planned evidence payload builder."""

    status = "planned"
    note = "추후 개발"

    def build(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """추후 개발."""
        raise NotImplementedError("EvidenceBuilder는 추후 개발 예정입니다.")


class EvidenceService(EvidenceBuilder):
    """Planned service alias until final naming is decided."""


__all__ = ["EvidenceBuilder", "EvidenceService"]

