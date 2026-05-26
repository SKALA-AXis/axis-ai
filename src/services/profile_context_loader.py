"""Runtime ProfileContext loader.

ProfileAgent는 회사별 profile snapshot을 생성하는 Agent이고, ProfileContextLoader는
AnalysisGraphRunner 실행 중 이미 저장된 snapshot과 최근 신호를 읽어
ProfileContext를 조립하는 Service다.
"""

from __future__ import annotations

from typing import Any

from src.analysis.models import AnalysisInputBundle, ProfileContext
from src.services.profile_context_v2 import build_profile_context_v2


class ProfileContextLoader:
    """Build ProfileContext for a single AnalysisGraphRunner invocation."""

    def __init__(self, *, lookback_days: int = 30) -> None:
        self.lookback_days = lookback_days

    def load(
        self,
        *,
        companies: list[str],
        sectors: list[str] | None = None,
        event_type: str | None = None,
        integrated_issue: dict[str, Any] | None = None,
        input_bundle: AnalysisInputBundle | None = None,
        peer_profile_context: dict[str, Any] | None = None,
        skax_profile_context: dict[str, Any] | None = None,
    ) -> ProfileContext:
        """Load compact runtime context without creating a new profile snapshot."""
        del integrated_issue, input_bundle
        return build_profile_context_v2(
            companies=companies,
            sectors=sectors,
            event_type=event_type,
            lookback_days=self.lookback_days,
            peer_profile_context=peer_profile_context,
            skax_profile_context=skax_profile_context,
        )


__all__ = ["ProfileContextLoader", "build_profile_context_v2"]
