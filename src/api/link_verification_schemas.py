# 작성일: 2026-05-15
# 작성자: 최종민
# 변경이력:
#   2026-05-15 최종민 — LinkVerification prototype 스키마 신설 (HTTP HEAD + GET hash diff)
"""LinkVerification endpoint 의 request / response Pydantic 모델.

design: ``axis-ai/design/30-analysis/link-verification.md`` §4 입력 / §5 출력.

prototype 범위 (Walking Skeleton Phase 2): HTTP HEAD + 옵션 GET hash diff. LLM 미사용,
deterministic. ``link_verification_logs`` 테이블 저장은 Day 90+ 후속 작업.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LinkVerificationRequest(BaseModel):
    """``POST /link/verify`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    card_id: str = Field(..., description="검증 대상 카드 id")


LinkStatusEnum = Literal["live", "dead", "redirected", "error", "live (content_changed)"]
OverallStatusEnum = Literal["all_live", "some_dead", "all_dead", "content_changed"]


class LinkStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    status: LinkStatusEnum
    http_code: Optional[int] = None
    final_url: Optional[str] = None
    content_changed: bool = False
    last_modified: Optional[str] = None
    checked_at: datetime


class LinkVerificationResponse(BaseModel):
    """``POST /link/verify`` 응답. design §5 LinkVerificationOutput schema."""

    model_config = ConfigDict(extra="allow")

    card_id: str
    sources: list[LinkStatus] = Field(default_factory=list)
    overall_status: OverallStatusEnum = "all_dead"
    verified_at: datetime
    warning: Optional[str] = None
