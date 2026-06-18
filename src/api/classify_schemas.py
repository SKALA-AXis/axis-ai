# 작성일: 2026-06-15
# 작성자: 최종민
# 변경이력:
#   2026-06-15 최종민 — 단일 기사 텍스트 분류 엔드포인트의 request/response 스키마 추가 (운영
"""Classify endpoint 의 request / response Pydantic 모델.

``POST /classify`` — 단일 기사 텍스트를 운영 수집 파이프라인과 동일한 로직
(``src.preprocessing.classification.classify_article_text``) 으로 분류한다.
DB 미접근, rule 우선 + GPT-4o fallback. backend 데모(``/api/demo/publish``)가 호출해
event_type·중요도를 받아 알림 게이트에 넣는다 — '서비스가 스스로 분류' 를 시연하기 위함.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    """``POST /classify`` 요청 body."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., description="기사 제목")
    content: str = Field("", description="기사 본문")
    company: str = Field("", description="peer id (예: lg_cns) — 회사 직접 언급 카운트용")
    source_type: str = Field("", description="출처 유형(dart/ir/official 등, 선택) — impact 보정")


class ClassifyResponse(BaseModel):
    """``POST /classify`` 응답 — 운영 분류 결과와 동일 필드."""

    model_config = ConfigDict(extra="allow")

    event_type: str
    sector: str
    sectors: list[str] = Field(default_factory=list)
    exposure_score: float
    exposure_band: str
    impact_score: float
    impact_band: str
    importance_score: float
    importance: str
    reasoning: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)
