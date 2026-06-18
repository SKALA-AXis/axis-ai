# 작성일: 2026-05-20
# 작성자: 박지원
# 변경이력:
#   2026-05-20 박지원 — 전처리 source routing/status 설정 추가 및 크롤러 catch 처리 보강
"""전처리 source routing/status 설정."""

from __future__ import annotations

from typing import Final

NEWS_SOURCE_TYPES: Final[set[str]] = {"news"}
OFFICIAL_SOURCE_TYPES: Final[set[str]] = {"official"}
COMPANY_SITE_SOURCE_TYPES: Final[set[str]] = {"company_site", "company_analysis"}
PARSED_DOCUMENT_SOURCE_TYPES: Final[set[str]] = {"dart", "ir", "securities_report"}
STRUCTURED_SIGNAL_SOURCE_TYPES: Final[set[str]] = {
    "job",
    "market_data",
    "search_trend",
    "social",
}
INDUSTRY_DOCUMENT_SOURCE_TYPES: Final[set[str]] = {"trend_report"}

METADATA_CHUNK_TEXT_CHARS: Final[int] = 1200
DEFAULT_GPT_WORKERS: Final[int] = 5

STATUS_RAW: Final[str] = "RAW"
STATUS_PROCESSED: Final[str] = "PROCESSED"
STATUS_REVIEW: Final[str] = "REVIEW"
STATUS_SKIPPED: Final[str] = "SKIPPED"
STATUS_FAILED: Final[str] = "FAILED"
