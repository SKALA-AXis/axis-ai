"""전처리 source routing/status 설정."""

from __future__ import annotations

from typing import Final

NEWS_SOURCE_TYPES: Final[set[str]] = {"news"}
OFFICIAL_SOURCE_TYPES: Final[set[str]] = {"official"}
COMPANY_SITE_SOURCE_TYPES: Final[set[str]] = {"company_site"}
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
STATUS_SKIPPED: Final[str] = "SKIPPED"
STATUS_FAILED: Final[str] = "FAILED"
