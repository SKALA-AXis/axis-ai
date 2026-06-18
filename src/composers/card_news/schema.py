"""card_news schema — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any
from zoneinfo import ZoneInfo

_DISPLAY_ZONE = ZoneInfo("Asia/Seoul")


_CARD_PROMPT_VERSION = "card-news-v1.0"


_DEFAULT_COVER_IMAGE_URL = "/png.png"


_DEFAULT_COVER_IMAGE_ALT = "카드뉴스 대표 이미지"


_SUMMARY_LINE_MIN = 3


_SUMMARY_LINE_MAX = 5


_CARD_DETAIL_MAX = 3


_DISPLAY_ITEM_MIN = 1


_DISPLAY_ITEM_PREFERRED = 3


_DISPLAY_ITEM_MAX = 5


_DISPLAY_TRUNCATED_MARKER_PATTERN = re.compile(r"\.\.\.|…")


_FRONTEND_SECTOR_IDS = {
    "security",
    "ax",
    "infra",
    "biz_area",
    "other",
}


_FRONTEND_EVENT_TYPES = {
    "launch",
    "partnership",
    "contract",
    "technology_update",
    "earnings",
    "stock_market",
    "analyst_report",
    "risk",
    "general_update",
    "investment",
    "hiring",
    "organization",
    "ma",
    "personnel",
    "tech",
    "regulation",
    "new_biz",
}


_FACT_BASIS_EVIDENCE_TYPES = {
    "core_fact",
    "unique_fact",
    "common_fact",
    "uncertain_fact",
    "reported_fact",
    "numeric_fact",
    "market_reaction_fact",
    "risk_fact",
}


_SUMMARY_ACTION_TOKENS = {
    "선정",
    "확정",
    "수주",
    "계약",
    "체결",
    "협약",
    "출시",
    "공개",
    "구축",
    "운영",
    "도입",
    "전환",
    "투자",
    "참여",
}


_EVENT_TO_FACT_BASIS_TYPE = {
    "launch": "unique_fact",
    "technology_update": "unique_fact",
    "contract": "core_fact",
    "partnership": "core_fact",
    "investment": "core_fact",
    "hiring": "reported_fact",
    "organization": "reported_fact",
    "regulation": "reported_fact",
    "general_update": "reported_fact",
    "earnings": "numeric_fact",
    "analyst_report": "reported_fact",
    "stock_market": "market_reaction_fact",
    "risk": "risk_fact",
    "unknown": "reported_fact",
}


def _attach_card_news_schema_fields(card: dict[str, Any]) -> None:
    """card_news 테이블 저장 스키마에 맞는 필드를 카드 dict에 붙인다."""
    implication = {
        "sector": card.get("sector", "other"),
        "sectors": card.get("sectors", ["other"]),
        "exposure_score": card.get("exposure_score", 0.0),
        "exposure_band": card.get("exposure_band", "low"),
        "signals": card.get("signals", {}),
        "evidence_chain": card.get("evidence_chain", {}),
    }
    raw_validation = card.get("validation")
    validation: dict[str, Any] = raw_validation if isinstance(raw_validation, dict) else {}
    validation_pass = bool(validation.get("pass", card.get("validation_pass", False)))
    validation_sc_score = validation.get("sc_score", card.get("validation_sc_score", 0.0))
    card.update(
        {
            "implication": implication,
            "validation_pass": validation_pass,
            "validation_sc_score": validation_sc_score,
            "validation": {
                "pass": validation_pass,
                "sc_score": validation_sc_score,
            },
        }
    )
    card["db_record"] = {
        "id": card.get("id"),
        "company": card.get("company"),
        "cluster_id": card.get("cluster_id"),
        "published_date": card.get("published_date"),
        "title": card.get("title"),
        "summary_lines": card.get("summary_lines", []),
        "event_type": card.get("event_type", "tech"),
        "importance": card.get("importance", "low"),
        "importance_score": card.get("importance_score", 0.0),
        "implication": implication,
        "sources": card.get("sources", []),
        "validation_pass": validation_pass,
        "validation_sc_score": validation_sc_score,
        "image_assets": card.get("image_assets", []),
    }
