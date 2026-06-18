"""card_news schema — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any
from zoneinfo import ZoneInfo

_DISPLAY_ZONE = ZoneInfo("Asia/Seoul")


_PROMPT_VERSION = "card-news-v1.0"


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


_ISSUE_CARD_PROMPT = """\
당신은 SK AX 전략기획팀의 AI 어시스턴트입니다.
아래 기사들을 종합하여 이슈 카드를 작성해주세요.
여러 기사가 있을 경우 교차 검증하여 가장 신뢰도 높은 사실만 포함하세요.

## 기사 정보
{articles_text}

## 작성 규칙
- 제목: 핵심 사실을 담은 한 문장 (40자 이내)
- 요약: 최소 3줄, 최대 5줄. 각 줄은 순번 없이 사실 문장만 작성
- 출처 목록: 사용한 기사의 title, source_name, url 포함

## 이벤트 타입 (하나만 선택, 가장 두드러진 성격 기준)
- partnership: 기업간 협력·MOU·공동사업·파운드리 제공계약
- ma: 인수·합병·지분 인수·투자 유치
- personnel: 채용·인사·임원 선임·조직 개편
- tech: 신기술·신제품·플랫폼 출시·기술 실증
- regulation: 법규·가이드라인·정부 정책
- new_biz: 신사업 진출·수주·실적 발표·시장 확장

복수 해당 시 본문이 가장 크게 다루는 측면을 선택.
애매하면 tech 대신 personnel/new_biz/partnership 우선.

다음 JSON 형식으로만 응답하세요 (추가 텍스트 금지):
{{
  "title": "이슈 제목",
  "summary_lines": ["...", "...", "...", "...", "..."],
  "event_type": "tech",
  "sources": [
    {{"index": 1, "title": "...", "source_name": "...", "url": "...", "credibility_score": 0.0}}
  ]
}}"""


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
