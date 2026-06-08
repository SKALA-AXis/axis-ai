"""Relevance 평가 전 rule/fallback 정책 설정."""

from __future__ import annotations

import os
from typing import Final

RELEVANCE_THRESHOLD: Final[float] = 0.60
PEER_CONTEXT_LIMIT: Final[int] = 1800
MIN_PEER_MENTIONS: Final[int] = int(os.getenv("RELEVANCE_MIN_PEER_MENTIONS", "1"))

MARKET_PRICE_PATTERN: Final[str] = (
    r"(주가|종가|장중|상승\s*마감|하락\s*마감|강세|약세|급등|급락|상한가|하한가|시가총액)"
)
MARKET_METRIC_PATTERN: Final[str] = r"(\d+(?:\.\d+)?\s*%|\d{1,3}(?:,\d{3})+\s*원)"

EVENT_LISTING_KEYWORDS: Final[list[str]] = [
    "전시회",
    "박람회",
    "컨퍼런스",
    "세미나",
    "포럼",
    "행사",
    "코엑스",
    "개최",
    "참가",
    "총집결",
    "부스",
    "시상식",
]

STRATEGIC_ACTION_KEYWORDS: Final[list[str]] = [
    "수주",
    "계약",
    "구축",
    "선정",
    "우선협상",
    "협약",
    "업무협약",
    "맞손",
    "협력",
    "협업",
    "제휴",
    "공동",
    "공략",
    "확대",
    "정조준",
    "지원",
    "강화",
    "진출",
    "출시",
    "공개",
    "개발",
    "도입",
    "실증",
    "검증",
    "poc",
    "PoC",
    "투자",
    "인수",
    "합병",
    "실적",
    "공시",
    "조직개편",
    "채용",
    # 흔한 출시/제휴/조직 표현이 누락돼 멀쩡한 피어 기사가 탈락하던 케이스 보강.
    "동맹",
    "선보",
    "선봬",
    "출범",
    "체결",
]

STRONG_STRATEGIC_ACTION_KEYWORDS: Final[list[str]] = [
    "수주",
    "계약",
    "구축",
    "우선협상",
    "협약",
    "업무협약",
    "맞손",
    "협력",
    "협업",
    "제휴",
    "공략",
    "확대",
    "정조준",
    "지원",
    "강화",
    "진출",
    "도입",
    "실증",
    "검증",
    "poc",
    "PoC",
    "투자",
    "인수",
    "합병",
    "실적",
    "공시",
    "조직개편",
    "채용",
    # 강한 딜/동맹 신호 — 핵심성 판정(direct role)·fast-pass 에 함께 반영.
    "동맹",
    "체결",
]

DIRECT_COMPANY_ROLE_KEYWORDS: Final[list[str]] = [
    *STRONG_STRATEGIC_ACTION_KEYWORDS,
    "선정",
    "참여",
    "맡",
    "맞손",
    "협력",
    "협업",
    "적용",
    "실증",
    "검증",
    "운영",
    "공급",
    "제공",
    "공략",
    "확대",
    "정조준",
    "지원",
    "강화",
    "진출",
    "등록",
    "할당",
    "도입",
    "확정",
    "추진",
]

FAST_PASS_ACTION_KEYWORDS: Final[list[str]] = [
    *STRONG_STRATEGIC_ACTION_KEYWORDS,
    "출시",
    "공개",
    "선정",
    "개발",
    "공략",
    "확대",
    "정조준",
    "지원",
    "강화",
    "진출",
    "고도화",
    "플랫폼",
    "맞손",
    "협력",
    "협업",
    "도입",
    "실증",
    "검증",
    "poc",
    "PoC",
    # 출시/조직 출범 류 — fast-pass 에서도 LLM 없이 통과 가능하도록 보강.
    "선보",
    "선봬",
    "출범",
]
FAST_PASS_SOURCE_TYPES: Final[set[str]] = {"news"}
ROLE_CONTEXT_WINDOW: Final[int] = 100

LISTING_CONTEXT_KEYWORDS: Final[list[str]] = [
    "etf",
    "펀드",
    "포트폴리오",
    "편입",
    "구성종목",
    "관련주",
    "테마주",
    "수익률",
    "종목",
    "시황",
]

MARKET_LISTING_KEYWORDS: Final[list[str]] = [
    *LISTING_CONTEXT_KEYWORDS,
    "특징주",
    "목표가",
    "투자의견",
    "시가총액",
    "per",
    "주가수익비율",
    "코스피",
    "코스닥",
    "마감시황",
    "증시키워드",
    "기업이슈",
    "레버리지",
    "외국인",
    "기관",
    "순매수",
    "순매도",
    "상승 견인",
    "랠리",
]

LOW_VALUE_NEWS_KEYWORDS: Final[list[str]] = [
    "뉴스브리핑",
    "대학 뉴스브리핑",
    "멘토링",
    "졸업동문",
    "초청 멘토링",
    "박람회 개최",
    "취업 상담",
    "사상 최고 경신",
    "투자유치",
    "하락종목도",
    "사랑의 취업 상담",
]

SUBJECT_MARKERS: Final[list[str]] = ["은", "는", "이", "가"]
