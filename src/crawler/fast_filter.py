"""경량 필터 + 중요도 사전 분류기 — LLM 없이 0.1~1ms/건."""

import logging
import re
from dataclasses import dataclass

from src.crawler.base import RawArticle

log = logging.getLogger(__name__)

# ── Peer사 관련성 키워드 ──────────────────────────────────────────
PEER_KEYWORDS: list[str] = [
    "삼성SDS",
    "Samsung SDS",
    "삼성에스디에스",
    "LG CNS",
    "엘지씨엔에스",
    "LGCNS",
]

# ── 중요도 사전 분류: 키워드 점수 ────────────────────────────────
# 1단계: 키워드 스코어링 (0.1ms/건)
_CRITICAL_KEYWORDS: list[str] = [
    "합병",
    "인수",
    "M&A",
    "지분취득",
    "대형이사교체",
    "상장",
    "IPO",
    "분사",
    "매각",
]
_HIGH_KEYWORDS: list[str] = [
    "MOU",
    "파트너십",
    "전략투자",
    "AI",
    "플랫폼 출시",
    "수주",
    "클라우드",
    "에이전틱",
    "생성형",
]
_MEDIUM_KEYWORDS: list[str] = [
    "협력",
    "출시",
    "계약",
    "발표",
    "채용",
    "인사",
]

# 2단계: 정규식 패턴 (1ms/건)
_REGEX_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\d+억\s?원.{0,10}(계약|투자|인수)"), 30),
    (re.compile(r"\d+조\s?원"), 50),
    (re.compile(r"\d+\.?\d*%\s?(지분|주식)"), 30),
    (re.compile(r"(공식|단독|최초|속보)"), 10),
]

# 품질 게이트
MIN_CONTENT_LENGTH = 200
MIN_KOREAN_RATIO = 0.1

_AD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(광고|PR|[Ss]ponsored|promoted)", re.IGNORECASE),
    re.compile(r"(이벤트|경품|할인|쿠폰|프로모션)"),
    re.compile(r"(부동산|아파트|분양|청약)"),
]


@dataclass
class FilterResult:
    article: RawArticle
    score: int
    importance_hint: str  # "urgent" | "notable" | "reference"


class FastFilter:
    """Gate 1: 관련성·품질 필터 + 경량 중요도 사전 분류."""

    def filter(self, articles: list[RawArticle]) -> tuple[list[RawArticle], int]:
        """관련성·품질 필터만 적용 (Track B 배치 경로)."""
        results = self.filter_and_score(articles)
        passed = [r.article for r in results]
        removed = len(articles) - len(passed)
        log.info("FastFilter | total=%d passed=%d removed=%d", len(articles), len(passed), removed)
        return passed, removed

    def filter_and_score(self, articles: list[RawArticle]) -> list[FilterResult]:
        """관련성·품질 필터 + 중요도 점수 계산 (Track A 실시간 경로)."""
        results = []
        for a in articles:
            if not self._is_relevant(a) or not self._is_quality(a):
                continue
            score = self._score(a)
            hint = "urgent" if score >= 60 else "notable" if score >= 30 else "reference"
            results.append(FilterResult(article=a, score=score, importance_hint=hint))
        return results

    # ── 내부 메서드 ────────────────────────────────────────────────

    def _is_relevant(self, article: RawArticle) -> bool:
        text = article.title + " " + article.content
        return any(kw.lower() in text.lower() for kw in PEER_KEYWORDS)

    def _is_quality(self, article: RawArticle) -> bool:
        if len(article.content) < MIN_CONTENT_LENGTH:
            return False
        korean_chars = sum(1 for c in article.content if "가" <= c <= "힣")
        if len(article.content) > 0 and korean_chars / len(article.content) < MIN_KOREAN_RATIO:
            return False
        combined = article.title + article.content
        return not any(p.search(combined) for p in _AD_PATTERNS)

    def _score(self, article: RawArticle) -> int:
        text = article.title + " " + article.content
        score = 0
        for kw in _CRITICAL_KEYWORDS:
            if kw in text:
                score += 50
        for kw in _HIGH_KEYWORDS:
            if kw in text:
                score += 20
        for kw in _MEDIUM_KEYWORDS:
            if kw in text:
                score += 5
        for pattern, points in _REGEX_PATTERNS:
            if pattern.search(text):
                score += points
        # credibility 보너스
        if article.credibility_score >= 0.9:
            score += 10
        return score
