"""Semantic lexicons used by deterministic issue integration.

이 파일의 토큰들은 source_type 단순 매핑이 아니라 여러 입력 신호를 점수화하기
위한 관찰 단서다. 운영 데이터가 쌓이면 이 목록은 샘플 오류 분석 기반으로
교체/확장한다.
"""

from __future__ import annotations

FAMILY_INDICATORS: dict[str, tuple[str, ...]] = {
    "news": (
        "news",
        "newsroom",
        "보도자료",
        "press",
        "article",
        "언론",
    ),
    "filing": (
        "dart",
        "disclosure",
        "filing",
        "공시",
        "사업보고서",
        "분기보고서",
        "반기보고서",
    ),
    "ir": (
        "ir",
        "earnings",
        "presentation",
        "실적발표",
        "investor",
        "가이던스",
    ),
    "research": (
        "securities",
        "analyst",
        "research",
        "증권",
        "리포트",
        "전망치",
        "밸류에이션",
    ),
    "trend": (
        "trend",
        "industry",
        "global_report",
        "spri",
        "bcg",
        "gartner",
        "시장",
        "산업",
        "트렌드",
    ),
}

SCOPE_INDICATORS: dict[str, tuple[str, ...]] = {
    "peer_company": (
        "company",
        "peer",
        "기업",
        "회사",
        "samsung_sds",
        "lg_cns",
        "hyundai_autoever",
        "posco_dx",
    ),
    "industry": (
        "industry",
        "sector",
        "산업",
        "시장",
        "트렌드",
        "수요",
        "기술",
    ),
    "market": (
        "market",
        "valuation",
        "forecast",
        "증권",
        "전망",
        "시장",
    ),
}

FACT_TYPE_INDICATORS: dict[str, tuple[str, ...]] = {
    "financial_metric": (
        "financial",
        "finance",
        "metric",
        "재무",
        "실적",
        "매출",
        "영업이익",
        "ebitda",
        "가이던스",
    ),
    "risk_fact": (
        "risk",
        "uncertain",
        "위험",
        "리스크",
        "규제",
        "소송",
        "보안",
        "침해",
    ),
    "business_signal": (
        "business",
        "segment",
        "strategy",
        "사업",
        "전략",
        "고객",
        "계약",
        "수주",
        "플랫폼",
        "서비스",
    ),
    "market_fact": (
        "market",
        "industry",
        "trend",
        "시장",
        "산업",
        "수요",
        "점유율",
        "전망",
    ),
    "governance_fact": (
        "governance",
        "board",
        "조직",
        "임원",
        "주주",
        "이사회",
    ),
}


__all__ = ["FACT_TYPE_INDICATORS", "FAMILY_INDICATORS", "SCOPE_INDICATORS"]
