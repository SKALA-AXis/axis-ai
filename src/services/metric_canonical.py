# 작성일: 2026-05-21
# 작성자: 최종민
# 변경이력:
#   2026-05-21 최종민 — Layer B 분석 파이프라인(컨텍스트 엔지니어링·LLM 추론·평가)의 일부로 추가
"""Financial metric_name 정규화 — P3-DATA-2 대응.

`raw_article_financial_metrics.metric_name` 이 동일 metric 인데 한글/영문이 혼재
저장되어 시계열 그룹화 시 데이터가 분산된다. 본 모듈은 단일 출처의 매핑을
제공하고, 모든 SQL VIEW 및 Python ContextBuilder 가 동일 매핑을 사용한다.

§3.4.3 의 `peer_financial_trend` VIEW 의 CASE 매핑과 1:1 동기화.
"""

from __future__ import annotations

from typing import Final

# 동일 metric 의 한글/영문/표기 변형을 canonical key 로 묶는다.
METRIC_CANONICAL: Final[dict[str, str]] = {
    # net income
    "net_income": "net_income",
    "당기순이익": "net_income",
    "순이익": "net_income",
    "net_profit": "net_income",
    "지배주주순이익": "net_income",
    # revenue
    "revenue_total": "revenue_total",
    "revenue": "revenue_total",
    "매출": "revenue_total",
    "매출액": "revenue_total",
    "총매출": "revenue_total",
    # operating profit
    "operating_profit": "operating_profit",
    "영업이익": "operating_profit",
    "operating_income": "operating_profit",
    # operating margin
    "operating_margin": "operating_margin",
    "영업이익률": "operating_margin",
    # gross profit
    "gross_profit": "gross_profit",
    "매출총이익": "gross_profit",
    # ebit / ebitda (best effort — 추후 추가)
    "ebit": "ebit",
    "ebitda": "ebitda",
    # employees
    "employees": "employees",
    "임직원수": "employees",
    "직원수": "employees",
}


def canonicalize_metric(metric_name: str | None) -> str:
    """metric_name 을 canonical key 로 변환. 매핑 없으면 입력 그대로 반환."""
    if not metric_name:
        return ""
    key = metric_name.strip()
    return METRIC_CANONICAL.get(key, key)


def is_canonical_known(metric_name: str | None) -> bool:
    """canonical lookup table 에 등록된 metric 인가."""
    if not metric_name:
        return False
    return metric_name.strip() in METRIC_CANONICAL


__all__ = [
    "METRIC_CANONICAL",
    "canonicalize_metric",
    "is_canonical_known",
]
