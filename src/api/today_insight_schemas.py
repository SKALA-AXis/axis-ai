# 작성일: 2026-06-05
# 작성자: 박진
# 변경이력:
#   2026-06-05 박진 — 투데이 인사이트 에이전트와 60일 카드뉴스 입력용 스키마 추가 및 코드 포맷 정리, mixer chat today insight 플로우 개선
#   2026-06-09 최종민 — dual-lane 비교 facts 스키마 추가, 공용 스키마를 src/contracts 로 분리(agents→api 의존 절단)
"""호환 shim — 정주소는 src.contracts.today_insight_schemas (2-A1 의존 절단)."""

from src.contracts.today_insight_schemas import (  # noqa: F401
    TodayInsightAction,
    TodayInsightChangeSummary,
    TodayInsightComparisonFacts,
    TodayInsightEvidence,
    TodayInsightGenerateRequest,
    TodayInsightGenerateResponse,
    TodayInsightKeywordTrend,
    TodayInsightMemoryDocument,
    TodayInsightReasoningStep,
    TodayInsightSection,
    TodayInsightSignal,
    TodayInsightSource,
    TodayInsightSourceTrace,
)
