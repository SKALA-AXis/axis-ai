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
