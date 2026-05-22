"""IT trend context agent structure.

ITTrendAgent는 카드뉴스 생성 에이전트가 아니다. IT 트렌드 입력은
SPRi / BCG 리서치 자료와 글로벌 회사 뉴스룸의 분석 결과로 나뉘며, 이 둘을
바탕으로 AnalysisAgent가 참고할 TrendContext를 생성·갱신한다.

글로벌 회사별 뉴스룸은 이 에이전트가 직접 카드뉴스로 만들지 않는다. 뉴스룸은
일반 이슈처럼 수집 → 통합 → 분석 → 시사점 → 카드뉴스 생성 흐름을 먼저 타고,
그 결과인 IntegratedIssue / AnalysisResult를 글로벌 실행 신호로 참고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.analysis.models import TrendContext

_TREND_SOURCE_NAMES = {"spri", "bcg"}
_GLOBAL_NEWSROOM_SOURCE_TYPES = {"global_newsroom", "company_newsroom"}


@dataclass
class ITTrendInput:
    """Runtime DTO for IT trend context generation."""

    trend_items: list[dict[str, Any]]
    period: str | None = None
    source_groups: list[str] = field(default_factory=list)
    previous_trend_context: dict[str, Any] | None = None
    reference_issue_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for callers that still read `.items`."""
        return self.trend_items


class ITTrendAgent:
    """Generate TrendContext from research trends and global execution signals."""

    prompt_version = "it-trend-structure-v0"

    def build_input(
        self,
        *,
        items: list[dict[str, Any]],
        period: str | None = None,
        source_groups: list[str] | None = None,
        previous_trend_context: dict[str, Any] | None = None,
        reference_issue_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ITTrendInput:
        trend_items, candidate_reference_items, unsupported_items = _split_trend_inputs(items or [])
        reference_results = list(reference_issue_results or [])
        reference_results.extend(_reference_issue_results_from_items(candidate_reference_items))
        normalized_groups = _source_groups(
            explicit=source_groups,
            trend_items=trend_items,
            reference_results=reference_results,
        )
        return ITTrendInput(
            trend_items=trend_items,
            period=period,
            source_groups=normalized_groups,
            previous_trend_context=dict(previous_trend_context or {}),
            reference_issue_results=reference_results,
            metadata={
                **dict(metadata or {}),
                "research_source_count": len(trend_items),
                "global_newsroom_reference_count": len(reference_results),
                "unsupported_input_count": len(unsupported_items),
                "unsupported_input_reason": (
                    "trend_context_accepts_spri_bcg_and_global_newsroom_analysis_results"
                    if unsupported_items
                    else ""
                ),
            },
        )

    def generate(self, trend_input: ITTrendInput) -> dict[str, Any]:
        """Return TrendContext-shaped output until detailed trend logic is added."""
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        context = TrendContext(
            period=trend_input.period,
            trend_summary="",
            trend_lines=[],
            signals=[],
            source_groups=trend_input.source_groups,
            sources=self._sources_from_items(trend_input.trend_items),
            reference_issue_ids=_reference_issue_ids(trend_input.reference_issue_results),
            updated_at=generated_at,
            validation={
                "pass": False,
                "reason": "structure_only",
                "trend_source_policy": (
                    "SPRi/BCG research sources + global newsroom IntegratedIssue/AnalysisResult"
                ),
            },
            metadata={
                **trend_input.metadata,
                "previous_trend_context_provided": bool(trend_input.previous_trend_context),
                "reference_issue_result_count": len(trend_input.reference_issue_results),
                "reference_issue_policy": (
                    "global_newsroom execution signals must be IntegratedIssue/AnalysisResult, "
                    "not raw newsroom text or card_news presentation payload"
                ),
            },
        )
        return {
            "agent": type(self).__name__,
            "prompt_version": self.prompt_version,
            "generated_at": generated_at,
            "period": context.period,
            "source_groups": context.source_groups,
            "source_count": len(trend_input.trend_items),
            "trend_summary": context.trend_summary,
            "trend_lines": context.trend_lines,
            "signals": context.signals,
            "sources": context.sources,
            "reference_issue_ids": context.reference_issue_ids,
            "trend_context": context.to_dict(),
            "validation": context.validation,
            "metadata": context.metadata,
        }

    def _sources_from_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            sources.append(
                {
                    "index": index,
                    "source_id": item.get("source_id") or item.get("id"),
                    "source_type": item.get("source_type"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                }
            )
        return sources


def _split_trend_inputs(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trend_items: list[dict[str, Any]] = []
    candidate_reference_items: list[dict[str, Any]] = []
    unsupported_items: list[dict[str, Any]] = []
    for item in items:
        if _is_spri_bcg_trend_item(item):
            trend_items.append(dict(item))
        elif _is_global_newsroom_reference_candidate(item):
            candidate_reference_items.append(dict(item))
        else:
            unsupported_items.append(dict(item))
    return trend_items, candidate_reference_items, unsupported_items


def _is_spri_bcg_trend_item(item: dict[str, Any]) -> bool:
    source_name = str(item.get("source_name") or item.get("publisher") or "").strip().lower()
    source_type = str(item.get("source_type") or "").strip().lower()
    return source_type == "trend_report" and source_name in _TREND_SOURCE_NAMES


def _is_global_newsroom_reference_candidate(item: dict[str, Any]) -> bool:
    source_type = str(item.get("source_type") or "").strip().lower()
    return source_type in _GLOBAL_NEWSROOM_SOURCE_TYPES


def _reference_issue_results_from_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for item in items:
        source_type = str(item.get("source_type") or "").strip().lower()
        if source_type not in _GLOBAL_NEWSROOM_SOURCE_TYPES:
            continue
        integrated_issue = item.get("integrated_issue")
        analysis_result = item.get("analysis") or item.get("analysis_result")
        if not isinstance(integrated_issue, dict) and not isinstance(analysis_result, dict):
            continue
        references.append(
            {
                "source_id": item.get("source_id") or item.get("id"),
                "source_type": source_type,
                "integrated_issue": integrated_issue if isinstance(integrated_issue, dict) else {},
                "analysis_result": analysis_result if isinstance(analysis_result, dict) else {},
            }
        )
    return references


def _source_groups(
    *,
    explicit: list[str] | None,
    trend_items: list[dict[str, Any]],
    reference_results: list[dict[str, Any]],
) -> list[str]:
    groups = [str(item) for item in (explicit or []) if item]
    groups.extend(
        str(item.get("source_name") or item.get("publisher") or "").strip()
        for item in trend_items
        if item.get("source_name") or item.get("publisher")
    )
    if reference_results:
        groups.append("global_newsroom_references")
    return _dedupe_strings(groups)


def _reference_issue_ids(reference_issue_results: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in reference_issue_results:
        integrated = item.get("integrated_issue") if isinstance(item, dict) else {}
        analysis = item.get("analysis_result") if isinstance(item, dict) else {}
        for source in (item, integrated, analysis):
            if not isinstance(source, dict):
                continue
            value = (
                source.get("bundle_id")
                or source.get("cluster_id")
                or source.get("source_id")
                or source.get("id")
            )
            if value:
                values.append(str(value))
                break
    return _dedupe_strings(values)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = ["ITTrendAgent", "ITTrendInput"]
