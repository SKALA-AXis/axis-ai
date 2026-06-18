# 작성일: 2026-05-29
# 작성자: 박지원
# 변경이력:
#   2026-05-29 박지원 — peer 프로필 스냅샷 파이프라인의 evidence 선별 모듈 추가
"""ProfileEvidenceSelector — trim and normalize profile evidence packs."""

from __future__ import annotations

from typing import Any

from src.services.profile_input_builder import normalize_business_area


class ProfileEvidenceSelector:
    """Select the small, high-signal evidence set passed to the LLM summarizer."""

    def __init__(
        self,
        *,
        max_business_area_evidence: int = 6,
        max_financial_evidence: int = 0,
        max_direction_evidence: int = 24,
        max_execution_evidence: int = 12,
        max_evolution_evidence: int = 40,
        max_market_evidence: int = 24,
    ) -> None:
        self.max_business_area_evidence = max_business_area_evidence
        self.max_financial_evidence = max_financial_evidence
        self.max_direction_evidence = max_direction_evidence
        self.max_execution_evidence = max_execution_evidence
        self.max_evolution_evidence = max_evolution_evidence
        self.max_market_evidence = max_market_evidence

    def select(self, input_pack: dict[str, Any]) -> dict[str, Any]:
        """Return a compact evidence pack without mutating the original input."""
        selected = {
            "company": dict(input_pack.get("company") or {}),
            "period": input_pack.get("period"),
            "source_coverage": dict(input_pack.get("source_coverage") or {}),
            "evidence_digest": list(input_pack.get("evidence_digest") or []),
            "business_area_evidence": self._select_items(
                input_pack.get("business_area_evidence"),
                limit=self.max_business_area_evidence,
                key_fields=("business_area", "claim"),
            ),
            "financial_evidence": self._select_financial(input_pack.get("financial_evidence")),
            "operational_evidence": self._select_items(
                input_pack.get("operational_evidence"),
                limit=24,
                key_fields=("business_area", "metric", "period"),
            ),
            "direction_evidence": self._select_items(
                input_pack.get("direction_evidence"),
                limit=self.max_direction_evidence,
                key_fields=("business_area", "claim", "signal_type"),
            ),
            "execution_evidence": self._select_items(
                input_pack.get("execution_evidence"),
                limit=self.max_execution_evidence,
                key_fields=("business_area", "claim"),
            ),
            "evolution_evidence": self._select_evolution_items(
                input_pack.get("evolution_evidence"),
                limit=self.max_evolution_evidence,
            ),
            "market_evidence": self._select_market_items(
                input_pack.get("market_evidence"),
                limit=self.max_market_evidence,
            ),
        }
        selected["source_index"] = _source_index_from_selected(selected)
        return selected

    def _select_financial(self, items: Any) -> list[dict[str, Any]]:
        if self.max_financial_evidence <= 0:
            return []
        normalized = self._normalize_items(items)
        deduped = _dedupe(normalized, key_fields=("business_area", "metric", "metric_scope"))

        company_priority = ("revenue_total", "operating_profit", "operating_margin", "net_income")
        segment_order = ("클라우드&AI", "스마트 엔지니어링", "Digital Business Service")

        company_total = [
            item
            for item in deduped
            if item.get("metric_scope") == "company_total"
            and item.get("metric") in company_priority
        ]
        company_total.sort(
            key=lambda item: company_priority.index(str(item.get("metric") or "net_income"))
        )

        segment_revenue = [
            item
            for item in deduped
            if item.get("metric_scope") == "segment" and item.get("metric") == "revenue_total"
        ]
        segment_revenue.sort(
            key=lambda item: (
                segment_order.index(str(item.get("business_area")))
                if item.get("business_area") in segment_order
                else len(segment_order),
                str(item.get("business_area") or ""),
            )
        )

        selected = [*company_total[:2], *segment_revenue[:3]]
        return selected[: self.max_financial_evidence]

    def _select_items(
        self,
        items: Any,
        *,
        limit: int,
        key_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        normalized = self._normalize_items(items)
        normalized.sort(key=lambda item: _confidence_sort(item), reverse=True)
        return _dedupe(normalized, key_fields=key_fields)[:limit]

    def _select_evolution_items(self, items: Any, *, limit: int) -> list[dict[str, Any]]:
        normalized = self._normalize_items(items)
        deduped = _dedupe(
            normalized,
            key_fields=("period", "business_area", "claim", "evidence_kind"),
        )
        ir_items = [item for item in deduped if item.get("evidence_kind") == "ir_signal"]
        official_items = [
            item for item in deduped if item.get("evidence_kind") == "official_execution"
        ]
        other_items = [
            item
            for item in deduped
            if item.get("evidence_kind") not in {"ir_signal", "official_execution"}
        ]

        ir_limit = max(1, int(limit * 0.65))
        official_limit = max(0, limit - ir_limit)

        selected = [
            *_balanced_by_period_and_area(ir_items, limit=ir_limit),
            *_balanced_by_period_and_area(official_items, limit=official_limit),
        ]
        if len(selected) < limit:
            selected.extend(
                item for item in _balanced_by_period_and_area(other_items, limit=limit) if item
            )
        return selected[:limit]

    def _select_market_items(self, items: Any, *, limit: int) -> list[dict[str, Any]]:
        normalized = self._normalize_items(items)
        deduped = _dedupe(
            normalized,
            key_fields=("evidence_kind", "signal_type", "metric", "claim", "value"),
        )
        metric_items = [
            item for item in deduped if item.get("evidence_kind") == "securities_report_metric"
        ]
        signal_items = [
            item for item in deduped if item.get("evidence_kind") != "securities_report_metric"
        ]
        priority = {
            "risk": 0,
            "valuation": 1,
            "forecast": 2,
            "orders_pipeline": 3,
            "growth": 4,
            "strategy": 5,
        }
        signal_items.sort(
            key=lambda item: (
                priority.get(str(item.get("signal_type") or ""), 9),
                -_confidence_sort(item),
            ),
            reverse=False,
        )
        metric_limit = min(3, max(0, limit // 4))
        selected = [*metric_items[:metric_limit], *signal_items[: max(0, limit - metric_limit)]]
        return selected[:limit]

    def _normalize_items(self, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            if "business_area" in copied:
                copied["business_area"] = normalize_business_area(copied.get("business_area"))
            if "evidence_text" in copied:
                copied["evidence_text"] = _truncate(str(copied.get("evidence_text") or ""), 900)
            out.append(copied)
        return out


def _dedupe(items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _confidence_sort(item: dict[str, Any]) -> float:
    value = item.get("confidence")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _balanced_by_period_and_area(
    items: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    sorted_items = sorted(items, key=_evolution_sort_key, reverse=True)
    selected: list[dict[str, Any]] = []
    period_area_seen: set[tuple[str, str]] = set()
    for item in sorted_items:
        key = (str(item.get("period") or ""), str(item.get("business_area") or ""))
        if key in period_area_seen:
            continue
        selected.append(item)
        period_area_seen.add(key)
        if len(selected) >= limit:
            return selected
    for item in sorted_items:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _evolution_sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
    return (*_period_sort_key(str(item.get("period") or "")), _confidence_sort(item))


def _period_sort_key(period: str) -> tuple[int, int]:
    if len(period) == 6 and period[4] == "Q":
        try:
            return (int(period[:4]), int(period[5]) * 3)
        except ValueError:
            return (0, 0)
    if len(period) >= 7 and period[4] == "-":
        try:
            return (int(period[:4]), int(period[5:7]))
        except ValueError:
            return (0, 0)
    return (0, 0)


def _truncate(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _source_index_from_selected(selected: dict[str, Any]) -> list[dict[str, Any]]:
    refs: dict[tuple[str, Any], dict[str, Any]] = {}
    for bucket in (
        "business_area_evidence",
        "financial_evidence",
        "operational_evidence",
        "direction_evidence",
        "execution_evidence",
        "evolution_evidence",
        "market_evidence",
    ):
        for item in selected.get(bucket) or []:
            ref = item.get("source_ref") if isinstance(item, dict) else None
            if not isinstance(ref, dict):
                continue
            key = (str(ref.get("table") or ""), ref.get("id"))
            if not key[0]:
                continue
            refs.setdefault(key, dict(ref))
    return list(refs.values())


__all__ = ["ProfileEvidenceSelector"]
