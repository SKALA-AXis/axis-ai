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
        max_direction_evidence: int = 10,
        max_execution_evidence: int = 5,
    ) -> None:
        self.max_business_area_evidence = max_business_area_evidence
        self.max_financial_evidence = max_financial_evidence
        self.max_direction_evidence = max_direction_evidence
        self.max_execution_evidence = max_execution_evidence

    def select(self, input_pack: dict[str, Any]) -> dict[str, Any]:
        """Return a compact evidence pack without mutating the original input."""
        selected = {
            "company": dict(input_pack.get("company") or {}),
            "period": input_pack.get("period"),
            "business_area_evidence": self._select_items(
                input_pack.get("business_area_evidence"),
                limit=self.max_business_area_evidence,
                key_fields=("business_area", "claim"),
            ),
            "financial_evidence": self._select_financial(input_pack.get("financial_evidence")),
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
                copied["evidence_text"] = _truncate(str(copied.get("evidence_text") or ""), 360)
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
        "direction_evidence",
        "execution_evidence",
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
