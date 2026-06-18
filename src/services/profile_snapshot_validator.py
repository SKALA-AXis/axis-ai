# 작성일: 2026-05-29
# 작성자: 박지원
# 변경이력:
#   2026-05-29 박지원 — peer 프로필 스냅샷 저장 전 검증 모듈 추가
"""ProfileSnapshotValidator — lightweight profile snapshot validation."""

from __future__ import annotations

from typing import Any


class ProfileSnapshotValidator:
    """Validate generated peer profile snapshots before persistence."""

    required_fields = (
        "company_id",
        "company_name",
        "schema_version",
        "generated_at",
        "business_areas",
        "financial_summary",
        "source_index",
    )

    def validate(
        self,
        snapshot: dict[str, Any],
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        for field in self.required_fields:
            if field not in snapshot:
                errors.append(f"missing required field: {field}")

        if not isinstance(snapshot.get("business_areas"), list):
            errors.append("business_areas must be a list")
        if not isinstance(snapshot.get("financial_summary"), dict):
            errors.append("financial_summary must be an object")
        if "market_view" in snapshot and not isinstance(snapshot.get("market_view"), dict):
            errors.append("market_view must be an object")
        if "source_coverage" in snapshot and not isinstance(snapshot.get("source_coverage"), dict):
            errors.append("source_coverage must be an object")
        if not isinstance(snapshot.get("source_index"), list):
            errors.append("source_index must be a list")

        refs = _collect_refs(snapshot)
        if not refs:
            warnings.append("source_refs are empty")
        if evidence_pack is not None:
            allowed_refs = _allowed_refs(evidence_pack)
            missing = [ref for ref in refs if ref not in allowed_refs]
            if missing:
                warnings.append(f"snapshot contains refs not found in evidence_pack: {missing[:5]}")

        result = {
            "is_valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "checked_items": {
                "required_fields": len(self.required_fields),
                "source_refs": len(refs),
            },
        }
        snapshot["validation"] = result
        return result


def _collect_refs(value: Any) -> set[tuple[str, Any]]:
    refs: set[tuple[str, Any]] = set()
    if isinstance(value, dict):
        if value.get("raw_article_id") is not None:
            refs.add(("raw_articles", value.get("raw_article_id")))
        if "table" in value and "id" in value:
            refs.add((str(value.get("table")), value.get("id")))
        for nested in value.values():
            refs |= _collect_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            refs |= _collect_refs(nested)
    return refs


def _allowed_refs(evidence_pack: dict[str, Any]) -> set[tuple[str, Any]]:
    refs: set[tuple[str, Any]] = set()
    for bucket in (
        "business_area_evidence",
        "financial_evidence",
        "operational_evidence",
        "direction_evidence",
        "execution_evidence",
        "evolution_evidence",
        "market_evidence",
        "source_index",
    ):
        for item in evidence_pack.get(bucket) or []:
            if not isinstance(item, dict):
                continue
            ref = item.get("source_ref", item)
            if isinstance(ref, dict) and ref.get("table") and "id" in ref:
                refs.add((str(ref.get("table")), ref.get("id")))
                if ref.get("raw_article_id") is not None:
                    refs.add(("raw_articles", ref.get("raw_article_id")))
    return refs


__all__ = ["ProfileSnapshotValidator"]
