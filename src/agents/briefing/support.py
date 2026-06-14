"""support — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

import json
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from src.shared.json_helpers import json_dict as _json_dict

KST = ZoneInfo("Asia/Seoul")


def _safe_float(value: object, *, default: float) -> float:
    if not isinstance(value, (str, int, float)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_analysis_unit_for_display(card: dict[str, Any]) -> dict[str, Any]:
    package = _analysis_package(card)
    evidence_payload = _json_dict(card.get("evidence_payload"))
    return {
        "integrated_issue_id": card.get("integrated_issue_id")
        or evidence_payload.get("integrated_issue_id")
        or package.get("integrated_issue_id"),
        "card_id": card.get("card_id") or card.get("id"),
        "company": card.get("company"),
        "peer_id": card.get("peer_id"),
        "company_label": _company_label(card),
        "title": card.get("title"),
        "source_raw_article_ids": card.get("source_raw_article_ids") or [],
        "evidence_refs": _json_list(evidence_payload.get("evidence_refs"))[:8],
        "quality_flags": _json_list(card.get("quality_flags")),
        "analysis_package": _compact_analysis_package(package),
    }


def _compact_analysis_package(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": package.get("bundle_id"),
        "classification": _json_dict(package.get("classification")),
        "integrated_issue": _json_dict(package.get("integrated_issue")),
        "analysis": _json_dict(package.get("analysis")),
        "implication": _json_dict(package.get("implication")),
        "validation": _json_dict(package.get("validation")),
    }


def _company_label(card: dict[str, Any]) -> str:
    value = str(card.get("peer_id") or card.get("company") or "").strip()
    labels = {
        "hyundai_autoever": "현대오토에버",
        "lg_cns": "LG CNS",
        "samsung_sds": "삼성SDS",
        "posco_dx": "포스코DX",
    }
    return labels.get(value, value)


def _analysis_package(card: dict[str, Any]) -> dict[str, Any]:
    return _analysis_package_from_sources(card, _json_dict(card.get("evidence_payload")))


def _analysis_package_from_sources(*sources: object) -> dict[str, Any]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        package = source.get("analysis_package")
        if isinstance(package, dict):
            return package
    return {}


def _nested_get(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _str_values(value: object) -> list[str]:
    values = _json_list(value)
    out: list[str] = []
    for item in values:
        text_value = str(item or "").strip()
        if text_value and text_value not in out:
            out.append(text_value)
    return out


def _int_list(value: object) -> list[int]:
    out: list[int] = []
    for item in _json_list(value):
        parsed = _optional_int(item)
        if parsed is not None and parsed not in out:
            out.append(parsed)
    return out


def _first_int(value: object) -> int | None:
    values = _int_list(value)
    return values[0] if values else None


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_from_list(value: object) -> str:
    values = _json_list(value)
    return str(values[0]) if values else ""


def _first_text(*values: object) -> str:
    for value in values:
        text_value = str(value or "").strip()
        if text_value:
            return text_value
    return ""


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(KST).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    return str(value)
