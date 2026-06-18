# 작성일: 2026-06-12
# 작성자: 최종민
# 변경이력:
#   2026-06-12 최종민 — briefing_generation_agent 분해 1단계로 support 분리
#   2026-06-14 안가은 — 브리핑 생성 카피와 기간 표기 개선 (#187)
"""support — strategic_insight_agent 에서 분리 (이동만, 동작 불변).

분리 근거: axis-infra docs/structure-tasks/agent-split-design.md (Phase 2 1단계)
"""

import json
import re
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from src.shared.json_helpers import json_dict as _json_dict

KST = ZoneInfo("Asia/Seoul")

_PROMPT_TEXT_LIMIT = 520
_PROMPT_SHORT_TEXT_LIMIT = 220
_PROMPT_LIST_LIMIT = 5


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
        "title": _compact_prompt_text(card.get("title"), max_chars=_PROMPT_SHORT_TEXT_LIMIT),
        "source_raw_article_ids": card.get("source_raw_article_ids") or [],
        "evidence_refs": _compact_evidence_refs(evidence_payload.get("evidence_refs")),
        "quality_flags": _json_list(card.get("quality_flags")),
        "analysis_package": _compact_analysis_package(package),
    }


def _dedupe_cards_for_prompt(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove near-identical card signals before building LLM prompt context."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        key = _prompt_card_dedupe_key(card)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(card)
    return result


def _prompt_card_dedupe_key(card: dict[str, Any]) -> str:
    package = _analysis_package(card)
    integrated = _json_dict(package.get("integrated_issue"))
    analysis = _json_dict(package.get("analysis"))
    text_value = _first_text(
        integrated.get("main_issue"),
        integrated.get("headline"),
        integrated.get("one_line_summary"),
        integrated.get("content_summary"),
        analysis.get("market_signal"),
        analysis.get("analysis_summary"),
        card.get("title"),
    )
    normalized = _normalize_prompt_dedupe_text(text_value)
    if not normalized:
        return ""
    return f"{_company_label(card)}:{normalized[:140]}"


def _normalize_prompt_dedupe_text(value: object) -> str:
    text = " ".join(str(value or "").split()).lower()
    text = re.sub(r"[\W_]+", "", text)
    return text


def _compact_analysis_package(package: dict[str, Any]) -> dict[str, Any]:
    classification = _json_dict(package.get("classification"))
    integrated = _json_dict(package.get("integrated_issue"))
    analysis = _json_dict(package.get("analysis"))
    implication = _json_dict(package.get("implication"))
    validation = _json_dict(package.get("validation"))
    return {
        "bundle_id": package.get("bundle_id"),
        "classification": _compact_prompt_dict(
            classification,
            (
                "main_company",
                "event_type",
                "sector",
                "sectors",
                "mentioned_peer_companies",
                "keywords",
            ),
        ),
        "integrated_issue": _compact_prompt_dict(
            integrated,
            (
                "integrated_issue_id",
                "main_issue",
                "headline",
                "one_line_summary",
                "content_summary",
                "integrated_text",
                "source_article_ids",
            ),
        ),
        "analysis": _compact_prompt_dict(
            analysis,
            (
                "analysis_summary",
                "market_signal",
                "strategic_meaning",
                "key_numbers",
                "confidence",
            ),
        ),
        "implication": _compact_implication_for_prompt(implication),
        "validation": _compact_prompt_dict(
            validation,
            ("pass", "is_valid", "sc_score", "confidence", "reason", "quality_flags"),
        ),
    }


def _compact_implication_for_prompt(implication: dict[str, Any]) -> dict[str, Any]:
    return {
        "peer_implication": _compact_prompt_dict(
            _json_dict(implication.get("peer_implication")),
            (
                "peer_meaning",
                "capability_change",
                "strategy_shift",
                "business_impact",
            ),
        ),
        "skax_implication": _compact_prompt_dict(
            _json_dict(implication.get("skax_implication")),
            (
                "why_important",
                "potential_impact",
                "recommended_actions",
                "risk_factors",
                "opportunity",
            ),
        ),
        "confidence": implication.get("confidence"),
    }


def _compact_prompt_dict(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value in (None, "", [], {}):
            continue
        compact[key] = _compact_prompt_value(value)
    return compact


def _compact_prompt_value(value: Any, *, max_chars: int = _PROMPT_TEXT_LIMIT) -> Any:
    if isinstance(value, str):
        return _compact_prompt_text(value, max_chars=max_chars)
    if isinstance(value, list):
        return [
            _compact_prompt_value(item, max_chars=max_chars)
            for item in value[:_PROMPT_LIST_LIMIT]
            if item not in (None, "", [], {})
        ]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in list(value.items())[:_PROMPT_LIST_LIMIT]:
            if item in (None, "", [], {}):
                continue
            compact[str(key)] = _compact_prompt_value(item, max_chars=max_chars)
        return compact
    return value


def _compact_prompt_text(value: object, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    for marker in (".", "。", "!", "?", "다."):
        index = clipped.rfind(marker)
        if index >= max_chars // 2:
            return clipped[: index + len(marker)].strip()
    return clipped.rsplit(" ", 1)[0].strip()


def _compact_evidence_refs(value: object) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in _json_list(value)[:3]:
        if not isinstance(item, dict):
            continue
        compact = _compact_prompt_dict(
            item,
            ("evidence_ref_id", "text", "source_ids", "source_raw_article_ids"),
        )
        if compact:
            refs.append(compact)
    return refs


def _company_label(card: dict[str, Any]) -> str:
    value = str(card.get("peer_id") or card.get("company") or "").strip()
    labels = {
        "hyundai_autoever": "현대오토에버",
        "lg_cns": "LG CNS",
        "samsung_sds": "삼성SDS",
        "posco_dx": "포스코DX",
        "industry_trend": "산업 동향",
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
