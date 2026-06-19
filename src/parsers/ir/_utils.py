"""ir _utils — extracted from facade (move-only)."""

from __future__ import annotations

import re
from typing import Any

from src.parsers.ir._constants import (  # noqa: F401
    _BACKLOG_PATTERNS,
    _CAPEX_PATTERNS,
    _EBITDA_PATTERNS,
    _IR_CHUNK_MAX_CHARS,
    _IR_CHUNK_MIN_CHARS,
    _IR_CHUNK_OVERLAP_CHARS,
    _IR_CHUNK_TARGET_CHARS,
    _IR_COMPANY_SECTION_HINTS,
    _IR_COMPANY_TOTAL_TERMS,
    _IR_FINANCIAL_METRIC_RULES,
    _IR_LOW_VALUE_PATTERNS,
    _IR_PORTFOLIO_TERMS,
    _IR_SECTION_RULES,
    _IR_SEGMENT_CONTEXT_RULES,
    _IR_SIGNAL_TERMS,
    _IR_TABLE_COMPARISON_PATTERN,
    _IR_TABLE_METRIC_ALIASES,
    _IR_TABLE_NOISE_LABEL_PATTERN,
    _IR_TABLE_NON_METRIC_LABEL_PATTERN,
    _IR_TABLE_NUMBER_PATTERN,
    _IR_TABLE_PERIOD_PATTERN,
    _NET_INCOME_PATTERNS,
    _OPERATING_MARGIN_PATTERNS,
    _OPERATING_PROFIT_PATTERNS,
    _ORDERS_PATTERNS,
    _PAGE_SPLIT_PATTERN,
    _PERIOD_PATTERNS,
    _REVENUE_PATTERNS,
    _SK_AX_PAGE_BUSINESS_TERMS,
    _SK_AX_PAGE_EXCLUDE_TERMS,
    _SK_AX_PAGE_STRONG_TERMS,
)


def _normalize_amount_krwbn(value: str, unit: str) -> float:
    """금액 문자열을 기존 peer_financials 관례인 억원 단위 값으로 변환한다."""
    n = float(value.replace(",", ""))
    if unit in ("조원", "조"):
        return n * 10_000
    return n


def _extract_period(text: str) -> str | None:
    year_match = re.search(r"(20\d{2})\s*년\s*(?:경영\s*)?실적", text or "")
    if year_match:
        return year_match.group(1)

    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue

        if len(match.group(1)) == 1 and len(match.group(2)) == 2:
            return f"20{match.group(2)}Q{match.group(1)}"
        if len(match.group(1)) == 2 and len(match.group(2)) == 1:
            return f"20{match.group(1)}Q{match.group(2)}"

        return f"{match.group(1)}Q{match.group(2)}"
    return None


def _period_parts(period: str | None) -> tuple[int | None, int | None, str | None]:
    if not period:
        return None, None, None

    match = re.match(r"^(20\d{2})Q([1-4])$", period)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return year, quarter, "quarter"

    match = re.match(r"^(20\d{2})$", period)
    if match:
        return int(match.group(1)), None, "year"

    return None, None, None


def _first_amount(
    text: str,
    patterns: list[re.Pattern[str]],
) -> tuple[float, str] | tuple[None, None]:
    for pattern in patterns:
        match = pattern.search(text or "")
        if match:
            return _normalize_amount_krwbn(match.group(1), match.group(2)), match.group(0)

    return None, None


def _normalize_percentage(value: str) -> float:
    return float(value.replace(",", ""))


def _metric_values(
    text: str,
    patterns: list[re.Pattern[str]],
    value_kind: str,
) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    seen: set[tuple[str, float]] = set()
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            if value_kind == "percentage":
                value = _normalize_percentage(match.group(1))
            else:
                value = _normalize_amount_krwbn(match.group(1), match.group(2))
            raw = match.group(0)
            dedupe_key = (raw, value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append((value, raw))

    return values


def _first_metric_value(
    text: str,
    patterns: list[re.Pattern[str]],
    value_kind: str,
) -> tuple[float, str] | tuple[None, None]:
    values = _metric_values(text, patterns, value_kind)
    if not values:
        return None, None
    return values[0]


def _candidate_value_key(value_kind: str) -> str:
    if value_kind == "percentage":
        return "value_pct"
    return "value_krwbn"


def _signed_percentage(raw_value: str, sign_token: str | None = None) -> float | None:
    try:
        value = float(str(raw_value).replace(",", "").rstrip("%"))
    except (TypeError, ValueError):
        return None
    sign = str(sign_token or "").strip()
    if sign in {"-", "△", "▲"}:
        value *= -1
    return value


def _inline_comparison_candidates(
    *,
    page_text: str,
    page_no: Any,
    metric_type: str,
    raw: str,
    base_candidate: dict[str, Any],
    report_period: str | None,
) -> list[dict[str, Any]]:
    if not raw or not report_period:
        return []

    compact_text = " ".join((page_text or "").split())
    raw_index = compact_text.find(raw)
    if raw_index < 0:
        return []

    window = compact_text[raw_index : raw_index + len(raw) + 80]
    candidates: list[dict[str, Any]] = []
    for comparison_type in ("yoy", "qoq"):
        label_pattern = (
            "YoY|전년\\s*(?:동기\\s*)?대비" if comparison_type == "yoy" else "QoQ|전분기\\s*대비"
        )
        match = re.search(
            rf"(?:{label_pattern})\s*([+\-△▲]?)\s*([\d,]+(?:\.\d+)?)\s*%",
            window,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        value = _signed_percentage(match.group(2), match.group(1))
        if value is None:
            continue
        year, quarter, period_type = _period_parts(report_period)
        candidates.append(
            {
                **base_candidate,
                "page": page_no,
                "type": f"{metric_type}_{comparison_type}",
                "base_metric_type": metric_type,
                "comparison_type": comparison_type,
                "value_kind": "percentage",
                "value_pct": value,
                "unit": "%",
                "raw": f"{raw} {match.group(0)}",
                "source": "ir_inline_comparison",
                "period": report_period,
                "period_year": year,
                "period_quarter": quarter,
                "period_type": period_type,
                "confidence": min(float(base_candidate.get("confidence") or 0.8), 0.86),
                "evidence_text": _metric_context_window(page_text, f"{raw} {match.group(0)}"),
            }
        )
    return candidates


def _has_same_table_metric_candidate(
    candidates: list[dict[str, Any]],
    *,
    page_no: Any,
    metric_type: str,
    value_key: str,
    value: float,
) -> bool:
    for candidate in candidates:
        if candidate.get("source") != "ir_table_matrix":
            continue
        if candidate.get("page") != page_no or candidate.get("type") != metric_type:
            continue
        candidate_value = candidate.get(value_key)
        if (
            isinstance(candidate_value, int | float)
            and abs(float(candidate_value) - value) < 0.0001
        ):
            return True
    return False


def _classify_metric_context(
    page_text: str,
    *,
    peer_id: str | None,
    raw_match: str | None = None,
) -> dict[str, Any]:
    text = " ".join((page_text or "").split()).lower()
    metric_window = _metric_context_window(page_text, raw_match)
    current_line_text = _metric_current_line_text(page_text, raw_match)
    metric_text = _metric_context_text(page_text, raw_match)
    has_total = any(term.lower() in text for term in _IR_COMPANY_TOTAL_TERMS)
    portfolio_entity = (
        _detect_portfolio_entity(metric_window)
        or _detect_portfolio_entity(current_line_text)
        or _detect_portfolio_entity(metric_text)
    )
    metric_segment_area = (
        _detect_business_area(current_line_text)
        or _detect_business_area(metric_window)
        or _detect_business_area(metric_text)
    )
    segment_area = metric_segment_area or _detect_business_area(text)

    if portfolio_entity:
        return {
            "metric_scope": "portfolio_company",
            "business_area": "portfolio",
            "entity_name": portfolio_entity,
            "confidence": 0.78,
            "classification_reason": (
                f"metric 주변 문맥에서 SK 포트폴리오 회사 '{portfolio_entity}'가 확인되어 "
                "SK AX/피어 본체 지표가 아닌 portfolio_company로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if metric_segment_area:
        return {
            "metric_scope": "segment",
            "business_area": metric_segment_area,
            "entity_name": peer_id,
            "confidence": 0.78,
            "classification_reason": (
                f"metric가 있는 행/주변 문맥에서 '{metric_segment_area}' 사업 키워드가 "
                "직접 확인되어 segment 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if segment_area and not has_total:
        return {
            "metric_scope": "segment",
            "business_area": segment_area,
            "entity_name": peer_id,
            "confidence": 0.72,
            "classification_reason": (
                f"페이지 문맥에서 '{segment_area}' 사업 키워드가 확인되고 전사/연결/전체 "
                "표현은 없어 segment 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    if has_total:
        return {
            "metric_scope": "company_total",
            "business_area": "company_total",
            "entity_name": peer_id,
            "confidence": 0.85,
            "classification_reason": (
                "페이지 또는 표 문맥에서 연결/전사/전체/경영실적 등 회사 전체를 나타내는 "
                "표현이 확인되어 company_total 지표로 분류"
            ),
            "context_evidence": _metric_context_evidence(
                current_line_text=current_line_text,
                metric_window=metric_window,
                metric_text=metric_text,
            ),
        }

    return {
        "metric_scope": "unknown",
        "business_area": segment_area or "company_total",
        "entity_name": peer_id,
        "confidence": 0.55,
        "classification_reason": (
            "metric 주변에서 전사/사업부문 판단 근거가 충분하지 않아 unknown으로 분류"
        ),
        "context_evidence": _metric_context_evidence(
            current_line_text=current_line_text,
            metric_window=metric_window,
            metric_text=metric_text,
        ),
    }


def _metric_context_evidence(
    *,
    current_line_text: str,
    metric_window: str,
    metric_text: str,
) -> str:
    evidence_parts = [
        f"metric 행: {current_line_text}" if current_line_text else "",
        f"주변 문맥: {metric_text or metric_window}" if metric_text or metric_window else "",
    ]
    return " | ".join(part for part in evidence_parts if part)[:1200]


def _metric_context_text(page_text: str, raw_match: str | None) -> str:
    line_context = _metric_line_context(page_text, raw_match)
    if line_context:
        return line_context
    return _metric_context_window(page_text, raw_match)


def _metric_current_line_text(page_text: str, raw_match: str | None) -> str:
    if not raw_match:
        return ""

    raw_lower = raw_match.lower()
    for line in str(page_text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if raw_lower in line.lower():
            return line.lower()
    return ""


def _metric_line_context(page_text: str, raw_match: str | None) -> str | None:
    if not raw_match:
        return None

    lines = [re.sub(r"\s+", " ", line).strip() for line in str(page_text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None

    raw_lower = raw_match.lower()
    for index, line in enumerate(lines):
        if raw_lower not in line.lower():
            continue

        start = max(0, index - 3)
        context_lines = lines[start : index + 1]
        return " ".join(context_lines).lower()

    return None


def _metric_context_window(page_text: str, raw_match: str | None) -> str:
    compact_text = " ".join((page_text or "").split())
    if not raw_match:
        return compact_text.lower()

    match_index = compact_text.lower().find(raw_match.lower())
    if match_index < 0:
        return compact_text.lower()

    start = max(0, match_index - 35)
    end = min(len(compact_text), match_index + len(raw_match))
    return compact_text[start:end].lower()


def _detect_business_area(text: str) -> str | None:
    lowered = (text or "").lower()
    for business_area, terms in _IR_SEGMENT_CONTEXT_RULES:
        if any(term.lower() in lowered for term in terms):
            return business_area
    return None


def _detect_portfolio_entity(text: str) -> str | None:
    entity_terms = (
        ("sk_innovation", ("sk이노베이션", "sk innovation", "에스케이이노베이션")),
        ("sk_square", ("sk스퀘어", "sk square")),
        ("sk_biopharmaceuticals", ("sk바이오팜", "sk biopharmaceuticals")),
        ("sk_telecom", ("sk텔레콤", "sk telecom")),
        ("sk_hynix", ("sk하이닉스", "sk hynix")),
        ("sk_e_and_s", ("sk e&s", "에스케이 e&s")),
    )
    for entity_name, terms in entity_terms:
        if any(term.lower() in text for term in terms):
            return entity_name
    return None
