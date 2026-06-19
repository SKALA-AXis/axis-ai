"""ir _table_columns — extracted from facade (move-only)."""

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
from src.parsers.ir._document_processing import (  # noqa: F401
    _append_ir_chunk,
    _article_extra,
    _article_get,
    _article_peer_id,
    _article_published_at,
    _build_page_index,
    _classify_page_section,
    _company_section_rules,
    _contains_token,
    _extract_sections_and_chunks,
    _filter_pages_for_peer,
    _has_financial_metric_text,
    _has_ir_signal,
    _ir_paragraphs,
    _is_informative_ir_chunk,
    _is_sk_ax_page,
    _match_topics,
    _normalize_ir_chunk_text,
    _page_text_from_pdf_blocks,
    _pages_from_article_content,
    _pages_from_ir_article,
    _period_from_ir_article,
    _should_use_content_pages,
    _split_long_ir_paragraph,
    _split_text_chunks,
)
from src.parsers.ir._utils import (  # noqa: F401
    _candidate_value_key,
    _classify_metric_context,
    _detect_business_area,
    _detect_portfolio_entity,
    _extract_period,
    _first_amount,
    _first_metric_value,
    _has_same_table_metric_candidate,
    _inline_comparison_candidates,
    _metric_context_evidence,
    _metric_context_text,
    _metric_context_window,
    _metric_current_line_text,
    _metric_line_context,
    _metric_values,
    _normalize_amount_krwbn,
    _normalize_percentage,
    _period_parts,
    _signed_percentage,
)


def _table_columns_from_header_lines(
    line: str,
    next_line: str | None,
    prev_line: str | None = None,
    *,
    report_period: str | None = None,
) -> list[dict[str, Any]]:
    posco_columns = _posco_dx_appendix_statement_columns(prev_line, line, next_line)
    if posco_columns:
        return posco_columns

    posco_compact_columns = _posco_dx_compact_statement_columns(
        prev_line,
        line,
        next_line,
        report_period=report_period,
    )
    if posco_compact_columns:
        return posco_compact_columns

    hierarchical = _hierarchical_table_columns(line, next_line)
    if hierarchical:
        return hierarchical
    return _table_columns_from_header(line)


def _table_columns_from_header(line: str) -> list[dict[str, Any]]:
    repaired_columns = _ocr_scrambled_same_line_columns(line)
    if repaired_columns:
        return repaired_columns

    columns: list[dict[str, Any]] = []
    for match in _IR_TABLE_PERIOD_PATTERN.finditer(line or ""):
        label = match.group(0).strip()
        period, period_year, period_quarter, period_type = _normalize_table_period(label)
        if not period:
            continue
        columns.append(
            {
                "label": label,
                "period": period,
                "period_year": period_year,
                "period_quarter": period_quarter,
                "period_type": period_type,
                "start": match.start(),
            }
        )
    for match in _IR_TABLE_COMPARISON_PATTERN.finditer(line or ""):
        label = match.group(0).strip()
        columns.append(
            {
                "label": label,
                "comparison_type": _comparison_column_type(label),
                "start": match.start(),
            }
        )
    columns.sort(key=lambda column: int(column.get("start") or 0))
    for column in columns:
        column.pop("start", None)
    return columns


def _hierarchical_table_columns(line: str, next_line: str | None) -> list[dict[str, Any]]:
    if not line or not next_line:
        return []

    year_matches = list(re.finditer(r"(20\d{2}|\d{2})\s*년", line))
    if not year_matches:
        year_matches = list(re.finditer(r"\b(20\d{2})\b", line))
    if not year_matches:
        return []

    subpattern = re.compile(
        r"[1-4]\s*분기|연간|\b[1-4]\s*Q\b|\b[12]\s*H\b|\bQoQ\b|\bYoY\b|"
        r"전분기\s*대비|전년\s*(?:동기\s*)?대비",
        re.IGNORECASE,
    )
    sub_matches = list(subpattern.finditer(next_line))
    if not sub_matches:
        return []
    if all(_IR_TABLE_COMPARISON_PATTERN.fullmatch(match.group(0).strip()) for match in sub_matches):
        return []
    if len(sub_matches) < 2 and _IR_TABLE_NUMBER_PATTERN.search(next_line):
        return []

    line_width = max(len(line), 1)
    sub_width = max(len(next_line), 1)
    years = [
        {
            "year": _header_year_value(match.group(1)),
            "start": match.start(),
        }
        for match in year_matches
    ]
    if not years:
        return []

    repaired_columns = _ocr_scrambled_hierarchical_columns(years, sub_matches)
    if repaired_columns:
        return repaired_columns

    assigned_years = _assigned_hierarchical_years(years, sub_matches)
    columns: list[dict[str, Any]] = []
    last_period_column_by_year: dict[int, dict[str, Any]] = {}
    for match, assigned_year in zip(sub_matches, assigned_years, strict=True):
        label = match.group(0).strip()
        scaled_start = int(match.start() * line_width / sub_width)
        year = assigned_year or _nearest_header_year(years, scaled_start)
        if year is None:
            continue

        comparison_type = None
        if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label):
            comparison_type = _comparison_column_type(label)
        if comparison_type:
            target = last_period_column_by_year.get(year)
            column: dict[str, Any] = {
                "label": label,
                "comparison_type": comparison_type,
                "parent_year": year,
            }
            if target:
                column.update(
                    {
                        "comparison_target_label": target.get("label"),
                        "comparison_target_period": target.get("period"),
                        "comparison_target_period_year": target.get("period_year"),
                        "comparison_target_period_quarter": target.get("period_quarter"),
                        "comparison_target_period_type": target.get("period_type"),
                    }
                )
            columns.append(column)
            continue

        period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
        if not period:
            continue
        column = {
            "label": f"{year}년 {label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": year,
        }
        columns.append(column)
        last_period_column_by_year[year] = column

    return columns


def _looks_like_parent_year_header_only(
    line: str,
    next_line: str | None,
    period_columns: list[dict[str, Any]],
) -> bool:
    compact_line = re.sub(r"\s+", "", line or "")
    if not re.fullmatch(r"(?:20\d{2}년?){2,4}", compact_line):
        return False
    next_compact = re.sub(r"\s+", "", next_line or "").lower()
    if "구분" not in next_compact:
        return False
    if len(period_columns) == len(re.findall(r"20\d{2}", line or "")):
        return True
    return bool(
        "yoy" in next_compact
        or re.search(r"\b[1-4]\s*q\b", next_line or "", flags=re.IGNORECASE)
        or len(re.findall(r"20\d{2}", next_line or "")) >= 2
    )


def _has_future_quarter_column(
    period_columns: list[dict[str, Any]],
    report_period: str | None,
) -> bool:
    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type != "quarter" or not report_year or not report_quarter:
        return False

    for column in period_columns:
        period_year = column.get("period_year")
        period_quarter = column.get("period_quarter")
        period_type = column.get("period_type")
        if period_type != "quarter":
            continue
        if not isinstance(period_year, int) or not isinstance(period_quarter, int):
            continue
        if period_year > report_year:
            return True
        if period_year == report_year and period_quarter > report_quarter:
            return True
    return False


def _posco_dx_appendix_statement_columns(
    prev_line: str | None,
    line: str,
    next_line: str | None,
) -> list[dict[str, Any]]:
    """Repair POSCO DX appendix headers split across three extracted lines.

    The 1Q26 appendix income statement is extracted as:
      `2025 2026`
      `구 분 2023 2024 YoY`
      `1Q 2Q 3Q 4Q 1Q`

    The intended columns are:
      2023 annual, 2024 annual, 2025 Q1~Q4, 2025 annual, 2026 Q1, YoY.
    Without this repair, the generic hierarchical parser maps the same values
    to impossible periods such as 2026Q2/2026Q3/2026Q4.
    """
    prev_years = [
        _header_year_value(match.group(1))
        for match in re.finditer(r"\b(20\d{2})\b", prev_line or "")
    ]
    base_years = [
        _header_year_value(match.group(1)) for match in re.finditer(r"\b(20\d{2})\b", line or "")
    ]
    quarter_labels = [
        re.sub(r"\s+", "", match.group(0)).upper()
        for match in re.finditer(r"\b[1-4]\s*Q\b", next_line or "", flags=re.IGNORECASE)
    ]
    if len(quarter_labels) not in {5, 8}:
        # Some OCR/text extraction paths collapse the sub-period row into the
        # base header: `구 분 2023 2024 YoY 1Q 2Q 3Q 4Q 1Q`.
        quarter_labels = [
            re.sub(r"\s+", "", match.group(0)).upper()
            for match in re.finditer(r"\b[1-4]\s*Q\b", line or "", flags=re.IGNORECASE)
        ]

    prev_year_values: list[int] = [year for year in prev_years if isinstance(year, int)]
    base_year_values: list[int] = [year for year in base_years if isinstance(year, int)]
    if len(prev_year_values) != 2:
        return []

    first_detail_year, second_detail_year = prev_year_values
    if second_detail_year != first_detail_year + 1:
        return []
    if not re.search(r"\bYoY\b|전년", line or "", flags=re.IGNORECASE):
        return []

    if len(base_year_values) == 1 and len(quarter_labels) == 8:
        return _posco_dx_annual_appendix_statement_columns(
            base_year=base_year_values[0],
            first_detail_year=first_detail_year,
            second_detail_year=second_detail_year,
            quarter_labels=quarter_labels,
        )

    if len(base_year_values) != 2 or len(quarter_labels) != 5:
        return []
    if base_year_values != [first_detail_year - 2, first_detail_year - 1]:
        return []

    columns: list[dict[str, Any]] = []
    for year in base_year_values:
        columns.append(
            {
                "label": f"{year}년 연간",
                "period": str(year),
                "period_year": year,
                "period_quarter": None,
                "period_type": "year",
                "parent_year": year,
            }
        )

    detail_year_quarters: list[dict[str, Any]] = []
    for label in quarter_labels[:4]:
        period, period_year, period_quarter, period_type = _normalize_subperiod(
            label,
            first_detail_year,
        )
        if not period:
            return []
        column = {
            "label": f"{first_detail_year}년 {label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": first_detail_year,
        }
        columns.append(column)
        detail_year_quarters.append(column)

    columns.append(
        {
            "label": f"{first_detail_year}년 연간",
            "period": str(first_detail_year),
            "period_year": first_detail_year,
            "period_quarter": None,
            "period_type": "year",
            "parent_year": first_detail_year,
        }
    )

    period, period_year, period_quarter, period_type = _normalize_subperiod(
        quarter_labels[4],
        second_detail_year,
    )
    if not period:
        return []
    current_column = {
        "label": f"{second_detail_year}년 {quarter_labels[4]}",
        "period": period,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "period_type": period_type,
        "parent_year": second_detail_year,
    }
    columns.append(current_column)
    columns.append(
        {
            "label": "YoY",
            "comparison_type": "yoy",
            "parent_year": second_detail_year,
            "comparison_target_label": current_column.get("label"),
            "comparison_target_period": current_column.get("period"),
            "comparison_target_period_year": current_column.get("period_year"),
            "comparison_target_period_quarter": current_column.get("period_quarter"),
            "comparison_target_period_type": current_column.get("period_type"),
        }
    )
    return columns


def _posco_dx_annual_appendix_statement_columns(
    *,
    base_year: int,
    first_detail_year: int,
    second_detail_year: int,
    quarter_labels: list[str],
) -> list[dict[str, Any]]:
    """Repair POSCO DX annual appendix statement headers.

    The 2025 annual IR appendix income statement is extracted as:
      `2024 2025`
      `구 분 2023 YoY`
      `1Q 2Q 3Q 4Q 1Q 2Q 3Q 4Q`

    The intended columns are:
      2023 annual,
      2024 Q1~Q4, 2024 annual,
      2025 Q1~Q4, 2025 annual,
      YoY for the 2025 annual column.
    """
    if base_year != first_detail_year - 1:
        return []
    if len(quarter_labels) != 8:
        return []

    columns: list[dict[str, Any]] = [
        {
            "label": f"{base_year}년 연간",
            "period": str(base_year),
            "period_year": base_year,
            "period_quarter": None,
            "period_type": "year",
            "parent_year": base_year,
        }
    ]

    for year, labels in (
        (first_detail_year, quarter_labels[:4]),
        (second_detail_year, quarter_labels[4:]),
    ):
        for label in labels:
            period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
            if not period:
                return []
            columns.append(
                {
                    "label": f"{year}년 {label}",
                    "period": period,
                    "period_year": period_year,
                    "period_quarter": period_quarter,
                    "period_type": period_type,
                    "parent_year": year,
                }
            )
        columns.append(
            {
                "label": f"{year}년 연간",
                "period": str(year),
                "period_year": year,
                "period_quarter": None,
                "period_type": "year",
                "parent_year": year,
            }
        )

    current_column = columns[-1]
    columns.append(
        {
            "label": "YoY",
            "comparison_type": "yoy",
            "parent_year": second_detail_year,
            "comparison_target_label": current_column.get("label"),
            "comparison_target_period": current_column.get("period"),
            "comparison_target_period_year": current_column.get("period_year"),
            "comparison_target_period_quarter": current_column.get("period_quarter"),
            "comparison_target_period_type": current_column.get("period_type"),
        }
    )
    return columns


def _posco_dx_compact_statement_columns(
    prev_line: str | None,
    line: str,
    next_line: str | None,
    *,
    report_period: str | None,
) -> list[dict[str, Any]]:
    """Repair POSCO DX compact quarterly/appendix tables.

    Several POSCO DX IR PDFs extract the table header as three separate lines:
      `2023 2024 2025`
      `구 분 2022`
      `1Q 1Q 4Q 1Q QoQ YoY`

    The annual columns are visually present in the PDF but often missing from
    the OCR subheader line. This helper reconstructs those hidden annual
    columns so the current quarter values are not shifted into historical
    annual periods.
    """
    if "구" not in (line or "") or "분" not in (line or ""):
        return []

    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type not in {"quarter", "year"} or not report_year:
        return []

    header_years = [
        year
        for year in (
            _header_year_value(match.group(1))
            for match in re.finditer(r"\b(20\d{2}|\d{2})\s*년?\b", prev_line or "")
        )
        if isinstance(year, int)
    ]
    base_years = [
        year
        for year in (
            _header_year_value(match.group(1))
            for match in re.finditer(r"\b(20\d{2}|\d{2})\s*년?\b", line or "")
        )
        if isinstance(year, int)
    ]
    if not header_years:
        return []

    label_matches = list(
        re.finditer(
            r"\b[1-4]\s*Q\b|\b[12]\s*H\b|연간|\bQoQ\b|\bYoY\b",
            next_line or "",
            flags=re.IGNORECASE,
        )
    )
    if not label_matches:
        return []

    labels = [re.sub(r"\s+", "", match.group(0)).upper() for match in label_matches]
    period_labels = [label for label in labels if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    comparison_labels = [label for label in labels if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    if len(header_years) < 2 or not period_labels:
        return []

    if len(base_years) == 1 and len(header_years) == 3 and len(period_labels) == 4:
        return _posco_dx_q1_with_hidden_annual_columns(
            base_year=base_years[0],
            header_years=header_years,
            quarter_labels=period_labels,
            comparison_labels=comparison_labels,
        )

    if (
        not base_years
        and len(header_years) == 3
        and len(period_labels) == 4
        and report_quarter == 1
    ):
        return _posco_dx_q1_with_hidden_annual_columns(
            base_year=None,
            header_years=header_years,
            quarter_labels=period_labels,
            comparison_labels=comparison_labels,
        )

    if not base_years and len(header_years) == 3 and len(period_labels) == 4:
        current_year = header_years[-1]
        columns = [
            _period_column(header_years[0], period_labels[0]),
            _period_column(header_years[1], period_labels[1]),
            _period_column(current_year, period_labels[2]),
            _period_column(current_year, period_labels[3]),
        ]
        return _append_comparison_columns(columns, comparison_labels)

    if not base_years and "연간" in period_labels:
        columns = _posco_dx_annual_interleaved_columns(
            header_years=header_years,
            period_labels=period_labels,
            comparison_labels=comparison_labels,
        )
        if columns:
            return columns

    if not base_years and len(header_years) == 3 and any("H" in label for label in labels):
        columns = _posco_dx_half_year_columns(
            header_years=header_years,
            labels=labels,
        )
        if columns:
            return columns

    if not base_years and len(header_years) == 3:
        columns = _posco_dx_quarter_history_columns(
            header_years=header_years,
            labels=labels,
        )
        if columns:
            return columns

    return []


def _posco_dx_q1_with_hidden_annual_columns(
    *,
    base_year: int | None,
    header_years: list[int],
    quarter_labels: list[str],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    if len(header_years) != 3 or len(quarter_labels) != 4:
        return []

    columns: list[dict[str, Any]] = []
    if base_year is not None:
        columns.append(_annual_column(base_year))
        detail_years = header_years
    else:
        first_year, second_year, current_year = header_years
        columns.extend(
            [
                _period_column(first_year, quarter_labels[0]),
                _annual_column(first_year),
            ]
        )
        detail_years = [second_year, current_year]
        quarter_labels = quarter_labels[1:]

    if len(detail_years) == 3:
        first_year, second_year, current_year = detail_years
        columns.extend(
            [
                _period_column(first_year, quarter_labels[0]),
                _annual_column(first_year),
                _period_column(second_year, quarter_labels[1]),
                _period_column(second_year, quarter_labels[2]),
                _annual_column(second_year),
                _period_column(current_year, quarter_labels[3]),
            ]
        )
    elif len(detail_years) == 2:
        previous_year, current_year = detail_years
        columns.extend(
            [
                _period_column(previous_year, quarter_labels[0]),
                _period_column(previous_year, quarter_labels[1]),
                _annual_column(previous_year),
                _period_column(current_year, quarter_labels[2]),
            ]
        )
    else:
        return []

    return _append_comparison_columns(columns, comparison_labels)


def _posco_dx_half_year_columns(
    *,
    header_years: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    first_comparison = _first_comparison_index(labels)
    if first_comparison is None:
        return []

    before = labels[:first_comparison]
    after_comparisons = labels[first_comparison:]
    comparison_labels = [
        label for label in after_comparisons if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
    ]
    trailing_period_labels = [
        label for label in after_comparisons if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
    ]
    if len(header_years) != 3 or len(before) < 8:
        return []

    columns: list[dict[str, Any]] = []
    chunks = [before[:3], before[3:6], before[6:]]
    for year, chunk in zip(header_years, chunks, strict=True):
        for label in chunk:
            columns.append(_period_column(year, label))
    columns = _append_comparison_columns(columns, comparison_labels)
    current_year = header_years[-1]
    for label in trailing_period_labels:
        columns.append(_period_column(current_year, label))
    return columns


def _posco_dx_annual_interleaved_columns(
    *,
    header_years: list[int],
    period_labels: list[str],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    if len(header_years) < 2:
        return []

    columns: list[dict[str, Any]] = []
    year_index = 0
    for label in period_labels:
        if year_index >= len(header_years):
            return []
        year = header_years[year_index]
        columns.append(_period_column(year, label))
        if label == "연간" and year_index < len(header_years) - 1:
            year_index += 1

    return _append_comparison_columns(columns, comparison_labels)


def _posco_dx_quarter_history_columns(
    *,
    header_years: list[int],
    labels: list[str],
) -> list[dict[str, Any]]:
    first_comparison = _first_comparison_index(labels)
    if first_comparison is None:
        period_labels = labels
        comparison_labels: list[str] = []
    else:
        period_labels = labels[:first_comparison]
        comparison_labels = labels[first_comparison:]
    if len(header_years) != 3 or len(period_labels) < 7:
        return []

    columns: list[dict[str, Any]] = []
    year_index = 0
    quarter_count_for_year = 0
    for label in period_labels:
        if year_index >= len(header_years):
            return []
        year = header_years[year_index]
        columns.append(_period_column(year, label))
        quarter_count_for_year += 1
        if quarter_count_for_year == 4 and year_index < len(header_years) - 1:
            columns.append(_annual_column(year))
            year_index += 1
            quarter_count_for_year = 0

    if quarter_count_for_year == 4 and year_index == len(header_years) - 1:
        columns.append(_annual_column(header_years[year_index]))

    return _append_comparison_columns(columns, comparison_labels)


def _first_comparison_index(labels: list[str]) -> int | None:
    for index, label in enumerate(labels):
        if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label):
            return index
    return None


def _annual_column(year: int) -> dict[str, Any]:
    return {
        "label": f"{year}년 연간",
        "period": str(year),
        "period_year": year,
        "period_quarter": None,
        "period_type": "year",
        "parent_year": year,
    }


def _period_column(year: int, label: str) -> dict[str, Any]:
    period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
    if not period:
        return {}
    return {
        "label": f"{year}년 {label}",
        "period": period,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "period_type": period_type,
        "parent_year": year,
    }


def _append_comparison_columns(
    columns: list[dict[str, Any]],
    comparison_labels: list[str],
) -> list[dict[str, Any]]:
    columns = [column for column in columns if column.get("period")]
    target = next(
        (column for column in reversed(columns) if column.get("period_type") == "quarter"),
        columns[-1] if columns else None,
    )
    for label in comparison_labels:
        comparison_column: dict[str, Any] = {
            "label": label,
            "comparison_type": _comparison_column_type(label),
        }
        if target:
            comparison_column.update(
                {
                    "comparison_target_label": target.get("label"),
                    "comparison_target_period": target.get("period"),
                    "comparison_target_period_year": target.get("period_year"),
                    "comparison_target_period_quarter": target.get("period_quarter"),
                    "comparison_target_period_type": target.get("period_type"),
                }
            )
        columns.append(comparison_column)
    return columns


def _ocr_scrambled_same_line_columns(line: str) -> list[dict[str, Any]]:
    year_matches = list(re.finditer(r"\b(20\d{2})\b", line or ""))
    if len(year_matches) != 2:
        return []
    subpattern = re.compile(
        r"연간|\b[1-4]\s*Q\b|\b[12]\s*H\b|\bQoQ\b|\bYoY\b|"
        r"전분기\s*대비|전년\s*(?:동기\s*)?대비",
        re.IGNORECASE,
    )
    sub_matches = list(subpattern.finditer(line or ""))
    if not sub_matches:
        return []
    return _ocr_scrambled_hierarchical_columns(
        [
            {"year": _header_year_value(match.group(1)), "start": match.start()}
            for match in year_matches
        ],
        sub_matches,
    )


def _ocr_scrambled_hierarchical_columns(
    years: list[dict[str, Any]],
    sub_matches: list[re.Match[str]],
) -> list[dict[str, Any]]:
    """Handle OCR headers like `2024 2025 YoY QoQ 2Q 연간 1Q 2Q 1H`.

    In the original table the comparison columns are usually at the far right, but text
    extraction sometimes pulls `YoY/QoQ` before the period subheaders. Reconstruct the
    intended order as previous-year periods, current-year periods, then comparisons.
    """
    if len(years) != 2:
        return []

    first_year = years[0].get("year")
    second_year = years[1].get("year")
    if not isinstance(first_year, int) or not isinstance(second_year, int):
        return []
    if second_year != first_year + 1:
        return []

    labels = [re.sub(r"\s+", "", match.group(0)).lower() for match in sub_matches]
    comparison_labels = [label for label in labels if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    period_labels = [label for label in labels if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)]
    if len(comparison_labels) < 1 or len(period_labels) < 3:
        return []
    first_comparison_index = next(
        (
            index
            for index, label in enumerate(labels)
            if _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
        ),
        None,
    )
    first_period_index = next(
        (
            index
            for index, label in enumerate(labels)
            if not _IR_TABLE_COMPARISON_PATTERN.fullmatch(label)
        ),
        None,
    )
    if first_comparison_index is None or first_period_index is None:
        return []
    if first_comparison_index > first_period_index:
        return []

    first_year_count = 1
    for index, label in enumerate(period_labels):
        if label == "연간":
            first_year_count = index + 1
            break
    if first_year_count >= len(period_labels):
        return []

    columns: list[dict[str, Any]] = []
    last_second_year_period: dict[str, Any] | None = None
    for index, label in enumerate(period_labels):
        year = first_year if index < first_year_count else second_year
        period, period_year, period_quarter, period_type = _normalize_subperiod(label, year)
        if not period:
            continue
        column = {
            "label": f"{year}년 {label.upper() if label.endswith('q') else label}",
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "parent_year": year,
        }
        columns.append(column)
        if year == second_year and period_type == "quarter":
            last_second_year_period = column

    if not columns:
        return []

    for label in comparison_labels:
        comparison_type = _comparison_column_type(label)
        comparison_column: dict[str, Any] = {
            "label": label.upper(),
            "comparison_type": comparison_type,
            "parent_year": second_year,
        }
        if last_second_year_period:
            comparison_column.update(
                {
                    "comparison_target_label": last_second_year_period.get("label"),
                    "comparison_target_period": last_second_year_period.get("period"),
                    "comparison_target_period_year": last_second_year_period.get("period_year"),
                    "comparison_target_period_quarter": last_second_year_period.get(
                        "period_quarter"
                    ),
                    "comparison_target_period_type": last_second_year_period.get("period_type"),
                }
            )
        columns.append(comparison_column)

    return columns


def _assigned_hierarchical_years(
    years: list[dict[str, Any]],
    sub_matches: list[re.Match[str]],
) -> list[int | None]:
    labels = [re.sub(r"\s+", "", match.group(0)).lower() for match in sub_matches]
    if len(years) == 2:
        first_year = years[0].get("year")
        second_year = years[1].get("year")
        if (
            isinstance(first_year, int)
            and isinstance(second_year, int)
            and second_year == first_year + 1
            and len(labels) >= 4
            and re.fullmatch(r"[1-4]분기|[1-4]q", labels[0], flags=re.IGNORECASE)
            and labels[1] == "연간"
        ):
            return [first_year if index < 2 else second_year for index in range(len(labels))]
        if (
            isinstance(first_year, int)
            and isinstance(second_year, int)
            and second_year == first_year + 1
            and len(labels) >= 2
            and _is_quarter_subheader_label(labels[0])
            and _is_quarter_subheader_label(labels[1])
        ):
            assigned: list[int | None] = []
            active_year = first_year
            period_seen = 0
            for label in labels:
                if _is_quarter_subheader_label(label) or label == "연간":
                    period_seen += 1
                    active_year = first_year if period_seen == 1 else second_year
                assigned.append(active_year)
            return assigned
    return [None for _ in labels]


def _is_quarter_subheader_label(label: str) -> bool:
    return bool(re.fullmatch(r"[1-4]분기|[1-4]q", label, flags=re.IGNORECASE))


def _header_year_value(value: str) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return 2000 + year if year < 100 else year


def _nearest_header_year(years: list[dict[str, Any]], scaled_start: int) -> int | None:
    eligible = [item for item in years if int(item["start"]) <= scaled_start]
    selected = eligible[-1] if eligible else years[0]
    year = selected.get("year")
    return int(year) if isinstance(year, int) else None


def _normalize_subperiod(
    label: str,
    year: int,
) -> tuple[str | None, int | None, int | None, str | None]:
    value = re.sub(r"\s+", "", label or "").lower()
    match = re.search(r"([1-4])분기", value)
    if not match:
        match = re.search(r"([1-4])q", value, flags=re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    if value == "연간":
        return str(year), year, None, "year"
    match = re.search(r"([12])h", value, flags=re.IGNORECASE)
    if match:
        half = int(match.group(1))
        return f"{year}H{half}", year, None, "half"
    return None, None, None, None


def _dedupe_table_columns(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for column in columns:
        key = (
            column.get("label"),
            column.get("period"),
            column.get("comparison_type"),
            column.get("comparison_target_period"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(column)
    return deduped


def _coerce_quarterly_context_columns(
    columns: list[dict[str, Any]],
    *,
    table_title: str | None,
    report_period: str | None,
) -> list[dict[str, Any]]:
    report_year, report_quarter, report_period_type = _period_parts(report_period)
    if report_period_type != "quarter" or not report_year or not report_quarter:
        return columns
    if not _looks_like_quarterly_table_context(table_title, report_period):
        return columns
    if any(column.get("period_type") == "quarter" for column in columns):
        return columns

    coerced: list[dict[str, Any]] = []
    for column in columns:
        if column.get("period_type") != "year":
            coerced.append(column)
            continue
        period_year = column.get("period_year")
        if not isinstance(period_year, int):
            coerced.append(column)
            continue
        patched = dict(column)
        patched["original_period"] = column.get("period")
        patched["original_period_type"] = column.get("period_type")
        patched["period"] = f"{period_year}Q{report_quarter}"
        patched["period_quarter"] = report_quarter
        patched["period_type"] = "quarter"
        patched["period_inferred_from_quarterly_context"] = True
        coerced.append(patched)
    return coerced


def _looks_like_quarterly_table_context(
    table_title: str | None,
    report_period: str | None,
) -> bool:
    if not table_title:
        return False
    report_year, report_quarter, _period_type = _period_parts(report_period)
    if not report_year or not report_quarter:
        return False

    compact = re.sub(r"\s+", "", table_title or "").lower()
    short_year = str(report_year)[2:]
    quarter_tokens = {
        f"{report_year}년{report_quarter}분기",
        f"{short_year}년{report_quarter}분기",
        f"{report_year}q{report_quarter}",
        f"{short_year}q{report_quarter}",
        f"{report_quarter}q{short_year}",
    }
    if any(token in compact for token in quarter_tokens):
        return True
    return bool("분기별" in compact or "quarterly" in compact)


def _comparison_columns_from_continuation(line: str) -> list[dict[str, Any]]:
    if _IR_TABLE_NUMBER_PATTERN.search(line or ""):
        return []
    return [
        {
            "label": match.group(0).strip(),
            "comparison_type": _comparison_column_type(match.group(0).strip()),
        }
        for match in _IR_TABLE_COMPARISON_PATTERN.finditer(line or "")
    ]


def _period_columns_from_header(line: str) -> list[dict[str, Any]]:
    return [column for column in _table_columns_from_header(line) if column.get("period")]


def _comparison_column_type(label: str) -> str:
    value = re.sub(r"\s+", "", label or "").lower()
    if value == "qoq" or "전분기" in value:
        return "qoq"
    return "yoy"


def _comparison_metric_name(metric_name: str, comparison_type: str) -> str:
    return f"{metric_name}_{comparison_type}"


def _primary_period_column(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((column for column in columns if column.get("period")), None)


def _report_period_column(report_period: str | None) -> dict[str, Any] | None:
    year, quarter, period_type = _period_parts(report_period)
    if not report_period or not year:
        return None
    return {
        "label": report_period,
        "period": report_period,
        "period_year": year,
        "period_quarter": quarter,
        "period_type": period_type,
    }


def _comparison_target_column(column: dict[str, Any]) -> dict[str, Any] | None:
    period = column.get("comparison_target_period")
    if not period:
        return None
    return {
        "label": column.get("comparison_target_label") or period,
        "period": period,
        "period_year": column.get("comparison_target_period_year"),
        "period_quarter": column.get("comparison_target_period_quarter"),
        "period_type": column.get("comparison_target_period_type"),
    }


def _normalize_table_period(label: str) -> tuple[str | None, int | None, int | None, str | None]:
    value = re.sub(r"\s+", " ", label or "").strip()
    match = re.search(r"(\d{2})\s*년\s*([1-4])\s*분기", value)
    if match:
        year = 2000 + int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*년\s*([1-4])\s*분기", value)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*Q\s*([1-4])", value, flags=re.IGNORECASE)
    if match:
        year = int(match.group(1))
        quarter = int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"([1-4])\s*Q\s*['’]?\s*(\d{2})", value, flags=re.IGNORECASE)
    if match:
        quarter = int(match.group(1))
        year = 2000 + int(match.group(2))
        return f"{year}Q{quarter}", year, quarter, "quarter"
    match = re.search(r"(20\d{2})\s*년\s*(상반기|하반기)", value)
    if match:
        year = int(match.group(1))
        half = 1 if match.group(2) == "상반기" else 2
        return f"{year}H{half}", year, None, "half"
    match = re.search(r"(\d{2})\s*년\s*(상반기|하반기)", value)
    if match:
        year = 2000 + int(match.group(1))
        half = 1 if match.group(2) == "상반기" else 2
        return f"{year}H{half}", year, None, "half"
    match = re.search(r"(\d{2})\s*년$", value)
    if match:
        year = 2000 + int(match.group(1))
        return str(year), year, None, "year"
    match = re.search(r"(20\d{2})\s*년$", value)
    if match:
        year = int(match.group(1))
        return str(year), year, None, "year"
    match = re.search(r"\b(20\d{2})\b", value)
    if match:
        year = int(match.group(1))
        return str(year), year, None, "year"
    return None, None, None, None
