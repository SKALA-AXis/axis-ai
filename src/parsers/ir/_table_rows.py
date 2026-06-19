"""ir _table_rows — extracted from facade (move-only)."""

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
from src.parsers.ir._table_columns import (  # noqa: F401
    _annual_column,
    _append_comparison_columns,
    _assigned_hierarchical_years,
    _coerce_quarterly_context_columns,
    _comparison_column_type,
    _comparison_columns_from_continuation,
    _comparison_metric_name,
    _comparison_target_column,
    _dedupe_table_columns,
    _first_comparison_index,
    _has_future_quarter_column,
    _header_year_value,
    _hierarchical_table_columns,
    _is_quarter_subheader_label,
    _looks_like_parent_year_header_only,
    _looks_like_quarterly_table_context,
    _nearest_header_year,
    _normalize_subperiod,
    _normalize_table_period,
    _ocr_scrambled_hierarchical_columns,
    _ocr_scrambled_same_line_columns,
    _period_column,
    _period_columns_from_header,
    _posco_dx_annual_appendix_statement_columns,
    _posco_dx_annual_interleaved_columns,
    _posco_dx_appendix_statement_columns,
    _posco_dx_compact_statement_columns,
    _posco_dx_half_year_columns,
    _posco_dx_q1_with_hidden_annual_columns,
    _posco_dx_quarter_history_columns,
    _primary_period_column,
    _report_period_column,
    _table_columns_from_header,
    _table_columns_from_header_lines,
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


def _extract_financial_table_candidates(
    pages: list[dict[str, Any]],
    *,
    report_period: str | None,
    peer_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for page in pages:
        page_no = page.get("page")
        page_text = str(page.get("text") or "")
        lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
        lines = [line for line in lines if line]
        lines = _merge_split_operating_margin_lines(lines)
        unit = _table_unit(page_text)
        unit_evidence = _table_unit_evidence(lines)

        for header_index, line in enumerate(lines):
            table_title = _nearby_table_title(lines, header_index)
            table_unit = (
                _table_unit("\n".join(lines[max(0, header_index - 3) : header_index + 2])) or unit
            )
            columns = _table_columns_from_header_lines(
                line,
                lines[header_index + 1] if header_index + 1 < len(lines) else None,
                lines[header_index - 1] if header_index > 0 else None,
                report_period=report_period,
            )
            if header_index + 1 < len(lines):
                columns.extend(_comparison_columns_from_continuation(lines[header_index + 1]))
                columns = _dedupe_table_columns(columns)
            columns = _coerce_quarterly_context_columns(
                columns,
                table_title=table_title,
                report_period=report_period,
            )
            period_columns = [column for column in columns if column.get("period")]
            if len(period_columns) < 2:
                continue
            if _looks_like_parent_year_header_only(
                line,
                lines[header_index + 1] if header_index + 1 < len(lines) else None,
                period_columns,
            ):
                continue
            if _has_future_quarter_column(period_columns, report_period):
                continue
            required_value_count = len(period_columns)

            table_uid = f"ir-p{page_no or 'x'}-t{len(tables) + 1}"
            if _should_skip_financial_table(
                peer_id=peer_id,
                table_title=table_title,
                lines=lines,
                header_index=header_index,
            ):
                continue
            table_context_business_area = _table_context_business_area(lines, header_index)
            table_rows: list[dict[str, Any]] = []
            active_metric: tuple[str, str] | None = None
            active_metric_parent_label: str | None = None
            active_amount_metric: tuple[str, str] | None = None
            active_amount_metric_parent_label: str | None = None
            active_business_area: str | None = None
            miss_count = 0
            miss_limit = 8 if required_value_count >= 10 or len(columns) >= 9 else 3
            for row_index, row_line in enumerate(
                lines[header_index + 1 : header_index + 24], start=1
            ):
                row_metric = _table_metric_from_label(row_line)
                row_label = _table_row_label(row_line)
                explicit_metric = row_metric is not None
                if row_metric is not None:
                    active_metric = row_metric
                    active_metric_parent_label = row_label
                    if row_metric[1] != "percentage":
                        active_amount_metric = row_metric
                        active_amount_metric_parent_label = row_label
                elif _looks_like_segment_value_row(row_line, required_value_count):
                    row_values_for_kind = _table_row_values(row_line, len(columns))
                    if (
                        active_metric
                        and active_metric[1] == "percentage"
                        and _row_values_have_percentage(row_values_for_kind)
                    ):
                        row_metric = active_metric
                    elif _row_values_are_percentage(row_values_for_kind) and active_metric:
                        row_metric = active_metric
                    elif active_amount_metric:
                        row_metric = active_amount_metric
                        active_metric_parent_label = active_amount_metric_parent_label
                    elif active_metric and active_metric[1] != "percentage":
                        row_metric = active_metric
                    elif active_metric and active_metric[1] == "percentage":
                        row_metric = ("revenue_total", "amount_krwbn")
                        active_metric_parent_label = None
                    else:
                        if table_rows:
                            miss_count += 1
                            if miss_count >= miss_limit:
                                break
                        continue
                else:
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue

                metric_name, value_kind = row_metric
                row_values = _table_row_values(row_line, len(columns))
                if _row_has_misaligned_leading_comparison_columns(columns, row_values):
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue
                if len(row_values) < required_value_count:
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue

                if not _is_valid_table_metric_row(row_label, row_metric):
                    if table_rows:
                        miss_count += 1
                        if miss_count >= miss_limit:
                            break
                    continue
                display_unit = _table_display_unit(unit=table_unit, value_kind=value_kind)
                row_unit_evidence = _effective_row_unit_evidence(
                    unit_evidence=unit_evidence,
                    row_label=row_label,
                    metric_parent_label=active_metric_parent_label,
                    value_kind=value_kind,
                )
                row_evidence = _table_row_evidence(
                    table_title=table_title,
                    context_business_area=table_context_business_area,
                    metric_parent_label=active_metric_parent_label if not explicit_metric else None,
                    row_label=row_label,
                    columns=columns,
                    row_values=row_values,
                    unit=display_unit,
                    unit_evidence=row_unit_evidence,
                )
                business_area = _table_business_area(
                    row_label,
                    page_text,
                    peer_id=peer_id,
                    context_business_area=table_context_business_area,
                    explicit_metric=explicit_metric,
                    active_metric_parent_label=active_metric_parent_label,
                )
                if (
                    explicit_metric
                    and value_kind == "percentage"
                    and active_business_area
                    and _is_percentage_metric_label(row_label, metric_name)
                ):
                    business_area = active_business_area
                metric_scope = "company_total" if business_area == "company_total" else "segment"
                classification_reason = _table_classification_reason(
                    business_area=business_area,
                    metric_scope=metric_scope,
                    row_label=row_label,
                    table_title=table_title,
                    context_business_area=table_context_business_area,
                    metric_parent_label=active_metric_parent_label if not explicit_metric else None,
                    explicit_metric=explicit_metric,
                )
                table_cells: list[dict[str, Any]] = []
                primary_period_column = _primary_period_column(columns)
                for column, raw_value in zip(columns, row_values, strict=False):
                    comparison_type = column.get("comparison_type")
                    candidate_metric_name = metric_name
                    candidate_value_kind = value_kind
                    candidate_unit = table_unit
                    period_column = column
                    if comparison_type:
                        if "%" not in str(raw_value):
                            continue
                        candidate_metric_name = _comparison_metric_name(
                            metric_name,
                            str(comparison_type),
                        )
                        candidate_value_kind = "percentage"
                        candidate_unit = "%"
                        period_column = (
                            _comparison_target_column(column) or primary_period_column or column
                        )
                    comparison_base = (
                        _comparison_base_cell(
                            comparison_type=str(comparison_type),
                            columns=columns,
                            row_values=row_values,
                            current_period=period_column.get("period"),
                            unit=table_unit,
                        )
                        if comparison_type
                        else {}
                    )

                    normalized = _table_value(
                        raw_value,
                        unit=candidate_unit,
                        value_kind=candidate_value_kind,
                    )
                    if normalized is None:
                        continue
                    cell_unit = _table_cell_unit(
                        raw_value,
                        table_unit=candidate_unit,
                        value_kind=candidate_value_kind,
                    )
                    period = period_column.get("period")
                    candidate: dict[str, Any] = {
                        "page": page_no,
                        "type": candidate_metric_name,
                        "base_metric_type": metric_name,
                        "comparison_type": comparison_type,
                        "value_kind": candidate_value_kind,
                        _candidate_value_key(candidate_value_kind): normalized,
                        "unit": cell_unit,
                        "unit_evidence": row_unit_evidence,
                        "raw": f"{row_label} {column['label']} {raw_value} ({cell_unit})",
                        "source": "ir_table_matrix",
                        "source_table_uid": table_uid,
                        "table_title": table_title,
                        "metric_parent_label": active_metric_parent_label
                        if not explicit_metric
                        else None,
                        "row_label": row_label,
                        "column_label": column["label"],
                        "row_evidence": row_evidence,
                        "evidence_text": (
                            f"{row_evidence} | 선택 셀: {column['label']}={raw_value} "
                            f"({cell_unit}) | 분류 근거: {classification_reason}"
                        ),
                        "classification_reason": classification_reason,
                        "context_evidence": row_evidence,
                        "period": period,
                        "period_year": period_column.get("period_year"),
                        "period_quarter": period_column.get("period_quarter"),
                        "period_type": period_column.get("period_type"),
                        "is_historical": _is_historical_period(period, report_period),
                        **comparison_base,
                        "metric_scope": metric_scope,
                        "business_area": business_area,
                        "entity_name": peer_id,
                        "confidence": 0.88,
                    }
                    candidates.append(candidate)
                    table_cells.append(candidate)
                if table_cells:
                    miss_count = 0
                    table_rows.append(
                        {
                            "row_index": row_index,
                            "row_label": row_label,
                            "metric_name": metric_name,
                            "business_area": business_area,
                            "cells": [
                                {
                                    "column_label": cell["column_label"],
                                    "period": cell["period"],
                                    "raw": cell["raw"],
                                    "type": cell.get("type"),
                                    "comparison_type": cell.get("comparison_type"),
                                    "value": cell.get(_candidate_value_key(cell["value_kind"])),
                                }
                                for cell in table_cells
                            ],
                        }
                    )
                    if metric_scope == "segment" and business_area:
                        active_business_area = str(business_area)
                    elif explicit_metric and value_kind != "percentage":
                        active_business_area = None

            if table_rows:
                tables.append(
                    {
                        "table_uid": table_uid,
                        "page": page_no,
                        "title": table_title,
                        "context_business_area": table_context_business_area,
                        "unit": table_unit,
                        "unit_evidence": unit_evidence,
                        "columns": columns,
                        "rows": table_rows,
                    }
                )

    return candidates, tables


def _merge_split_operating_margin_lines(lines: list[str]) -> list[str]:
    """Merge OCR rows split as `영업` / percentage row / `이익률`."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        next_next = lines[index + 2] if index + 2 < len(lines) else ""
        if (
            re.fullmatch(r"영업", current or "")
            and _row_values_are_percentage(_table_row_values(next_line, 32))
            and re.fullmatch(r"이익률", next_next or "")
        ):
            merged.append(f"영업이익률 {next_line}")
            index += 3
            continue
        merged.append(current)
        index += 1
    return merged


def _extract_chart_block_candidates(
    pages: list[dict[str, Any]],
    *,
    report_period: str | None,
    peer_id: str | None,
) -> list[dict[str, Any]]:
    year, quarter, period_type = _period_parts(report_period)
    if period_type != "quarter" or not year or not quarter:
        return []

    axis_tokens = {
        f"{quarter}q{str(year)[2:]}",
        f"{quarter} q{str(year)[2:]}",
    }
    candidates: list[dict[str, Any]] = []
    for page in pages:
        blocks = page.get("blocks") or []
        if not isinstance(blocks, list):
            continue

        metric_labels = [
            block
            for block in blocks
            if "영업이익률" in str(block.get("text") or "") and _block_center(block) is not None
        ]
        if not metric_labels:
            continue

        axis_blocks = []
        for block in blocks:
            text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip().lower()
            compact = text.replace(" ", "")
            if any(token.replace(" ", "") in compact for token in axis_tokens):
                axis_blocks.append(block)
        if not axis_blocks:
            continue

        axis_center = _block_center(axis_blocks[-1])
        if axis_center is None:
            continue
        axis_x, axis_y = axis_center

        pct_blocks: list[tuple[float, float, float, str]] = []
        for block in blocks:
            text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
            match = re.fullmatch(r"([+\-△▲]?)\s*([\d,]+(?:\.\d+)?)\s*%", text)
            if not match:
                continue
            value = _signed_percentage(match.group(2), match.group(1))
            center = _block_center(block)
            if value is None or center is None:
                continue
            x, y = center
            if y >= axis_y:
                continue
            pct_blocks.append((abs(x - axis_x), x, value, text))

        if not pct_blocks:
            continue

        pct_blocks.sort(key=lambda item: item[0])
        distance, _x, value, raw = pct_blocks[0]
        if distance > 90:
            continue

        metric_context = _classify_metric_context(
            str(page.get("text") or ""),
            peer_id=peer_id,
            raw_match=raw,
        )
        if metric_context.get("metric_scope") == "unknown":
            metric_context = {
                **metric_context,
                "metric_scope": "company_total",
                "business_area": "company_total",
                "entity_name": peer_id,
                "confidence": 0.82,
                "classification_reason": "IR 차트 축의 보고기간 라벨과 영업이익률 값을 좌표로 매칭",
            }
        candidates.append(
            {
                "page": page.get("page"),
                "type": "operating_margin",
                "value_kind": "percentage",
                "value_pct": value,
                "unit": "%",
                "raw": f"영업이익률 {report_period} {raw}",
                "source": "ir_chart_blocks",
                "period": report_period,
                "period_year": year,
                "period_quarter": quarter,
                "period_type": period_type,
                "evidence_text": f"영업이익률 차트 {report_period}={raw}",
                **metric_context,
                "confidence": max(float(metric_context.get("confidence") or 0.0), 0.86),
            }
        )

    return candidates


def _block_center(block: dict[str, Any]) -> tuple[float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return (x0 + x1) / 2, (y0 + y1) / 2


def _comparison_base_cell(
    *,
    comparison_type: str,
    columns: list[dict[str, Any]],
    row_values: list[str],
    current_period: str | None,
    unit: str | None,
) -> dict[str, Any]:
    period_cells = [
        (column, value)
        for column, value in zip(columns, row_values, strict=False)
        if column.get("period")
    ]
    if not period_cells:
        return {}

    current_index = next(
        (
            index
            for index, (column, _value) in enumerate(period_cells)
            if column.get("period") == current_period
        ),
        len(period_cells) - 1,
    )

    base_index: int | None = None
    if comparison_type == "qoq":
        base_index = current_index - 1 if current_index > 0 else None
    elif comparison_type == "yoy":
        current_column = period_cells[current_index][0]
        current_year = current_column.get("period_year")
        current_quarter = current_column.get("period_quarter")
        for index, (column, _value) in enumerate(period_cells):
            if (
                isinstance(current_year, int)
                and column.get("period_year") == current_year - 1
                and column.get("period_quarter") == current_quarter
            ):
                base_index = index
                break
        if base_index is None and current_index > 0:
            base_index = current_index - 1

    if base_index is None or base_index < 0 or base_index >= len(period_cells):
        return {}

    base_column, base_raw_value = period_cells[base_index]
    base_value = _table_value(base_raw_value, unit=unit, value_kind="amount_krwbn")
    if base_value is None:
        base_value = _table_value(base_raw_value, unit="%", value_kind="percentage")
    return {
        "comparison_base_period": base_column.get("period"),
        "comparison_base_period_year": base_column.get("period_year"),
        "comparison_base_period_quarter": base_column.get("period_quarter"),
        "comparison_base_period_type": base_column.get("period_type"),
        "comparison_base_column_label": base_column.get("label"),
        "comparison_base_raw_value": base_raw_value,
        "comparison_base_value": base_value,
    }


def _table_row_evidence(
    *,
    table_title: str | None,
    context_business_area: str | None,
    metric_parent_label: str | None,
    row_label: str,
    columns: list[dict[str, Any]],
    row_values: list[str],
    unit: str,
    unit_evidence: str | None,
) -> str:
    path = [
        part
        for part in (
            table_title,
            context_business_area,
            metric_parent_label,
            row_label,
        )
        if part
    ]
    series = [
        f"{column['label']} {value}" for column, value in zip(columns, row_values, strict=False)
    ]
    prefix = " > ".join(path) if path else row_label
    unit_part = f"단위: {unit}"
    if unit_evidence:
        unit_part = f"{unit_part} | 단위 근거: {unit_evidence}"
    else:
        unit_part = f"{unit_part} | 단위 근거 없음"
    return f"{prefix} | {' | '.join(series)} ({unit_part})"


def _effective_row_unit_evidence(
    *,
    unit_evidence: str | None,
    row_label: str,
    metric_parent_label: str | None,
    value_kind: str,
) -> str | None:
    if value_kind == "percentage":
        if "%" in (row_label or ""):
            return row_label
        if metric_parent_label and "%" in metric_parent_label:
            return metric_parent_label
    return unit_evidence


def _table_classification_reason(
    *,
    business_area: str | None,
    metric_scope: str,
    row_label: str,
    table_title: str | None,
    context_business_area: str | None,
    metric_parent_label: str | None,
    explicit_metric: bool,
) -> str:
    table_part = f"표 제목 '{table_title}'" if table_title else "표 제목 없음"
    if metric_scope == "company_total":
        if context_business_area:
            return (
                f"{table_part}, 상위 문맥 '{context_business_area}' 아래의 전체/합계성 행 "
                f"'{row_label}'로 판단되어 company_total로 분류"
            )
        if _is_company_total_table_label(row_label):
            return (
                f"{table_part}, 행 라벨 '{row_label}'이 전체/합계성 라벨이라 company_total로 분류"
            )
        if explicit_metric:
            return (
                f"{table_part}, 행 라벨 '{row_label}'이 별도 사업부문명이 아닌 metric 라벨이라 "
                "company_total로 분류"
            )
        return f"{table_part}, 사업부문 라벨이 확인되지 않아 company_total로 분류"

    parent_part = f", 상위 metric '{metric_parent_label}'" if metric_parent_label else ""
    context_part = f", 표 문맥 '{context_business_area}'" if context_business_area else ""
    return (
        f"{table_part}{context_part}{parent_part}, 행 라벨 '{row_label}'을 "
        f"사업부문/서비스 라벨로 판단해 business_area='{business_area}' segment로 분류"
    )


def _table_metric_from_label(line: str) -> tuple[str, str] | None:
    label = re.split(r"\(?-?\d", line or "", maxsplit=1)[0]
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label).lower()
    if _looks_like_table_noise_label(label):
        return None
    if _looks_like_non_target_metric_label(label):
        return None
    for metric_name, aliases, value_kind in _IR_TABLE_METRIC_ALIASES:
        if any(_metric_alias_matches_label(alias, compact) for alias in aliases):
            return metric_name, value_kind
    return None


def _looks_like_non_target_metric_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact:
        return False
    if "ebitda" in compact and ("마진" in compact or "margin" in compact):
        return True
    if "영업이익률" in compact or "operatingmargin" in compact or compact in {"opm", "margin"}:
        return False
    if compact in {"총이익", "총이익률"}:
        return True
    return bool(_IR_TABLE_NON_METRIC_LABEL_PATTERN.search(label or ""))


def _metric_alias_matches_label(alias: str, compact_label: str) -> bool:
    compact_alias = alias.replace(" ", "").lower()
    if not compact_alias:
        return False
    if compact_alias in {"매출", "sales", "op"}:
        return compact_label == compact_alias
    return compact_alias in compact_label


def _is_percentage_metric_label(row_label: str, metric_name: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", row_label or "").lower()
    for candidate_metric, aliases, value_kind in _IR_TABLE_METRIC_ALIASES:
        if candidate_metric != metric_name or value_kind != "percentage":
            continue
        if any(_metric_alias_matches_label(alias, compact) for alias in aliases):
            return True
    return False


def _is_valid_table_metric_row(row_label: str, row_metric: tuple[str, str]) -> bool:
    cleaned_label = re.sub(r"\s+", " ", row_label or "").strip()
    if len(cleaned_label) > 60:
        return False
    if re.search(r"[.!?。]|다$", cleaned_label):
        return False
    if _looks_like_non_target_metric_label(row_label):
        return False

    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", row_label or "").lower()
    if not compact:
        return True
    if _looks_like_table_noise_label(row_label):
        return False

    metric_name, _value_kind = row_metric
    metric_aliases = {
        alias.replace(" ", "").lower()
        for candidate_metric, aliases, _candidate_kind in _IR_TABLE_METRIC_ALIASES
        if candidate_metric == metric_name
        for alias in aliases
    }
    label_without_metric = compact
    for alias in metric_aliases:
        label_without_metric = label_without_metric.replace(alias, "")

    if not label_without_metric:
        return True
    if _is_company_total_table_label(label_without_metric):
        return True
    if len(label_without_metric) > 28 and not _detect_business_area(label_without_metric):
        return False
    return True


def _looks_like_segment_value_row(line: str, expected_count: int) -> bool:
    row_label = _table_row_label(line)
    if _looks_like_non_target_metric_label(row_label) or _looks_like_table_noise_label(row_label):
        return False
    if not _looks_like_business_area_label(row_label):
        return False
    values = _table_row_values(line, expected_count)
    return len(values) >= expected_count


def _looks_like_table_noise_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "")
    if not compact:
        return False
    if re.fullmatch(r"[-+△▲▵▴▽▼()]+", compact):
        return True
    return bool(_IR_TABLE_NOISE_LABEL_PATTERN.search(label or ""))


def _table_row_values(line: str, expected_count: int) -> list[str]:
    values = _IR_TABLE_NUMBER_PATTERN.findall(line or "")
    clean_values = [
        value
        for value in values
        if not re.fullmatch(r"20\d{2}", value.strip())
        and not _looks_like_table_footnote_marker(value)
    ]
    return clean_values[:expected_count]


def _row_has_misaligned_leading_comparison_columns(
    columns: list[dict[str, Any]],
    row_values: list[str],
) -> bool:
    if not columns or not row_values:
        return False
    leading_comparison_count = 0
    for column in columns:
        if column.get("comparison_type"):
            leading_comparison_count += 1
            continue
        break
    if leading_comparison_count == 0:
        return False
    if any(
        not column.get("comparison_target_period") for column in columns[:leading_comparison_count]
    ):
        return True
    leading_values = row_values[:leading_comparison_count]
    return any("%" not in str(value) for value in leading_values)


def _row_values_are_percentage(values: list[str]) -> bool:
    return bool(values) and all(str(value).strip().endswith("%") for value in values)


def _row_values_have_percentage(values: list[str]) -> bool:
    return any(str(value).strip().endswith("%") for value in values)


def _looks_like_table_footnote_marker(value: str) -> bool:
    stripped = value.strip()
    return bool(re.fullmatch(r"\d+\)", stripped))


def _table_row_label(line: str) -> str:
    label = re.split(r"\(?-?\d", line or "", maxsplit=1)[0]
    return re.sub(r"\s+", " ", label).strip(" :-|'\"`‘’“”.,;")


def _table_value(raw_value: str, *, unit: str | None, value_kind: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    sign = -1.0 if value.startswith(("△", "▲")) else 1.0
    value = value.lstrip("△▲")
    has_pct_marker = value.endswith("%")
    if value_kind != "percentage" and has_pct_marker:
        return None
    if value_kind == "percentage" and unit != "%" and not has_pct_marker:
        return None
    if value_kind == "percentage" or has_pct_marker:
        try:
            return sign * float(value.rstrip("%").replace(",", ""))
        except ValueError:
            return None
    if not unit:
        return None
    return sign * _normalize_table_amount_krwbn(value.strip("()"), unit)


def _table_display_unit(*, unit: str | None, value_kind: str) -> str:
    if value_kind == "percentage":
        return "%"
    return unit or "단위 미확인"


def _table_cell_unit(raw_value: str, *, table_unit: str | None, value_kind: str) -> str:
    if value_kind == "percentage" or raw_value.strip().endswith("%"):
        return "%"
    return table_unit or "단위 미확인"


def _normalize_table_amount_krwbn(value: str, unit: str) -> float:
    number = float(value.replace(",", ""))
    if unit in {"조원", "조"}:
        return number * 10_000
    if unit in {"십억원", "KRW bn", "krw bn"}:
        return number * 10
    if unit == "백만원":
        return number / 100
    return number


def _table_unit(text: str) -> str | None:
    value = text or ""
    lowered = value.lower()
    compact = re.sub(r"\s+", "", lowered)
    if (
        "십억원" in compact
        or "십억krw" in compact
        or "krwbn" in compact
        or "krwbillion" in compact
        or "billionkrw" in compact
    ):
        return "십억원"
    if "백만원" in compact or "krwmn" in compact or "krwmillion" in compact:
        return "백만원"
    if "조원" in compact:
        return "조원"
    if "억원" in compact:
        return "억원"
    return None


def _table_unit_evidence(lines: list[str]) -> str | None:
    for line in lines:
        compact = re.sub(r"\s+", "", line.lower())
        if any(
            token in compact
            for token in (
                "단위:",
                "unit:",
                "unitof",
                "억원",
                "조원",
                "백만원",
                "십억원",
                "krwbn",
                "krwmn",
                "krwbillion",
                "krwmillion",
                "billionkrw",
            )
        ):
            return line[:300]
    return None


def _nearby_table_title(lines: list[str], header_index: int) -> str | None:
    for line in reversed(lines[max(0, header_index - 3) : header_index]):
        if _looks_like_unit_line(line):
            continue
        if len(_period_columns_from_header(line)) < 2:
            return line
    return None


def _should_skip_financial_table(
    *,
    peer_id: str | None,
    table_title: str | None,
    lines: list[str],
    header_index: int,
) -> bool:
    if peer_id != "sk_ax":
        return False
    context = " ".join(lines[max(0, header_index - 16) : header_index + 1])
    text = f"{table_title or ''} {context}".lower()
    compact = re.sub(r"\s+", "", text)
    if any(
        token in compact
        for token in (
            "skax",
            "skc&c",
            "sk씨앤씨",
            "it서비스부문",
            "it서비스",
        )
    ):
        return False
    return any(term.lower() in text for term in _IR_PORTFOLIO_TERMS if term != "appendix")


def _looks_like_unit_line(line: str) -> bool:
    compact = re.sub(r"\s+", "", line or "").lower()
    if not compact:
        return False
    return bool(re.fullmatch(r"[\(\[]?단위[:：]?(?:억원|십억원|백만원|조원|%)?[\)\]]?", compact))


def _table_context_business_area(lines: list[str], header_index: int) -> str | None:
    context_lines = lines[max(0, header_index - 6) : header_index]
    for line in reversed(context_lines):
        candidate = _business_area_label_from_context(line)
        if candidate:
            return candidate
    return None


def _business_area_label_from_context(text: str) -> str | None:
    cleaned = _clean_table_context_label(text)
    if not cleaned:
        return None
    if len(cleaned) > 50:
        return None
    if _looks_like_table_noise_label(cleaned):
        return None
    if re.search(r"[.!?。]|다$", cleaned):
        return None
    if _looks_like_financial_table_title(cleaned):
        return None
    detected = _detect_business_area(cleaned)
    if detected:
        return cleaned if _looks_like_business_area_label(cleaned) else detected
    return None


def _clean_table_context_label(text: str) -> str:
    cleaned = re.sub(
        r"\([^)]*(?:단위|unit|krw|억원|십억원|백만원|%).*?\)", "", text or "", flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(?:revenue|sales|operating profit|op|margin|results?|financial)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"매출|영업이익|영업이익률|실적|손익|요약|현황|단위|억원|십억원|백만원", "", cleaned
    )
    return re.sub(r"\s+", " ", cleaned).strip(" :-|")


def _looks_like_financial_table_title(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    return any(
        token in compact
        for token in (
            "financialresults",
            "income",
            "손익",
            "경영실적",
            "실적요약",
            "재무",
            "매출",
            "영업이익",
            "margin",
        )
    )


def _is_historical_period(period: str | None, report_period: str | None) -> bool:
    if not period or not report_period:
        return False
    return period != report_period


def _table_business_area(
    row_label: str,
    page_text: str,
    *,
    peer_id: str | None,
    context_business_area: str | None = None,
    explicit_metric: bool = True,
    active_metric_parent_label: str | None = None,
) -> str | None:
    metric_removed = row_label
    for _metric_name, aliases, _value_kind in _IR_TABLE_METRIC_ALIASES:
        for alias in aliases:
            metric_removed = re.sub(re.escape(alias), "", metric_removed, flags=re.IGNORECASE)
    metric_removed = re.sub(r"\s+", " ", metric_removed).strip(" :-|")
    if _is_non_business_area_label(metric_removed):
        return context_business_area or "company_total"
    if _is_punctuation_only_label(metric_removed):
        return context_business_area or "company_total"
    if not explicit_metric and _looks_like_business_area_label(row_label):
        return row_label
    if _is_company_total_table_label(metric_removed):
        return context_business_area or "company_total"
    if active_metric_parent_label and metric_removed == active_metric_parent_label:
        return context_business_area or "company_total"
    if metric_removed and _detect_business_area(metric_removed):
        return metric_removed
    if metric_removed and _looks_like_business_area_label(metric_removed):
        return metric_removed
    return _detect_business_area(metric_removed) or context_business_area or "company_total"


def _is_company_total_table_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact or _is_punctuation_only_label(label):
        return True
    return compact in {
        "합계",
        "총계",
        "계",
        "전체",
        "전사",
        "연결",
        "별도",
        "total",
        "subtotal",
        "companytotal",
        "consolidated",
    }


def _looks_like_business_area_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    if not compact or _is_company_total_table_label(label):
        return False
    if _is_non_business_area_label(label):
        return False
    if _is_punctuation_only_label(label):
        return False
    if re.fullmatch(r"[-+%.,\d]+", compact):
        return False
    metric_aliases = {
        alias.replace(" ", "").lower()
        for _metric_name, aliases, _value_kind in _IR_TABLE_METRIC_ALIASES
        for alias in aliases
    }
    return compact not in metric_aliases


def _is_non_business_area_label(label: str) -> bool:
    compact = re.sub(r"[\sㆍ·\[\]\(\)]", "", label or "").lower()
    return compact in {
        "fy",
        "year",
        "annual",
        "profit",
        "operating",
        "operatingprofit",
        "revenue",
        "sales",
    }


def _is_punctuation_only_label(label: str) -> bool:
    compact = re.sub(r"\s+", "", label or "")
    if not compact:
        return True
    return not bool(re.search(r"[A-Za-z가-힣]", compact))
