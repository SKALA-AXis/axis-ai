# 작성일: 2026-05-12
# 작성자: 박지원
# 변경이력:
#   2026-05-12 박지원 — IR 파서 작성 및 전처리·peer profile·스코프 분석 보강
#   2026-05-12 심유정 — feature 머지분 lint·mypy 정리 및 카드뉴스 에이전트 플로우 정리
"""IR PDF 크롤링 결과를 재무 후보 레코드로 변환하는 deterministic parser.

IRCrawler는 PDF 파일을 직접 저장하지 않고 RawArticle 형태로 본문 텍스트와
PDF 페이지 블록을 담는다. 이 파서는 그 RawArticle 결과를 받아
peer_financials 적재에 사용할 수 있는 핵심 재무 후보를 만든다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.crawler.parsers.pdf_payload import extract_pdf_payload
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
from src.parsers.ir._table_rows import (  # noqa: F401
    _block_center,
    _business_area_label_from_context,
    _clean_table_context_label,
    _comparison_base_cell,
    _effective_row_unit_evidence,
    _extract_chart_block_candidates,
    _extract_financial_table_candidates,
    _is_company_total_table_label,
    _is_historical_period,
    _is_non_business_area_label,
    _is_percentage_metric_label,
    _is_punctuation_only_label,
    _is_valid_table_metric_row,
    _looks_like_business_area_label,
    _looks_like_financial_table_title,
    _looks_like_non_target_metric_label,
    _looks_like_segment_value_row,
    _looks_like_table_footnote_marker,
    _looks_like_table_noise_label,
    _looks_like_unit_line,
    _merge_split_operating_margin_lines,
    _metric_alias_matches_label,
    _nearby_table_title,
    _normalize_table_amount_krwbn,
    _row_has_misaligned_leading_comparison_columns,
    _row_values_are_percentage,
    _row_values_have_percentage,
    _should_skip_financial_table,
    _table_business_area,
    _table_cell_unit,
    _table_classification_reason,
    _table_context_business_area,
    _table_display_unit,
    _table_metric_from_label,
    _table_row_evidence,
    _table_row_label,
    _table_row_values,
    _table_unit,
    _table_unit_evidence,
    _table_value,
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

log = logging.getLogger(__name__)


def _candidate_page(candidates: list[dict[str, Any]], metric_type: str) -> int | None:
    for candidate in candidates:
        if candidate.get("type") == metric_type:
            page = candidate.get("page")
            return int(page) if isinstance(page, int) else None
    return None


def _build_financial_record(
    *,
    source: str,
    peer_id: str | None,
    period: str | None,
    title: str,
    url: str,
    published_at: str | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_details = {
        str(candidate["type"]): {
            key: value
            for key, value in candidate.items()
            if key
            in {
                "page",
                "raw",
                "value_krwbn",
                "value_pct",
                "value_kind",
                "metric_scope",
                "business_area",
                "entity_name",
                "confidence",
                "source",
                "source_table_uid",
                "table_title",
                "metric_parent_label",
                "row_label",
                "column_label",
                "row_evidence",
                "evidence_text",
                "classification_reason",
                "context_evidence",
                "base_metric_type",
                "comparison_type",
                "unit",
                "unit_evidence",
                "period",
                "period_year",
                "period_quarter",
                "period_type",
                "is_historical",
                "comparison_base_period",
                "comparison_base_period_year",
                "comparison_base_period_quarter",
                "comparison_base_period_type",
                "comparison_base_column_label",
                "comparison_base_raw_value",
                "comparison_base_value",
            }
        }
        for candidate in candidates
        if candidate.get("type")
    }
    accepted_scopes = {"company_total", "segment"}
    revenue_total = _candidate_value(
        candidates,
        "revenue_total",
        "value_krwbn",
        allowed_scopes=accepted_scopes,
    )
    operating_profit = _candidate_value(
        candidates,
        "operating_profit",
        "value_krwbn",
        allowed_scopes=accepted_scopes,
    )
    return {
        "peer_id": peer_id,
        "period": period,
        "revenue_total_krwbn": revenue_total,
        "operating_profit_krwbn": operating_profit,
        "net_income_krwbn": _candidate_value(
            candidates,
            "net_income",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "ebitda_krwbn": _candidate_value(
            candidates,
            "ebitda",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "backlog_krwbn": _candidate_value(
            candidates,
            "backlog",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "orders_krwbn": _candidate_value(
            candidates,
            "orders",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "capex_krwbn": _candidate_value(
            candidates,
            "capex",
            "value_krwbn",
            allowed_scopes=accepted_scopes,
        ),
        "operating_margin_pct": _candidate_value(
            candidates,
            "operating_margin",
            "value_pct",
            allowed_scopes=accepted_scopes,
        ),
        "metric_details": metric_details,
        "source": source,
        "title": title,
        "url": url,
        "published_at": published_at,
        "ir_page": _candidate_page(candidates, "revenue_total")
        or _candidate_page(candidates, "operating_profit"),
    }


def _candidate_value(
    candidates: list[dict[str, Any]],
    metric_type: str,
    value_key: str,
    *,
    allowed_scopes: set[str] | None = None,
) -> float | None:
    matching = [
        candidate
        for candidate in candidates
        if candidate.get("type") == metric_type
        and (allowed_scopes is None or candidate.get("metric_scope") in allowed_scopes)
    ]
    for candidate in sorted(matching, key=lambda item: bool(item.get("is_historical"))):
        value = candidate.get(value_key)
        if isinstance(value, int | float):
            return float(value)
    return None


class IRParser:
    """IR RawArticle에서 핵심 재무 지표 후보를 추출한다."""

    def parse_article(
        self,
        article: Any,
        *,
        include_raw_pages: bool = False,
    ) -> dict[str, Any]:
        """IRCrawler가 만든 RawArticle 또는 dict 결과를 파싱한다."""
        extra = _article_extra(article)
        text = str(_article_get(article, "content", "") or "")
        pages = _pages_from_ir_article(article, extra)
        peer_id = _article_peer_id(article)
        pages = _filter_pages_for_peer(pages, peer_id)
        title = str(_article_get(article, "title", "") or "")
        url = str(_article_get(article, "url", "") or extra.get("pdf_url", "") or "")
        published_at = _article_published_at(article, extra)

        warnings: list[str] = []
        candidates: list[dict[str, Any]] = []
        period = _period_from_ir_article(article, extra, text)
        period_year, period_quarter, period_type = _period_parts(period)
        sections, document_chunks, topics, topic_signals = _extract_sections_and_chunks(
            pages,
            peer_id=peer_id,
        )
        table_candidates, financial_tables = _extract_financial_table_candidates(
            pages,
            report_period=period,
            peer_id=peer_id,
        )
        candidates.extend(table_candidates)
        candidates.extend(
            _extract_chart_block_candidates(
                pages,
                report_period=period,
                peer_id=peer_id,
            )
        )
        seen_metric_candidates: set[tuple[str, int | None, str]] = set()

        for page in pages:
            page_no = page.get("page")
            page_text = str(page.get("text", "") or "")

            if not period:
                period = _extract_period(page_text)
                period_year, period_quarter, period_type = _period_parts(period)

            for metric_type, patterns, value_kind in _IR_FINANCIAL_METRIC_RULES:
                for value, raw in _metric_values(page_text, patterns, value_kind):
                    dedupe_key = (metric_type, page_no if isinstance(page_no, int) else None, raw)
                    if dedupe_key in seen_metric_candidates:
                        continue
                    seen_metric_candidates.add(dedupe_key)
                    value_key = _candidate_value_key(value_kind)
                    if _has_same_table_metric_candidate(
                        candidates,
                        page_no=page_no,
                        metric_type=metric_type,
                        value_key=value_key,
                        value=value,
                    ):
                        continue
                    metric_context = _classify_metric_context(
                        page_text,
                        peer_id=peer_id,
                        raw_match=raw,
                    )
                    candidate = {
                        "page": page_no,
                        "type": metric_type,
                        "value_kind": value_kind,
                        value_key: value,
                        "raw": raw,
                        **metric_context,
                    }
                    candidates.append(candidate)
                    candidates.extend(
                        _inline_comparison_candidates(
                            page_text=page_text,
                            page_no=page_no,
                            metric_type=metric_type,
                            raw=raw,
                            base_candidate=metric_context,
                            report_period=period,
                        )
                    )

        revenue_total = _candidate_value(
            candidates,
            "revenue_total",
            "value_krwbn",
            allowed_scopes={"company_total", "segment"},
        )
        operating_profit = _candidate_value(
            candidates,
            "operating_profit",
            "value_krwbn",
            allowed_scopes={"company_total", "segment"},
        )

        if not period:
            warnings.append("period 추출 실패")
        if revenue_total is None:
            warnings.append("revenue_total 추출 실패")
        if operating_profit is None:
            warnings.append("operating_profit 추출 실패")

        financial_record = _build_financial_record(
            source="ir",
            peer_id=peer_id,
            period=period,
            title=title,
            url=url,
            published_at=published_at,
            candidates=candidates,
        )

        result = {
            "ok": bool(text),
            "source": "ir",
            "peer_id": peer_id,
            "title": title,
            "url": url,
            "published_at": published_at,
            "period": period,
            "period_year": period_year,
            "period_quarter": period_quarter,
            "period_type": period_type,
            "revenue_total_krwbn": revenue_total,
            "operating_profit_krwbn": operating_profit,
            "net_income_krwbn": financial_record.get("net_income_krwbn"),
            "ebitda_krwbn": financial_record.get("ebitda_krwbn"),
            "backlog_krwbn": financial_record.get("backlog_krwbn"),
            "orders_krwbn": financial_record.get("orders_krwbn"),
            "capex_krwbn": financial_record.get("capex_krwbn"),
            "operating_margin_pct": financial_record.get("operating_margin_pct"),
            "candidates": candidates,
            "financial_tables": financial_tables,
            "sections": sections,
            "page_index": _build_page_index(pages, document_chunks),
            "document_chunks": document_chunks,
            "topics": topics,
            "topic_signals": topic_signals,
            "metadata": {
                "source_page": extra.get("source_page"),
                "detail_url": extra.get("detail_url"),
                "pdf_url": extra.get("pdf_url") or url,
                "pdf_pages": extra.get("pdf_pages"),
                "pdf_parsed_pages": extra.get("pdf_parsed_pages"),
                "pdf_text_chars": extra.get("pdf_text_chars") or len(text),
                "pdf_parse_strategy": extra.get("pdf_parse_strategy"),
                "table_parse_strategy": extra.get("table_parse_strategy"),
                "chart_parse_strategy": extra.get("chart_parse_strategy"),
                "contains_images": extra.get("contains_images"),
                "image_count": extra.get("image_count"),
                "drawing_count": extra.get("drawing_count"),
                "ocr_applied": extra.get("ocr_applied"),
                "ocr_pages": extra.get("ocr_pages"),
            },
            "financial_record": financial_record,
            "warnings": warnings,
        }

        if include_raw_pages:
            result["raw_text_pages"] = pages

        log.info(
            "IR article 파싱 완료 | peer_id=%s period=%s rev=%s op=%s",
            peer_id,
            period,
            revenue_total,
            operating_profit,
        )
        return result

    def parse(
        self,
        pdf_path: str | Path,
        *,
        peer_id: str | None = None,
        max_pages: int = 30,
    ) -> dict[str, Any]:
        """PDF 파일 경로 기반 파싱. 이전 PoC 코드 호환용."""
        path = Path(pdf_path)
        if not path.exists():
            return {"ok": False, "reason": f"파일 없음: {path}", "warnings": []}

        try:
            with path.open("rb") as pdf_file:
                payload = extract_pdf_payload(pdf_file.read())
        except Exception as exc:
            return {"ok": False, "reason": f"PDF 열기 실패: {exc}", "warnings": []}

        pages_payload = payload.get("pages", [])[:max_pages]
        pages = [str(page.get("text") or "") for page in pages_payload if isinstance(page, dict)]

        article = {
            "title": path.name,
            "url": str(path),
            "content": "\n".join(f"[PAGE {idx + 1}]\n{text}" for idx, text in enumerate(pages)),
            "peer_id": peer_id,
            "extra": {
                "pdf_pages": payload.get("page_count"),
                "pdf_parsed_pages": len(pages),
                "pdf_page_blocks": pages_payload,
                "pdf_text_chars": len(payload.get("text") or ""),
                "pdf_parse_strategy": payload.get("pdf_parse_strategy"),
                "table_parse_strategy": payload.get("table_parse_strategy"),
                "chart_parse_strategy": payload.get("chart_parse_strategy"),
                "contains_images": payload.get("contains_images"),
                "image_count": payload.get("image_count"),
                "ocr_applied": payload.get("ocr_applied"),
                "ocr_pages": payload.get("ocr_pages"),
            },
        }
        result = self.parse_article(article, include_raw_pages=True)
        result["page_count"] = len(pages)
        return result


__all__ = [
    "IRParser",
    "_OPERATING_PROFIT_PATTERNS",
    "_REVENUE_PATTERNS",
    "_extract_period",
    "_first_amount",
    "_normalize_amount_krwbn",
]
