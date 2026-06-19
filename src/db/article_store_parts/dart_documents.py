"""article_store_parts dart_documents — extracted from facade (move-only)."""

from __future__ import annotations

import json
from typing import Any

from src.db.article_store_parts._constants import (  # noqa: F401
    _CONTROL_CHAR_RE,
    _DELETE_BUSINESS_SIGNALS_SQL,
    _DELETE_FINANCIAL_METRICS_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_SQL,
    _GLOBAL_TREND_UPSERT_SQL,
    _INDUSTRY_MARKER_KEYS,
    _INDUSTRY_REPORT_TYPES,
    _INDUSTRY_SOURCE_TYPES,
    _INSERT_EVIDENCE,
    _INSERT_PIPELINE_LOG,
    _PARSE_RESULT_METADATA_KEYS,
    _SELECT_ARTICLE_ID_BY_URL,
    _SOURCE_METADATA_EXCLUDED_KEYS,
    _UPSERT_BUSINESS_SIGNAL_SQL,
    _UPSERT_FINANCIAL_METRIC_SQL,
    _UPSERT_MARKET_PRICE_OHLCV_SQL,
    DEFAULT_PEER_COMPANY_IDS,
    GLOBAL_NEWSROOM_SOURCE_NAMES,
    GLOBAL_RESEARCH_SOURCE_NAMES,
    INDUSTRY_TREND_COMPANY,
    SK_AX_RAW_SOURCE_NAMES,
)
from src.db.article_store_parts.article_retrieval import (  # noqa: F401
    article_exists_by_url,
)
from src.db.article_store_parts.helpers import (  # noqa: F401
    _article_company_ids,
    _article_matches_company,
    _build_evidence_payload,
    _canonical_card_id,
    _card_company_filter_id,
    _card_created_at_param,
    _collection_metadata,
    _company_match_aliases,
    _company_match_text,
    _date_string_or_none,
    _fetch_dart_rows,
    _filter_articles_for_card_company,
    _generate_card_id,
    _ilike_any_clause,
    _integrated_issue_context_lines,
    _integrated_issue_from_card_context,
    _is_industry_trend_article,
    _is_missing_card_news_articles_table,
    _is_missing_column,
    _is_missing_relation,
    _is_self_company_card,
    _is_undefined_column_error,
    _is_valid,
    _iso_or_none,
    _join_issue_lines,
    _json_or_value,
    _labeled_lines_from_main_detail_blocks,
    _looks_like_fragment,
    _main_detail_blocks,
    _main_detail_blocks_from_labeled_lines,
    _market_int,
    _market_number,
    _merge_source_dicts,
    _metadata_dict,
    _metadata_json,
    _normalize_int_list,
    _normalize_peer_company_fk,
    _parse_datetime,
    _parse_market_trade_date,
    _parse_result_payload,
    _resolve_integrated_issue_id,
    _resolve_peer_company_id,
    _sanitize_jsonish,
    _sanitize_text,
    _sentence_text,
    _source_article_dict_from_article,
    _source_articles_payload,
    _source_basis_datetime,
    _source_dict_from_article,
    _source_ids_from_sources,
    _source_metadata_json,
    _string_list,
    _ticker_from_market_data_url,
    _update_company_analysis_if_changed,
    _update_dart_content_if_better,
    _update_ir_content_if_better,
    _upsert_crawl_run_article,
    _upsert_parse_result,
    _upsert_source_metadata,
    fetch_domestic_trend_representative_inputs,
    fetch_global_trend_inputs,
    fetch_peer_cards_for_alignment,
    fetch_previous_trend_context_for_delta,
    fetch_sk_ax_raw_for_alignment,
    get_articles_by_ids,
    upsert_global_industry_trends,
)


def list_dart_documents(
    *,
    company: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """프론트 DART 문서 목록에 필요한 파싱 요약을 조회한다."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    company_filter = "AND (:company IS NULL OR r.company @> CAST(:company_json AS jsonb))"
    rows = _fetch_dart_rows(
        select_sql="""
            SELECT r.id, r.company, r.title, r.url, r.published_at, r.collected_at,
                   r.processing_status, r.metadata,
                   COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
                   COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
                   COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
            FROM raw_articles r
            LEFT JOIN raw_article_parse_results pr
                ON pr.raw_article_id = r.id
        """,
        where_sql=company_filter,
        order_limit_sql="""
            ORDER BY published_at DESC NULLS LAST, collected_at DESC
            LIMIT :limit OFFSET :offset
        """,
        params={
            "company": company,
            "company_json": json.dumps([company], ensure_ascii=False) if company else "[]",
            "limit": limit,
            "offset": offset,
        },
    )
    return [_dart_document_summary(dict(row._mapping)) for row in rows]


def get_dart_document_detail(article_id: int) -> dict[str, Any] | None:
    """DART 문서 상세 페이지에 필요한 파싱 결과를 조회한다."""
    rows = _fetch_dart_rows(
        select_sql="""
            SELECT r.id, r.company, r.title, r.content, r.url, r.published_at, r.collected_at,
                   r.processing_status, r.metadata,
                   COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
                   COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
                   COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
            FROM raw_articles r
            LEFT JOIN raw_article_parse_results pr
                ON pr.raw_article_id = r.id
        """,
        where_sql="AND r.id = :id",
        order_limit_sql="LIMIT 1",
        params={"id": article_id},
    )
    if not rows:
        return None

    row = dict(rows[0]._mapping)
    metadata = _metadata_dict(row.get("metadata"))
    parser_result = _metadata_dict(row.get("parser_result") or metadata.get("parser_result"))
    document = _metadata_dict(parser_result.get("document"))
    return {
        **_dart_document_summary(row),
        "content": row.get("content"),
        "document": document,
        "sections": parser_result.get("sections") or metadata.get("dart_sections") or {},
        "section_tree": parser_result.get("section_tree")
        or metadata.get("dart_section_tree")
        or [],
        "document_chunks": parser_result.get("document_chunks")
        or metadata.get("dart_document_chunks")
        or [],
        "classified_tables": parser_result.get("classified_tables")
        or metadata.get("dart_classified_tables")
        or [],
        "financial_statements": parser_result.get("financial_statements")
        or metadata.get("dart_financial_statements")
        or [],
        "topic_signals": metadata.get("topic_signals") or parser_result.get("topic_signals") or [],
        "warnings": row.get("parser_warnings") or parser_result.get("warnings") or [],
    }


def _dart_document_summary(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata_dict(row.get("metadata"))
    parser_result = _metadata_dict(row.get("parser_result") or metadata.get("parser_result"))
    financial_record = _metadata_dict(
        row.get("financial_record")
        or metadata.get("financial_record")
        or parser_result.get("financial_record")
    )
    company = row.get("company")
    return {
        "id": row.get("id"),
        "company": company if isinstance(company, list) else _json_or_value(company, []),
        "title": row.get("title"),
        "url": row.get("url"),
        "published_at": _iso_or_none(row.get("published_at")),
        "collected_at": _iso_or_none(row.get("collected_at")),
        "processing_status": row.get("processing_status"),
        "rcept_no": metadata.get("rcept_no") or metadata.get("receipt_no"),
        "corp_code": metadata.get("corp_code"),
        "corp_name": metadata.get("corp_name"),
        "report_name": metadata.get("report_name") or metadata.get("report_nm") or row.get("title"),
        "period": metadata.get("period") or parser_result.get("period"),
        "period_year": metadata.get("period_year") or parser_result.get("period_year"),
        "period_quarter": metadata.get("period_quarter") or parser_result.get("period_quarter"),
        "period_type": metadata.get("period_type") or parser_result.get("period_type"),
        "document_text_length": metadata.get("document_text_length"),
        "table_count": metadata.get("table_count"),
        "structured_table_count": metadata.get("structured_table_count"),
        "section_count": len(metadata.get("dart_sections") or parser_result.get("sections") or {}),
        "financial_statement_count": financial_record.get("financial_statement_count")
        or len(metadata.get("dart_financial_statements") or []),
        "revenue_total_krwbn": financial_record.get("revenue_total_krwbn")
        or parser_result.get("revenue_total_krwbn"),
        "operating_profit_krwbn": financial_record.get("operating_profit_krwbn")
        or parser_result.get("operating_profit_krwbn"),
    }
