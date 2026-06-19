"""article_store_parts business_signals — extracted from facade (move-only)."""

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
from src.db.article_store_parts.dart_documents import (  # noqa: F401
    _dart_document_summary,
    get_dart_document_detail,
    list_dart_documents,
)
from src.db.article_store_parts.financial_metrics import (  # noqa: F401
    _financial_metric_params,
    _market_price_ohlcv_rows,
    _upsert_market_price_ohlcv_for_article,
    delete_raw_article_financial_metrics,
    ensure_market_price_ohlcv_schema,
    upsert_raw_article_financial_metrics,
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
from src.db.article_store_parts.pipeline_log import (  # noqa: F401
    save_pipeline_log,
)
from src.db.postgres import SessionLocal


def upsert_raw_article_business_signals(
    signals: list[dict[str, Any]],
) -> int:
    """IR/DART 분석에서 추출한 사업/전략/리스크 신호를 upsert한다."""
    if not signals:
        return 0

    with SessionLocal() as db:
        for signal in signals:
            db.execute(_UPSERT_BUSINESS_SIGNAL_SQL, _business_signal_params(signal))
        db.commit()

    return len(signals)


def delete_raw_article_business_signals(
    raw_article_ids: list[int],
    *,
    source_type: str | None = None,
) -> int:
    """선택한 원문 기사에 연결된 사업/전략/리스크 신호를 삭제한다."""
    if not raw_article_ids:
        return 0

    with SessionLocal() as db:
        result = db.execute(
            _DELETE_BUSINESS_SIGNALS_SQL,
            {
                "raw_article_ids": raw_article_ids,
                "source_type": source_type,
            },
        )
        deleted_rows = result.fetchall()
        db.commit()

    return len(deleted_rows)


def _business_signal_params(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_article_id": signal["raw_article_id"],
        "signal_uid": signal["signal_uid"],
        "source_type": signal["source_type"],
        "source_name": signal.get("source_name"),
        "peer_id": signal.get("peer_id"),
        "period": signal.get("period"),
        "period_year": signal.get("period_year"),
        "period_quarter": signal.get("period_quarter"),
        "period_type": signal.get("period_type"),
        "business_area": signal["business_area"],
        "signal_type": signal["signal_type"],
        "sentiment": signal.get("sentiment"),
        "summary": signal["summary"],
        "evidence_text": signal.get("evidence_text"),
        "source_page": signal.get("source_page"),
        "source_chunk_uid": signal.get("source_chunk_uid"),
        "confidence": signal.get("confidence"),
        "extraction_method": signal.get("extraction_method"),
        "payload": json.dumps(
            _sanitize_jsonish(signal.get("payload") or {}),
            ensure_ascii=False,
        ),
    }
