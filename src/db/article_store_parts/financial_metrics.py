"""article_store_parts financial_metrics — extracted from facade (move-only)."""

from __future__ import annotations

import json
from typing import Any

from src.crawler.base import RawArticle
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


def upsert_raw_article_financial_metrics(
    metrics: list[dict[str, Any]],
) -> int:
    """IR/DART 분석에서 추출한 숫자형 fact를 upsert한다."""
    if not metrics:
        return 0

    with SessionLocal() as db:
        for metric in metrics:
            db.execute(_UPSERT_FINANCIAL_METRIC_SQL, _financial_metric_params(metric))
        db.commit()

    return len(metrics)


def ensure_market_price_ohlcv_schema() -> None:
    """주가 OHLCV 정규화 테이블을 보장한다."""
    with SessionLocal() as db:
        db.execute(_ENSURE_MARKET_PRICE_OHLCV_SQL)
        db.execute(_ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL)
        db.commit()


def _upsert_market_price_ohlcv_for_article(db, article_id: int, article: RawArticle) -> int:
    rows = _market_price_ohlcv_rows(article_id, article)
    if not rows:
        return 0

    db.execute(_ENSURE_MARKET_PRICE_OHLCV_SQL)
    db.execute(_ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL)
    for row in rows:
        db.execute(_UPSERT_MARKET_PRICE_OHLCV_SQL, row)
    return len(rows)


def _market_price_ohlcv_rows(article_id: int, article: RawArticle) -> list[dict[str, Any]]:
    if article.source_type != "market_data":
        return []

    data = article.extra.get("data")
    if not isinstance(data, list):
        return []

    ticker = str(article.extra.get("ticker") or "").strip()
    if not ticker:
        ticker = _ticker_from_market_data_url(article.url) or ""
    if not ticker:
        return []

    peer_id = article.peer_id or (article.company[0] if article.company else None)
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        trade_date = _parse_market_trade_date(item.get("date"))
        if trade_date is None:
            continue
        rows.append(
            {
                "raw_article_id": article_id,
                "peer_id": peer_id,
                "ticker": ticker,
                "trade_date": trade_date,
                "open": _market_number(item.get("open")),
                "high": _market_number(item.get("high")),
                "low": _market_number(item.get("low")),
                "close": _market_number(item.get("close")),
                "volume": _market_int(item.get("volume")),
                "change_pct": _market_number(item.get("change_pct")),
                "currency": str(article.extra.get("currency") or "KRW"),
                "source_type": article.source_type,
                "source_name": article.source_name,
                "publisher": article.publisher,
                "collected_at": article.collected_at,
                "payload": json.dumps(_sanitize_jsonish(item), ensure_ascii=False),
            }
        )
    return rows


def delete_raw_article_financial_metrics(
    raw_article_ids: list[int],
    *,
    source_type: str | None = None,
) -> int:
    """선택한 원문 기사에 연결된 숫자형 fact를 삭제한다."""
    if not raw_article_ids:
        return 0

    with SessionLocal() as db:
        result = db.execute(
            _DELETE_FINANCIAL_METRICS_SQL,
            {
                "raw_article_ids": raw_article_ids,
                "source_type": source_type,
            },
        )
        deleted_rows = result.fetchall()
        db.commit()

    return len(deleted_rows)


def _financial_metric_params(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_article_id": metric["raw_article_id"],
        "metric_uid": metric["metric_uid"],
        "source_type": metric["source_type"],
        "source_name": metric.get("source_name"),
        "peer_id": metric.get("peer_id"),
        "period": metric.get("period"),
        "period_year": metric.get("period_year"),
        "period_quarter": metric.get("period_quarter"),
        "period_type": metric.get("period_type"),
        "metric_name": metric["metric_name"],
        "metric_label": metric.get("metric_label"),
        "metric_scope": metric.get("metric_scope"),
        "business_area": metric.get("business_area"),
        "value_numeric": metric.get("value_numeric"),
        "value_krwbn": metric.get("value_krwbn"),
        "value_krw": metric.get("value_krw"),
        "unit": metric.get("unit"),
        "currency": metric.get("currency", "KRW"),
        "source_page": metric.get("source_page"),
        "source_table_uid": metric.get("source_table_uid"),
        "source_chunk_uid": metric.get("source_chunk_uid"),
        "confidence": metric.get("confidence"),
        "extraction_method": metric.get("extraction_method"),
        "evidence_text": metric.get("evidence_text"),
        "payload": json.dumps(
            _sanitize_jsonish(metric.get("payload") or {}),
            ensure_ascii=False,
        ),
    }
