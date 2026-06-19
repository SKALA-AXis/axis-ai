"""article_store_parts evidence_chain — extracted from facade (move-only)."""

from __future__ import annotations

import json
import logging
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
from src.db.article_store_parts.business_signals import (  # noqa: F401
    _business_signal_params,
    delete_raw_article_business_signals,
    upsert_raw_article_business_signals,
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

log = logging.getLogger(__name__)


def save_evidence_chain(
    card_news_id: str,
    chain: dict[str, Any],
    passed: bool,
    missing: list[str],
) -> None:
    """evidence_chain 테이블에 검증 체인 4종을 저장한다."""
    if not card_news_id:
        return
    try:
        with SessionLocal() as db:
            db.execute(
                _INSERT_EVIDENCE,
                {
                    "card_news_id": card_news_id,
                    "source_links": json.dumps(chain.get("source_links", []), ensure_ascii=False),
                    "provenance": json.dumps(chain.get("provenance", {}), ensure_ascii=False),
                    "financial_refs": json.dumps(
                        chain.get("financial_refs", []), ensure_ascii=False
                    ),
                    "mbb_refs": json.dumps(chain.get("mbb_refs", []), ensure_ascii=False),
                    "financial_link": json.dumps(
                        chain.get("financial_link", {}), ensure_ascii=False
                    ),
                    "evidence_version": chain.get("provenance", {}).get("evidence_version", "v3.0"),
                    "passed": passed,
                    "missing": missing,
                },
            )
            db.commit()
    except Exception as e:
        log.error("evidence_chain 저장 실패 | card_id=%s error=%s", card_news_id, e)
