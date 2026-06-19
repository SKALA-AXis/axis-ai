"""article_store_parts cluster_operations — extracted from facade (move-only)."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import text

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
from src.db.article_store_parts.evidence_chain import (  # noqa: F401
    save_evidence_chain,
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


def list_card_news_cluster_candidates(
    limit: int = 10,
    today_only: bool = False,
) -> list[dict[str, Any]]:
    """프론트 카드뉴스 생성을 위한 뉴스 대표 클러스터 후보를 조회한다."""
    limit = max(1, min(limit, 30))
    where_today = (
        """
          AND EXISTS (
              SELECT 1
              FROM raw_articles recent
              WHERE recent.cluster_id = r.cluster_id
                AND recent.source_type IN ('news', 'official')
                AND recent.published_at >= NOW() - INTERVAL '24 hours'
          )
        """
        if today_only
        else ""
    )
    query = text(f"""
        SELECT
            r.cluster_id,
            r.id AS representative_id,
            r.company,
            r.title,
            r.url,
            r.importance_level,
            r.importance_score,
            r.published_at,
            r.collected_at,
            COALESCE(
                array_agg(
                    a.id
                    ORDER BY
                        a.published_at DESC NULLS LAST,
                        a.collected_at DESC NULLS LAST,
                        a.id DESC
                )
                    FILTER (WHERE a.id IS NOT NULL),
                ARRAY[]::bigint[]
            ) AS article_ids,
            COUNT(a.id) AS cluster_size,
            MAX(a.published_at) AS latest_published_at,
            MAX(a.collected_at) AS latest_collected_at
        FROM raw_articles r
        LEFT JOIN raw_articles a
            ON a.cluster_id = r.cluster_id
           AND a.source_type IN ('news', 'official')
           AND a.company = r.company
        WHERE r.source_type IN ('news', 'official')
          AND r.is_representative = true
          AND r.cluster_id IS NOT NULL
          AND r.processing_status IN ('PROCESSED', 'CLASSIFIED')
          {where_today}
        GROUP BY
            r.cluster_id, r.id, r.company, r.title, r.url,
            r.importance_level, r.importance_score,
            r.published_at, r.collected_at
        ORDER BY
            COALESCE(r.importance_score, 0) DESC,
            MAX(a.published_at) DESC NULLS LAST,
            MAX(a.collected_at) DESC,
            r.published_at DESC NULLS LAST
        LIMIT :limit
    """)
    with SessionLocal() as db:
        rows = db.execute(query, {"limit": limit}).fetchall()
    return [dict(row._mapping) for row in rows]


def list_existing_news_cluster_candidates(
    *,
    company_keys: list[str] | None = None,
    exclude_article_ids: list[int] | None = None,
    lookback_hours: int = 168,
    published_window: tuple[str, str] | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """최근 뉴스 대표 클러스터 후보를 조회한다.

    실시간/백필 전처리에서 새 기사 클러스터를 기존 클러스터에 붙일 때 사용한다.
    cluster_id는 현재 대표 기사 ID 기반으로 저장되므로, 대표 row를 anchor로 삼는다.
    """
    lookback_hours = max(1, min(lookback_hours, 24 * 30))
    limit = max(1, min(limit, 1000))
    company_filter = bool(company_keys)
    exclude_ids = exclude_article_ids or []
    published_since, published_until = published_window or (None, None)

    query = text("""
        SELECT
            r.cluster_id,
            r.id AS representative_id,
            r.company,
            r.title,
            r.content,
            r.url,
            r.source_type,
            r.content_type,
            r.publisher,
            r.language,
            r.relevance_score,
            r.relevance_label,
            r.relevance_reason,
            r.matched_companies,
            r.matched_sectors,
            r.matched_sector_details,
            r.source_name,
            r.published_at,
            r.collected_at,
            r.metadata,
            COALESCE(
                array_agg(
                    a.id
                    ORDER BY
                        a.published_at DESC NULLS LAST,
                        a.collected_at DESC NULLS LAST,
                        a.id DESC
                ) FILTER (WHERE a.id IS NOT NULL),
                ARRAY[]::bigint[]
            ) AS article_ids
        FROM raw_articles r
        LEFT JOIN raw_articles a
            ON a.cluster_id = r.cluster_id
           AND a.source_type = 'news'
        WHERE r.source_type = 'news'
          AND r.is_representative = true
          AND r.cluster_id IS NOT NULL
          AND r.processing_status IN ('PROCESSED', 'CLASSIFIED')
          AND (
              (
                  :published_since IS NOT NULL
                  AND r.published_at >= CAST(:published_since AS timestamptz)
                  AND (
                      :published_until IS NULL
                      OR r.published_at < CAST(:published_until AS timestamptz)
                  )
              )
              OR (
                  :published_since IS NULL
                  AND r.collected_at >= NOW() - (:lookback_hours * INTERVAL '1 hour')
              )
          )
          AND (NOT :company_filter OR r.company ?| :company_keys)
          AND (
              cardinality(CAST(:exclude_article_ids AS bigint[])) = 0
              OR r.id <> ALL(CAST(:exclude_article_ids AS bigint[]))
          )
        GROUP BY
            r.cluster_id, r.id, r.company, r.title, r.content, r.url,
            r.source_type, r.content_type, r.publisher, r.language,
            r.relevance_score, r.relevance_label, r.relevance_reason,
            r.matched_companies, r.matched_sectors, r.matched_sector_details,
            r.source_name, r.published_at, r.collected_at, r.metadata
        ORDER BY
            r.published_at DESC NULLS LAST,
            r.collected_at DESC,
            r.id DESC
        LIMIT :limit
    """)
    fallback_query = text("""
        SELECT
            r.cluster_id,
            r.id AS representative_id,
            r.company,
            r.title,
            r.content,
            r.url,
            r.source_type,
            r.content_type,
            r.publisher,
            r.language,
            r.relevance_score,
            r.relevance_label,
            r.relevance_reason,
            r.matched_companies,
            r.matched_sectors,
            '{}'::jsonb AS matched_sector_details,
            r.source_name,
            r.published_at,
            r.collected_at,
            r.metadata,
            COALESCE(
                array_agg(
                    a.id
                    ORDER BY
                        a.published_at DESC NULLS LAST,
                        a.collected_at DESC NULLS LAST,
                        a.id DESC
                ) FILTER (WHERE a.id IS NOT NULL),
                ARRAY[]::bigint[]
            ) AS article_ids
        FROM raw_articles r
        LEFT JOIN raw_articles a
            ON a.cluster_id = r.cluster_id
           AND a.source_type = 'news'
        WHERE r.source_type = 'news'
          AND r.is_representative = true
          AND r.cluster_id IS NOT NULL
          AND r.processing_status IN ('PROCESSED', 'CLASSIFIED')
          AND (
              (
                  :published_since IS NOT NULL
                  AND r.published_at >= CAST(:published_since AS timestamptz)
                  AND (
                      :published_until IS NULL
                      OR r.published_at < CAST(:published_until AS timestamptz)
                  )
              )
              OR (
                  :published_since IS NULL
                  AND r.collected_at >= NOW() - (:lookback_hours * INTERVAL '1 hour')
              )
          )
          AND (NOT :company_filter OR r.company ?| :company_keys)
          AND (
              cardinality(CAST(:exclude_article_ids AS bigint[])) = 0
              OR r.id <> ALL(CAST(:exclude_article_ids AS bigint[]))
          )
        GROUP BY
            r.cluster_id, r.id, r.company, r.title, r.content, r.url,
            r.source_type, r.content_type, r.publisher, r.language,
            r.relevance_score, r.relevance_label, r.relevance_reason,
            r.matched_companies, r.matched_sectors,
            r.source_name, r.published_at, r.collected_at, r.metadata
        ORDER BY
            r.published_at DESC NULLS LAST,
            r.collected_at DESC,
            r.id DESC
        LIMIT :limit
    """)
    params = {
        "company_filter": company_filter,
        "company_keys": company_keys or [""],
        "exclude_article_ids": exclude_ids,
        "lookback_hours": lookback_hours,
        "published_since": published_since,
        "published_until": published_until,
        "limit": limit,
    }
    with SessionLocal() as db:
        try:
            rows = db.execute(query, params).fetchall()
        except Exception as exc:
            if not _is_missing_column(exc, "matched_sector_details"):
                raise
            db.rollback()
            rows = db.execute(fallback_query, params).fetchall()
    return [dict(row._mapping) for row in rows]


def update_preprocess_status(
    article_id: int,
    processing_status: str,
    metadata_patch: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """전처리 라우팅 결과를 raw_articles에 반영한다."""
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET processing_status = :processing_status,
                    error_message = COALESCE(:error_message, error_message)
                WHERE id = :id
            """),
            {
                "processing_status": processing_status,
                "error_message": error_message,
                "id": article_id,
            },
        )
        if metadata_patch:
            source_type = db.execute(
                text("SELECT source_type FROM raw_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one_or_none()
            if source_type:
                _upsert_source_metadata(
                    db,
                    article_id=article_id,
                    source_type=source_type,
                    source_metadata=json.dumps(
                        _sanitize_jsonish(metadata_patch),
                        ensure_ascii=False,
                    ),
                )
        db.commit()


def update_cluster(
    article_id: int,
    cluster_id: int,
    is_representative: bool,
) -> None:
    """cluster_id, is_representative를 업데이트한다.

    CLASSIFIED 상태의 기사를 cluster-only로 다시 묶을 때 분류 상태가
    PROCESSED로 되돌아가지 않도록 기존 CLASSIFIED는 유지한다.
    """
    status = "PROCESSED"
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET cluster_id = :cluster_id,
                    is_representative = :is_rep,
                    processing_status = CASE
                        WHEN processing_status = 'CLASSIFIED' THEN 'CLASSIFIED'
                        ELSE :status
                    END
                WHERE id = :id
            """),
            {
                "cluster_id": cluster_id,
                "is_rep": is_representative,
                "status": status,
                "id": article_id,
            },
        )
        db.commit()


def update_classification(
    article_id: int,
    importance: str,
    importance_score: float,
    qdrant_vector_id: Optional[str] = None,
) -> None:
    """중요도 분류 결과를 raw_articles에 반영한다.

    qdrant_vector_id를 넘기지 않으면 기존 값을 보존한다 (재분류 시 인덱싱 결과 유실 방지).
    """
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET importance_level = :importance,
                    importance_score = :score,
                    processing_status = 'PROCESSED',
                    qdrant_vector_id = COALESCE(CAST(:qdrant_id AS uuid), qdrant_vector_id)
                WHERE id = :id
            """),
            {
                "importance": importance,
                "score": importance_score,
                "qdrant_id": qdrant_vector_id,
                "id": article_id,
            },
        )
        db.commit()
