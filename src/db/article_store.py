"""raw_articles / card_news 테이블 저장·조회·업데이트 레이어."""

import json
import logging
import re
import threading
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import text

from src.config.company_tiers import SELF_COMPANY_IDS, resolve_company_id
from src.crawler.base import CrawlRunContext, RawArticle
from src.db.postgres import SessionLocal

log = logging.getLogger(__name__)

INDUSTRY_TREND_COMPANY = "industry_trend"
_INDUSTRY_SOURCE_TYPES = {"trend_report", "search_trend"}
_INDUSTRY_MARKER_KEYS = {"sector", "industry", "upjong_code"}
_INDUSTRY_REPORT_TYPES = {"industry", "industry_report", "sector_report"}

_INSERT_SQL = text("""
    INSERT INTO raw_articles (
        source_type, source_name, publisher, title, content, url, url_hash,
        published_at, collected_at, company, language, content_type,
        crawl_status, error_message, processing_status, metadata, crawl_run_id
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW', CAST(:metadata AS jsonb),
        CAST(:crawl_run_id AS uuid)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")

_INSERT_SQL_WITHOUT_CRAWL_RUN_ID = text("""
    INSERT INTO raw_articles (
        source_type, source_name, publisher, title, content, url, url_hash,
        published_at, collected_at, company, language, content_type,
        crawl_status, error_message, processing_status, metadata
    ) VALUES (
        :source_type, :source_name, :publisher, :title, :content, :url, :url_hash,
        :published_at, :collected_at, CAST(:company AS jsonb), :language, :content_type,
        :crawl_status, :error_message, 'RAW', CAST(:metadata AS jsonb)
    )
    ON CONFLICT (url) DO NOTHING
    RETURNING id
""")

_SELECT_ARTICLE_ID_BY_URL = text("SELECT id FROM raw_articles WHERE url = :url")

_UPDATE_DART_CONTENT_IF_BETTER_SQL = text("""
    UPDATE raw_articles
    SET content = :content,
        content_type = COALESCE(:content_type, content_type),
        error_message = COALESCE(:error_message, error_message)
    WHERE id = :id
      AND source_type = 'dart'
      AND length(COALESCE(content, '')) < :new_content_length
      AND :new_content_length >= 1000
""")

_UPDATE_IR_CONTENT_IF_BETTER_SQL = text("""
    UPDATE raw_articles
    SET content = :content,
        content_type = COALESCE(:content_type, content_type),
        error_message = COALESCE(:error_message, error_message),
        collected_at = COALESCE(:collected_at, collected_at)
    WHERE id = :id
      AND source_type = 'ir'
      AND length(COALESCE(content, '')) < :new_content_length
      AND :new_content_length >= 1000
""")

_UPDATE_COMPANY_ANALYSIS_IF_CHANGED_SQL = text("""
    UPDATE raw_articles
    SET title = :title,
        content = :content,
        content_type = COALESCE(:content_type, content_type),
        published_at = COALESCE(:published_at, published_at),
        collected_at = COALESCE(:collected_at, collected_at),
        processing_status = 'RAW',
        metadata = metadata || CAST(:metadata AS jsonb),
        error_message = COALESCE(:error_message, error_message)
    WHERE (id = :id OR url = :url)
      AND source_type = 'company_analysis'
""")

_INSERT_CRAWL_RUN_ARTICLE = text("""
    INSERT INTO crawl_run_articles (
        crawl_run_id, raw_article_id, url, url_hash, discovered_at,
        action, fetch_status, error_message, raw_payload
    ) VALUES (
        CAST(:crawl_run_id AS uuid), :raw_article_id, :url, :url_hash, :discovered_at,
        :action, :fetch_status, :error_message, CAST(:raw_payload AS jsonb)
    )
    ON CONFLICT (crawl_run_id, url_hash) DO UPDATE SET
        raw_article_id = COALESCE(EXCLUDED.raw_article_id, crawl_run_articles.raw_article_id),
        action = EXCLUDED.action,
        fetch_status = EXCLUDED.fetch_status,
        error_message = EXCLUDED.error_message,
        raw_payload = crawl_run_articles.raw_payload || EXCLUDED.raw_payload
""")

_SOURCE_METADATA_EXCLUDED_KEYS = {
    "url_hash",
    "source_type",
    "content_type",
    "company_tier",
    "collection_mode",
    "crawl_run_id",
    "crawl_source_name",
    "track",
    "window_start",
    "window_end",
    "matched_companies",
    "matched_sectors",
    "matched_sector_details",
}

_UPSERT_FINANCIAL_METRIC_SQL = text("""
    INSERT INTO raw_article_financial_metrics (
        raw_article_id, metric_uid, source_type, source_name, peer_id,
        period, period_year, period_quarter, period_type,
        metric_name, metric_label, metric_scope, business_area,
        value_numeric, value_krwbn, value_krw, unit, currency,
        source_page, source_table_uid, source_chunk_uid,
        confidence, extraction_method, evidence_text, payload
    ) VALUES (
        :raw_article_id, :metric_uid, :source_type, :source_name, :peer_id,
        :period, :period_year, :period_quarter, :period_type,
        :metric_name, :metric_label, :metric_scope, :business_area,
        :value_numeric, :value_krwbn, :value_krw, :unit, :currency,
        :source_page, :source_table_uid, :source_chunk_uid,
        :confidence, :extraction_method, :evidence_text, CAST(:payload AS jsonb)
    )
    ON CONFLICT (raw_article_id, metric_uid) DO UPDATE SET
        source_type = EXCLUDED.source_type,
        source_name = EXCLUDED.source_name,
        peer_id = EXCLUDED.peer_id,
        period = EXCLUDED.period,
        period_year = EXCLUDED.period_year,
        period_quarter = EXCLUDED.period_quarter,
        period_type = EXCLUDED.period_type,
        metric_name = EXCLUDED.metric_name,
        metric_label = EXCLUDED.metric_label,
        metric_scope = EXCLUDED.metric_scope,
        business_area = EXCLUDED.business_area,
        value_numeric = EXCLUDED.value_numeric,
        value_krwbn = EXCLUDED.value_krwbn,
        value_krw = EXCLUDED.value_krw,
        unit = EXCLUDED.unit,
        currency = EXCLUDED.currency,
        source_page = EXCLUDED.source_page,
        source_table_uid = EXCLUDED.source_table_uid,
        source_chunk_uid = EXCLUDED.source_chunk_uid,
        confidence = EXCLUDED.confidence,
        extraction_method = EXCLUDED.extraction_method,
        evidence_text = EXCLUDED.evidence_text,
        payload = EXCLUDED.payload
""")

_UPSERT_BUSINESS_SIGNAL_SQL = text("""
    INSERT INTO raw_article_business_signals (
        raw_article_id, signal_uid, source_type, source_name, peer_id,
        period, period_year, period_quarter, period_type,
        business_area, signal_type, sentiment, summary, evidence_text,
        source_page, source_chunk_uid, confidence, extraction_method, payload
    ) VALUES (
        :raw_article_id, :signal_uid, :source_type, :source_name, :peer_id,
        :period, :period_year, :period_quarter, :period_type,
        :business_area, :signal_type, :sentiment, :summary, :evidence_text,
        :source_page, :source_chunk_uid, :confidence, :extraction_method,
        CAST(:payload AS jsonb)
    )
    ON CONFLICT (raw_article_id, signal_uid) DO UPDATE SET
        source_type = EXCLUDED.source_type,
        source_name = EXCLUDED.source_name,
        peer_id = EXCLUDED.peer_id,
        period = EXCLUDED.period,
        period_year = EXCLUDED.period_year,
        period_quarter = EXCLUDED.period_quarter,
        period_type = EXCLUDED.period_type,
        business_area = EXCLUDED.business_area,
        signal_type = EXCLUDED.signal_type,
        sentiment = EXCLUDED.sentiment,
        summary = EXCLUDED.summary,
        evidence_text = EXCLUDED.evidence_text,
        source_page = EXCLUDED.source_page,
        source_chunk_uid = EXCLUDED.source_chunk_uid,
        confidence = EXCLUDED.confidence,
        extraction_method = EXCLUDED.extraction_method,
        payload = EXCLUDED.payload
""")

_DELETE_FINANCIAL_METRICS_SQL = text("""
    DELETE FROM raw_article_financial_metrics
    WHERE raw_article_id = ANY(:raw_article_ids)
      AND (:source_type IS NULL OR source_type = :source_type)
    RETURNING 1
""")

_DELETE_BUSINESS_SIGNALS_SQL = text("""
    DELETE FROM raw_article_business_signals
    WHERE raw_article_id = ANY(:raw_article_ids)
      AND (:source_type IS NULL OR source_type = :source_type)
    RETURNING 1
""")

_ENSURE_MARKET_PRICE_OHLCV_SQL = text("""
    CREATE TABLE IF NOT EXISTS market_price_ohlcv (
        id BIGSERIAL PRIMARY KEY,
        raw_article_id BIGINT REFERENCES raw_articles(id) ON DELETE SET NULL,
        peer_id TEXT,
        ticker TEXT NOT NULL,
        trade_date DATE NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume BIGINT,
        change_pct NUMERIC,
        currency TEXT DEFAULT 'KRW',
        source_type TEXT NOT NULL DEFAULT 'market_data',
        source_name TEXT,
        publisher TEXT,
        collected_at TIMESTAMPTZ,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (ticker, trade_date, source_name)
    )
""")

_ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL = text("""
    CREATE INDEX IF NOT EXISTS idx_market_price_ohlcv_peer_date
    ON market_price_ohlcv (peer_id, trade_date DESC)
""")

_UPSERT_MARKET_PRICE_OHLCV_SQL = text("""
    INSERT INTO market_price_ohlcv (
        raw_article_id, peer_id, ticker, trade_date,
        open, high, low, close, volume, change_pct,
        currency, source_type, source_name, publisher, collected_at, payload
    ) VALUES (
        :raw_article_id, :peer_id, :ticker, :trade_date,
        :open, :high, :low, :close, :volume, :change_pct,
        :currency, :source_type, :source_name, :publisher, :collected_at,
        CAST(:payload AS jsonb)
    )
    ON CONFLICT (ticker, trade_date, source_name) DO UPDATE SET
        raw_article_id = EXCLUDED.raw_article_id,
        peer_id = EXCLUDED.peer_id,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        change_pct = EXCLUDED.change_pct,
        currency = EXCLUDED.currency,
        source_type = EXCLUDED.source_type,
        publisher = EXCLUDED.publisher,
        collected_at = EXCLUDED.collected_at,
        payload = EXCLUDED.payload,
        updated_at = NOW()
""")


def save_articles(
    articles: list[RawArticle],
    run_context: CrawlRunContext | None = None,
) -> int:
    """RawArticle 목록을 raw_articles 테이블에 저장. 중복 URL은 스킵.

    Returns:
        실제 삽입된 건수
    """
    if not articles:
        return 0

    inserted = 0
    with SessionLocal() as db:
        for article in articles:
            storage_company = _company_for_storage(article)
            if not _is_valid(article, storage_company):
                continue
            try:
                sanitized_title = _sanitize_text(article.title)[:500]
                sanitized_content = _sanitize_text(article.content if article.content else "")
                source_metadata = _source_metadata_json(article, storage_company)
                insert_params = {
                    "source_type": article.source_type,
                    "source_name": article.source_name,
                    "publisher": article.publisher,
                    "title": sanitized_title,
                    "content": sanitized_content,
                    "url": article.url,
                    "url_hash": article.url_hash,
                    "published_at": article.published_at or article.collected_at,
                    "collected_at": article.collected_at,
                    "company": json.dumps(storage_company, ensure_ascii=False),
                    "language": article.language,
                    "content_type": article.content_type,
                    "crawl_status": article.crawl_status,
                    "error_message": article.error_message,
                    "metadata": source_metadata,
                    "crawl_run_id": run_context.crawl_run_id if run_context else None,
                }
                try:
                    result = db.execute(_INSERT_SQL, insert_params)
                except Exception as exc:  # noqa: BLE001
                    if not _is_missing_column(exc, "crawl_run_id"):
                        raise
                    db.rollback()
                    log.info(
                        "raw_articles.crawl_run_id 미적용 DB 감지 → legacy INSERT 사용 | url=%s",
                        article.url,
                    )
                    result = db.execute(_INSERT_SQL_WITHOUT_CRAWL_RUN_ID, insert_params)
                row = result.fetchone()
                if row:
                    article_id = row[0]
                    inserted += 1
                    action = "inserted"
                else:
                    existing = db.execute(
                        _SELECT_ARTICLE_ID_BY_URL,
                        {"url": article.url},
                    ).fetchone()
                    article_id = existing[0] if existing else None
                    action = "duplicate"
                if article_id:
                    _update_dart_content_if_better(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_content=sanitized_content,
                    )
                    _update_ir_content_if_better(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_content=sanitized_content,
                    )
                    _update_company_analysis_if_changed(
                        db,
                        article_id=article_id,
                        article=article,
                        sanitized_title=sanitized_title,
                        sanitized_content=sanitized_content,
                        source_metadata=source_metadata,
                    )
                    _upsert_source_metadata(
                        db,
                        article_id=article_id,
                        source_type=article.source_type,
                        source_metadata=source_metadata,
                    )
                    _upsert_market_price_ohlcv_for_article(db, article_id, article)
                    _upsert_crawl_run_article(
                        db,
                        article_id=article_id,
                        article=article,
                        run_context=run_context,
                        action=action,
                        source_metadata=source_metadata,
                    )
            except Exception as e:
                log.error("raw_articles 저장 실패 | url=%s error=%s", article.url, e)
                db.rollback()
        db.commit()

    log.info(
        "raw_articles 저장 완료 | total=%d inserted=%d skipped=%d",
        len(articles),
        inserted,
        len(articles) - inserted,
    )
    return inserted


def _update_dart_content_if_better(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_content: str,
) -> None:
    if article.source_type != "dart":
        return
    db.execute(
        _UPDATE_DART_CONTENT_IF_BETTER_SQL,
        {
            "id": article_id,
            "content": sanitized_content,
            "content_type": article.content_type,
            "error_message": article.error_message,
            "new_content_length": len(sanitized_content),
        },
    )


def _update_ir_content_if_better(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_content: str,
) -> None:
    if article.source_type != "ir":
        return
    db.execute(
        _UPDATE_IR_CONTENT_IF_BETTER_SQL,
        {
            "id": article_id,
            "content": sanitized_content,
            "content_type": article.content_type,
            "error_message": article.error_message,
            "collected_at": article.collected_at,
            "new_content_length": len(sanitized_content),
        },
    )


def _update_company_analysis_if_changed(
    db,
    *,
    article_id: int,
    article: RawArticle,
    sanitized_title: str,
    sanitized_content: str,
    source_metadata: str,
) -> None:
    if article.source_type != "company_analysis":
        return
    result = db.execute(
        _UPDATE_COMPANY_ANALYSIS_IF_CHANGED_SQL,
        {
            "id": article_id,
            "url": article.url,
            "title": sanitized_title,
            "content": sanitized_content,
            "content_type": article.content_type,
            "published_at": article.published_at or article.collected_at,
            "collected_at": article.collected_at,
            "metadata": source_metadata,
            "error_message": article.error_message,
        },
    )
    log.info(
        "company_analysis 최신 본문 갱신 | url=%s rows=%d",
        article.url,
        result.rowcount,
    )


def article_exists_by_url(url: str) -> bool:
    """Return whether a raw article URL has already been stored."""
    if not url:
        return False

    with SessionLocal() as db:
        row = db.execute(_SELECT_ARTICLE_ID_BY_URL, {"url": url}).fetchone()
    return row is not None


# ──────────────────────────────────────────────────────────────
# 조회
# ──────────────────────────────────────────────────────────────


def get_articles_by_ids(ids: list[int]) -> list[dict[str, Any]]:
    """raw_articles 테이블에서 ID 목록으로 기사를 조회한다."""
    if not ids:
        return []
    query = text("""
        SELECT raw_articles.id, raw_articles.company, raw_articles.title,
               raw_articles.content, raw_articles.url,
               raw_articles.source_type, raw_articles.content_type,
               raw_articles.publisher, raw_articles.language,
               raw_articles.cluster_id, raw_articles.is_representative,
               raw_articles.processing_status,
               raw_articles.importance_score, raw_articles.importance_level,
               raw_articles.relevance_score, raw_articles.relevance_label,
               raw_articles.relevance_reason,
               raw_articles.matched_companies, raw_articles.matched_sectors,
               raw_articles.matched_sector_details,
               raw_articles.source_name, raw_articles.published_at,
               raw_articles.collected_at,
               raw_articles.metadata,
               COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
               COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
               COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
        FROM raw_articles
        LEFT JOIN raw_article_parse_results pr
            ON pr.raw_article_id = raw_articles.id
        WHERE raw_articles.id = ANY(:ids)
        ORDER BY raw_articles.published_at DESC NULLS LAST,
                 raw_articles.collected_at DESC NULLS LAST,
                 raw_articles.id DESC
    """)
    fallback_query = text("""
        SELECT raw_articles.id, raw_articles.company, raw_articles.title,
               raw_articles.content, raw_articles.url,
               raw_articles.source_type, raw_articles.content_type,
               raw_articles.publisher, raw_articles.language,
               raw_articles.cluster_id, raw_articles.is_representative,
               raw_articles.processing_status,
               raw_articles.importance_score, raw_articles.importance_level,
               raw_articles.relevance_score, raw_articles.relevance_label,
               raw_articles.relevance_reason,
               raw_articles.matched_companies, raw_articles.matched_sectors,
               '{}'::jsonb AS matched_sector_details,
               raw_articles.source_name, raw_articles.published_at,
               raw_articles.collected_at,
               raw_articles.metadata,
               COALESCE(pr.raw_result, '{}'::jsonb) AS parser_result,
               COALESCE(pr.financial_record, '{}'::jsonb) AS financial_record,
               COALESCE(pr.warnings, '[]'::jsonb) AS parser_warnings
        FROM raw_articles
        LEFT JOIN raw_article_parse_results pr
            ON pr.raw_article_id = raw_articles.id
        WHERE raw_articles.id = ANY(:ids)
        ORDER BY raw_articles.published_at DESC NULLS LAST,
                 raw_articles.collected_at DESC NULLS LAST,
                 raw_articles.id DESC
    """)
    with SessionLocal() as db:
        try:
            rows = db.execute(query, {"ids": ids}).fetchall()
        except Exception as exc:
            if not _is_missing_column(exc, "matched_sector_details"):
                raise
            db.rollback()
            rows = db.execute(fallback_query, {"ids": ids}).fetchall()
    return [dict(row._mapping) for row in rows]


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


def _fetch_dart_rows(
    *,
    select_sql: str,
    where_sql: str,
    order_limit_sql: str,
    params: dict[str, Any],
) -> list[Any]:
    query = text(f"""
        {select_sql}
        WHERE r.source_type = 'dart'
          AND r.crawl_status = 'success'
          {where_sql}
        {order_limit_sql}
    """)
    with SessionLocal() as db:
        return list(db.execute(query, params).fetchall())


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


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_or_value(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else default


def _merge_source_dicts(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in [*existing, *incoming]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("raw_article_id") or item.get("id") or item.get("url") or "")
        if not key:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        clean_item = dict(item)
        clean_item["index"] = len(merged) + 1
        merged.append(clean_item)

    return merged


def _source_dict_from_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": 0,
        "raw_article_id": int(article["id"]),
        "title": article.get("title") or "",
        "source_name": article.get("source_name") or article.get("publisher") or "",
        "url": article.get("url") or "",
        "published_at": _iso_or_none(article.get("published_at")),
        "collected_at": _iso_or_none(article.get("collected_at")),
    }


def _source_article_dict_from_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(article["id"]),
        "title": article.get("title") or "",
        "url": article.get("url") or "",
        "source_name": article.get("source_name") or "",
        "publisher": article.get("publisher") or "",
        "published_at": _iso_or_none(article.get("published_at")),
        "collected_at": _iso_or_none(article.get("collected_at")),
    }


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


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


def _ticker_from_market_data_url(url: str) -> str | None:
    match = re.search(r"(?:code=|/item/)(\d{6})(?!\d)", url or "")
    return match.group(1) if match else None


def _parse_market_trade_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _market_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _market_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
        return None


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


# ──────────────────────────────────────────────────────────────
# 업데이트 — 파이프라인 각 단계에서 호출
# ──────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────
# 카드 뉴스 저장 (구 issue_cards → V9 에서 card_news 로 rename)
# ──────────────────────────────────────────────────────────────

# V33 이후 — 모든 v2 컬럼 (`peer_company_id` / `primary_keyword_category` /
# `source_raw_article_ids` / `evidence_payload` / `card_schema_version` /
# `evaluation_payload`) 을 INSERT 시점에 한 번에 채운다.
_INSERT_CARD_NEWS_V2 = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score,
        peer_company_id, primary_keyword_category, integrated_issue_id, source_raw_article_ids,
        keyword_categories, evidence_payload, source_articles,
        card_schema_version, evaluation_payload
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score,
        :peer_company_id, :primary_keyword_category, CAST(:integrated_issue_id AS uuid),
        CAST(:source_raw_article_ids AS bigint[]),
        CAST(:keyword_categories AS jsonb),
        CAST(:evidence_payload AS jsonb),
        CAST(:source_articles AS jsonb),
        :card_schema_version,
        CAST(:evaluation_payload AS jsonb)
    )
    ON CONFLICT (id) DO UPDATE SET
        status = 'ACTIVE',
        title = EXCLUDED.title,
        summary_lines = EXCLUDED.summary_lines,
        event_type = EXCLUDED.event_type,
        importance = EXCLUDED.importance,
        importance_score = EXCLUDED.importance_score,
        implication = CAST(:implication AS jsonb),
        sources = CAST(:sources AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score,
        peer_company_id = COALESCE(EXCLUDED.peer_company_id, card_news.peer_company_id),
        primary_keyword_category = COALESCE(
            EXCLUDED.primary_keyword_category, card_news.primary_keyword_category
        ),
        integrated_issue_id = COALESCE(EXCLUDED.integrated_issue_id, card_news.integrated_issue_id),
        source_raw_article_ids = COALESCE(
            EXCLUDED.source_raw_article_ids, card_news.source_raw_article_ids
        ),
        keyword_categories = COALESCE(EXCLUDED.keyword_categories, card_news.keyword_categories),
        evidence_payload = COALESCE(EXCLUDED.evidence_payload, card_news.evidence_payload),
        source_articles = COALESCE(EXCLUDED.source_articles, card_news.source_articles),
        card_schema_version = EXCLUDED.card_schema_version,
        evaluation_payload =
            COALESCE(card_news.evaluation_payload, '{}'::jsonb)
            || COALESCE(EXCLUDED.evaluation_payload, '{}'::jsonb)
    RETURNING id
""")

_INSERT_CARD_NEWS_V2_WITHOUT_INTEGRATED_ISSUE = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score,
        peer_company_id, primary_keyword_category, source_raw_article_ids,
        keyword_categories, evidence_payload, source_articles,
        card_schema_version, evaluation_payload
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score,
        :peer_company_id, :primary_keyword_category,
        CAST(:source_raw_article_ids AS bigint[]),
        CAST(:keyword_categories AS jsonb),
        CAST(:evidence_payload AS jsonb),
        CAST(:source_articles AS jsonb),
        :card_schema_version,
        CAST(:evaluation_payload AS jsonb)
    )
    ON CONFLICT (id) DO UPDATE SET
        status = 'ACTIVE',
        title = EXCLUDED.title,
        summary_lines = EXCLUDED.summary_lines,
        event_type = EXCLUDED.event_type,
        importance = EXCLUDED.importance,
        importance_score = EXCLUDED.importance_score,
        implication = CAST(:implication AS jsonb),
        sources = CAST(:sources AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score,
        peer_company_id = COALESCE(EXCLUDED.peer_company_id, card_news.peer_company_id),
        primary_keyword_category = COALESCE(
            EXCLUDED.primary_keyword_category, card_news.primary_keyword_category
        ),
        source_raw_article_ids = COALESCE(
            EXCLUDED.source_raw_article_ids, card_news.source_raw_article_ids
        ),
        keyword_categories = COALESCE(EXCLUDED.keyword_categories, card_news.keyword_categories),
        evidence_payload = COALESCE(EXCLUDED.evidence_payload, card_news.evidence_payload),
        source_articles = COALESCE(EXCLUDED.source_articles, card_news.source_articles),
        card_schema_version = EXCLUDED.card_schema_version,
        evaluation_payload =
            COALESCE(card_news.evaluation_payload, '{}'::jsonb)
            || COALESCE(EXCLUDED.evaluation_payload, '{}'::jsonb)
    RETURNING id
""")

_INSERT_CARD_NEWS_ARTICLE = text("""
    INSERT INTO card_news_articles (
        card_news_id,
        raw_article_id,
        article_order,
        relation_source
    ) VALUES (
        :card_news_id,
        :raw_article_id,
        :article_order,
        :relation_source
    )
    ON CONFLICT (card_news_id, raw_article_id) DO UPDATE SET
        article_order = EXCLUDED.article_order,
        relation_source = EXCLUDED.relation_source
""")

# pre-V33 환경 fallback (v2 컬럼 미존재 시). 신규 컬럼 6개 빼고 INSERT.
_INSERT_CARD_NEWS_V1_FALLBACK = text("""
    INSERT INTO card_news (
        id, company, cluster_id, title, summary_lines,
        event_type, importance, importance_score,
        implication, sources, validation_pass, validation_sc_score
    ) VALUES (
        :id, :company, :cluster_id, :title, :summary_lines,
        :event_type, :importance, :importance_score,
        CAST(:implication AS jsonb), CAST(:sources AS jsonb),
        :validation_pass, :validation_sc_score
    )
    ON CONFLICT (id) DO UPDATE SET
        implication = CAST(:implication AS jsonb),
        validation_pass = :validation_pass,
        validation_sc_score = :validation_sc_score
    RETURNING id
""")


def save_card_news(card: dict[str, Any]) -> Optional[str]:
    """카드 뉴스를 card_news 테이블에 저장한다.

    외부 리뷰 R-2 반영 (2026-05-21) — 모든 v2 컬럼을 단일 INSERT 로 채운다.
    더 이상 두 번째 UPDATE 가 필요하지 않으며, FK / sector / provenance / schema /
    evaluation 모두 카드 INSERT 시점에 일관 보장된다.

    호환 fallback — V33 미적용 환경 (`card_schema_version` 컬럼 없음) 에서는
    `UndefinedColumn` 발생 → 기존 v1 컬럼만 사용하는 fallback INSERT 로 자동 재시도.
    """
    try:
        if _is_self_company_card(card):
            log.info(
                "카드 뉴스 저장 제외 | id=%s company=%s reason=self company",
                card.get("id"),
                card.get("company") or card.get("peer_id") or card.get("peer_company_id"),
            )
            return None
        params = _card_news_insert_params(card)
        try:
            card_id = _execute_v2_insert(card_id=card["id"], params=params)
        except Exception as exc:  # noqa: BLE001
            if not _is_undefined_column_error(exc):
                raise
            log.info(
                (
                    "card_news v2 INSERT 실패 (V40 integrated_issue_id 미적용 가능) "
                    "→ legacy v2 재시도 | id=%s"
                ),
                card.get("id"),
            )
            try:
                card_id = _execute_v2_without_integrated_issue(card_id=card["id"], params=params)
            except Exception as legacy_exc:  # noqa: BLE001
                if not _is_undefined_column_error(legacy_exc):
                    raise
                log.info(
                    "card_news legacy v2 INSERT 실패 (v33 미적용) → v1 fallback 사용 | id=%s",
                    card.get("id"),
                )
                card_id = _execute_v1_fallback(card_id=card["id"], params=params)
        if card_id:
            _sync_card_news_articles(
                card_id=card_id,
                source_raw_article_ids=params["source_raw_article_ids"],
                relation_source="source_raw_article_ids",
            )
            sync_card_sources_for_cluster(card.get("cluster_id"))
        return card_id
    except Exception as e:
        log.error("카드 뉴스 저장 실패 | id=%s error=%s", card.get("id"), e)
    return None


def _is_self_company_card(card: dict[str, Any]) -> bool:
    for key in ("peer_company_id", "company", "peer_id"):
        company_id = resolve_company_id(str(card.get(key) or ""))
        if company_id in SELF_COMPANY_IDS:
            return True
    return False


def _execute_v2_insert(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V2, params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v2) | id=%s", card_id)
            return card_id
    return None


def _execute_v2_without_integrated_issue(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    legacy_params = dict(params)
    legacy_params.pop("integrated_issue_id", None)
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V2_WITHOUT_INTEGRATED_ISSUE, legacy_params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v2 legacy no integrated_issue_id) | id=%s", card_id)
            return card_id
    return None


def _execute_v1_fallback(*, card_id: str, params: dict[str, Any]) -> Optional[str]:
    legacy_params = {
        key: params[key]
        for key in (
            "id",
            "company",
            "cluster_id",
            "title",
            "summary_lines",
            "event_type",
            "importance",
            "importance_score",
            "implication",
            "sources",
            "validation_pass",
            "validation_sc_score",
        )
    }
    with SessionLocal() as db:
        result = db.execute(_INSERT_CARD_NEWS_V1_FALLBACK, legacy_params)
        row = result.fetchone()
        db.commit()
        if row:
            log.info("카드 뉴스 저장 완료 (v1 fallback) | id=%s", card_id)
            return card_id
    return None


def _is_undefined_column_error(exc: Exception) -> bool:
    """psycopg / SQLAlchemy 가 던지는 UndefinedColumn 인지 확인."""
    text_repr = str(exc).lower()
    return "undefinedcolumn" in text_repr or "does not exist" in text_repr


def _card_news_insert_params(card: dict[str, Any]) -> dict[str, Any]:
    implication_payload = _merge_implication_payload(card)
    source_ids = _normalize_int_list(card.get("source_raw_article_ids"))
    if not source_ids:
        source_ids = _source_ids_from_sources(card.get("sources"))
    evidence_payload = _build_evidence_payload(card)
    evaluation_payload = card.get("evaluation_payload") or {}
    if not isinstance(evaluation_payload, dict):
        evaluation_payload = {}
    keyword_categories = card.get("keyword_categories") or {}
    if not isinstance(keyword_categories, dict):
        keyword_categories = {}
    source_articles = _source_articles_payload(card, source_ids)
    return {
        "id": card["id"],
        "company": card.get("company") or card.get("peer_id") or _resolve_peer_company_id(card),
        "cluster_id": card.get("cluster_id"),
        "title": card["title"][:500],
        "summary_lines": card.get("summary_lines", []),
        "event_type": card.get("event_type", "tech"),
        "importance": card.get("importance", "low"),
        "importance_score": card.get("importance_score", 0.0),
        "implication": json.dumps(implication_payload, ensure_ascii=False, default=str),
        "sources": json.dumps(card.get("sources", []), ensure_ascii=False, default=str),
        "validation_pass": card.get("validation", {}).get("pass", False),
        "validation_sc_score": card.get("validation", {}).get("sc_score", 0.0),
        "peer_company_id": _resolve_peer_company_id(card),
        "primary_keyword_category": card.get("primary_keyword_category") or card.get("sector"),
        "integrated_issue_id": _resolve_integrated_issue_id(card, evidence_payload),
        "source_raw_article_ids": source_ids,
        "keyword_categories": json.dumps(keyword_categories, ensure_ascii=False, default=str),
        "evidence_payload": json.dumps(evidence_payload, ensure_ascii=False, default=str),
        "source_articles": json.dumps(source_articles, ensure_ascii=False, default=str),
        "card_schema_version": str(card.get("card_schema_version") or "v2"),
        "evaluation_payload": json.dumps(evaluation_payload, ensure_ascii=False, default=str),
    }


def _resolve_integrated_issue_id(
    card: dict[str, Any], evidence_payload: dict[str, Any]
) -> Optional[str]:
    for value in (
        card.get("integrated_issue_id"),
        evidence_payload.get("integrated_issue_id"),
        (evidence_payload.get("analysis_package") or {}).get("integrated_issue_id")
        if isinstance(evidence_payload.get("analysis_package"), dict)
        else None,
    ):
        text_value = str(value or "").strip()
        if text_value:
            return text_value
    return None


def _sync_card_news_articles(
    *,
    card_id: str,
    source_raw_article_ids: list[int],
    relation_source: str,
) -> None:
    """Keep normalized card -> raw article links in step with card_news."""
    ids = _normalize_int_list(source_raw_article_ids)
    if not ids:
        return
    try:
        with SessionLocal() as db:
            db.execute(
                text("""
                    DELETE FROM card_news_articles
                    WHERE card_news_id = :card_news_id
                      AND NOT (raw_article_id = ANY(CAST(:source_raw_article_ids AS bigint[])))
                """),
                {
                    "card_news_id": card_id,
                    "source_raw_article_ids": ids,
                },
            )
            db.execute(
                _INSERT_CARD_NEWS_ARTICLE,
                [
                    {
                        "card_news_id": card_id,
                        "raw_article_id": raw_id,
                        "article_order": index,
                        "relation_source": relation_source,
                    }
                    for index, raw_id in enumerate(ids, start=1)
                ],
            )
            db.commit()
    except Exception as e:  # noqa: BLE001
        if _is_missing_card_news_articles_table(e):
            log.info(
                "card_news_articles table unavailable; skip normalized card/article sync | "
                "card_id=%s article_ids=%s",
                card_id,
                ids,
            )
            return
        log.warning(
            "card_news_articles 동기화 실패 | card_id=%s article_ids=%s error=%s",
            card_id,
            ids,
            e,
        )


def _is_missing_card_news_articles_table(exc: Exception) -> bool:
    text_repr = str(exc).lower()
    return "card_news_articles" in text_repr and (
        "undefinedtable" in text_repr or "does not exist" in text_repr
    )


def sync_card_sources_for_cluster(cluster_id: Any) -> int:
    """Sync ACTIVE card provenance from all relevant processed raw articles in a cluster."""
    try:
        normalized_cluster_id = int(cluster_id)
    except (TypeError, ValueError):
        return 0
    try:
        with SessionLocal() as db:
            result = db.execute(
                text(
                    """
                    WITH ranked_articles AS (
                        SELECT
                            ra.cluster_id,
                            ra.id,
                            ra.title,
                            ra.url,
                            ra.source_name,
                            ra.publisher,
                            ra.published_at,
                            ra.collected_at,
                            ROW_NUMBER() OVER (
                                PARTITION BY ra.cluster_id
                                ORDER BY
                                    ra.published_at DESC NULLS LAST,
                                    ra.collected_at DESC NULLS LAST,
                                    ra.id DESC
                            ) AS rn
                        FROM raw_articles ra
                        WHERE ra.cluster_id = :cluster_id
                          AND ra.source_type = 'news'
                          AND ra.processing_status = 'PROCESSED'
                          AND ra.relevance_label = 'relevant'
                    ),
                    cluster_sources AS (
                        SELECT
                            cluster_id,
                            ARRAY_AGG(
                                id
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
                            ) AS raw_ids,
                            JSONB_AGG(
                                JSONB_BUILD_OBJECT(
                                    'index', rn,
                                    'raw_article_id', id,
                                    'title', COALESCE(title, ''),
                                    'source_name', COALESCE(source_name, publisher, ''),
                                    'url', COALESCE(url, ''),
                                    'published_at', published_at,
                                    'collected_at', collected_at
                                )
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
                            ) AS sources,
                            JSONB_AGG(
                                JSONB_BUILD_OBJECT(
                                    'id', id,
                                    'title', COALESCE(title, ''),
                                    'url', COALESCE(url, ''),
                                    'source_name', COALESCE(source_name, ''),
                                    'publisher', COALESCE(publisher, ''),
                                    'published_at', published_at,
                                    'collected_at', collected_at
                                )
                                ORDER BY
                                    published_at DESC NULLS LAST,
                                    collected_at DESC NULLS LAST,
                                    id DESC
                            ) AS source_articles
                        FROM ranked_articles
                        GROUP BY cluster_id
                    )
                    UPDATE card_news cn
                    SET source_raw_article_ids = cs.raw_ids,
                        sources = cs.sources,
                        source_articles = cs.source_articles
                    FROM cluster_sources cs
                    WHERE cn.status = 'ACTIVE'
                      AND cn.cluster_id = cs.cluster_id
                    """
                ),
                {"cluster_id": normalized_cluster_id},
            )
            db.commit()
            return int(getattr(result, "rowcount", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "card_news cluster source sync 실패 | cluster_id=%s error=%s",
            cluster_id,
            exc,
        )
        return 0


def _source_ids_from_sources(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for source in value:
        if not isinstance(source, dict):
            continue
        ids.extend(
            _normalize_int_list(
                source.get("raw_article_id") or source.get("article_id") or source.get("id")
            )
        )
    return _normalize_int_list(ids)


def _source_articles_payload(card: dict[str, Any], source_ids: list[int]) -> list[dict[str, Any]]:
    source_articles = card.get("source_articles")
    if isinstance(source_articles, list) and source_articles:
        return [item for item in source_articles if isinstance(item, dict)]

    payload: list[dict[str, Any]] = []
    sources = card.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            raw_id = _normalize_int_list(
                source.get("raw_article_id") or source.get("article_id") or source.get("id")
            )
            if not raw_id:
                continue
            item = {
                "id": raw_id[0],
                "title": source.get("title") or "",
                "url": source.get("url") or "",
                "source_name": source.get("source_name") or source.get("publisher") or "",
                "publisher": source.get("publisher") or "",
                "published_at": source.get("published_at"),
                "collected_at": source.get("collected_at"),
            }
            for image_key in (
                "image_url",
                "thumbnail_url",
                "thumbnail",
                "og_image",
                "main_image",
                "image",
                "image_urls",
                "images",
                "media_assets",
                "visual_images",
            ):
                if source.get(image_key):
                    item[image_key] = source[image_key]
            payload.append(item)
    if payload:
        return payload
    return [{"id": raw_id} for raw_id in source_ids]


def _resolve_peer_company_id(card: dict[str, Any]) -> Optional[str]:
    """카드 INSERT 시 peer_company_id FK 를 직접 확정한다.

    우선순위: card['peer_company_id'] (CardNewsComposer 가 set 했을 수 있음) →
    card['company'] (peer_companies.id 와 동일 표기) → card['peer_id'].
    SK AX 같은 self 회사는 FK NULL (peer_companies 가 self 도 포함하지만,
    안전을 위해 None 으로 두고 보조 컬럼 company 만 사용).
    """
    direct = card.get("peer_company_id")
    if direct:
        return str(direct)
    company = card.get("company") or card.get("peer_id")
    if not company:
        return None
    return str(company)


def _normalize_int_list(value: Any) -> list[int]:
    if not value:
        return []
    if isinstance(value, list | tuple | set):
        out: list[int] = []
        seen: set[int] = set()
        for item in value:
            try:
                ivalue = int(item)
            except (TypeError, ValueError):
                continue
            if ivalue <= 0 or ivalue in seen:
                continue
            seen.add(ivalue)
            out.append(ivalue)
        return out
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return []
    return [ivalue] if ivalue > 0 else []


def _build_evidence_payload(card: dict[str, Any]) -> dict[str, Any]:
    """카드 저장 시 `evidence_payload` JSONB 의 단일 출처.

    supervisor 의 메모리 `evidence_payload` 와 CardNewsComposer 의 `evidence_chain` /
    `sources` 를 합쳐서 sidecar (W5-2) 가 단일 컬럼만 보고도 풍부한 근거에 접근 가능.
    """
    payload: dict[str, Any] = {}
    raw_evidence = card.get("evidence_payload")
    if isinstance(raw_evidence, dict):
        payload.update(raw_evidence)
    analysis_package = card.get("analysis_package")
    if isinstance(analysis_package, dict) and analysis_package:
        payload.setdefault("analysis_package", analysis_package)
    evidence_chain = card.get("evidence_chain") or {}
    if isinstance(evidence_chain, dict):
        payload.setdefault("source_links", evidence_chain.get("source_links") or [])
        payload.setdefault("financial_refs", evidence_chain.get("financial_refs") or {})
        payload.setdefault("mbb_refs", evidence_chain.get("mbb_refs") or [])
        payload.setdefault("provenance", evidence_chain.get("provenance") or {})
    sources = card.get("sources") or []
    if isinstance(sources, list) and "source_links" not in payload:
        payload["source_links"] = sources
    return payload


def _merge_implication_payload(card: dict[str, Any]) -> dict[str, Any]:
    """ImplicationAgent v4.0/v5.0 결과 + 보조 메타데이터를 단일 JSONB 로 합성.

    v2 schema 가 우선. legacy v3_payload (sector / exposure / signals) 는 보조 key 로
    함께 보존하여 frontend / sidecar 가 둘 다 읽을 수 있게 한다.
    """
    payload: dict[str, Any] = {}
    for candidate in _implication_payload_candidates(card):
        if _has_structured_implication(candidate):
            payload = dict(candidate)
            break
    frontend = card.get("frontend_implication")
    if isinstance(frontend, dict) and frontend:
        payload["frontend"] = dict(frontend)
        if frontend.get("suggested_actions"):
            payload.setdefault("recommended_actions", frontend.get("suggested_actions"))
            skax = payload.get("skax_implication")
            if isinstance(skax, dict):
                skax.setdefault("recommended_actions", frontend.get("suggested_actions"))
    if not payload.get("frontend"):
        display_frontend = _frontend_implication_from_display_sections(card.get("display_sections"))
        if display_frontend:
            payload["frontend"] = display_frontend
            if display_frontend.get("suggested_actions"):
                payload.setdefault("recommended_actions", display_frontend["suggested_actions"])
    # 보조 메타데이터 (sector / exposure / signals / evidence_chain) 는 별도 namespace.
    payload.setdefault(
        "sector_meta",
        {
            "sector": card.get("sector", "other"),
            "sectors": card.get("sectors", ["other"]),
            "exposure_score": card.get("exposure_score", 0.0),
            "exposure_band": card.get("exposure_band", "low"),
            "signals": card.get("signals", {}),
            "evidence_chain": card.get("evidence_chain", {}),
        },
    )
    return payload


def _implication_payload_candidates(card: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in (card.get("implication"), card.get("implication_result")):
        if isinstance(value, dict):
            candidates.append(value)
    analysis_package = card.get("analysis_package")
    if isinstance(analysis_package, dict):
        for value in (
            analysis_package.get("implication"),
            (analysis_package.get("evidence_payload") or {}).get("implication")
            if isinstance(analysis_package.get("evidence_payload"), dict)
            else None,
        ):
            if isinstance(value, dict):
                candidates.append(value)
    evidence_payload = card.get("evidence_payload")
    if isinstance(evidence_payload, dict):
        package = evidence_payload.get("analysis_package")
        if isinstance(package, dict) and isinstance(package.get("implication"), dict):
            candidates.append(package["implication"])
    return candidates


def _has_structured_implication(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "skax_implication",
            "peer_implication",
            "frontend",
            "recommended_actions",
            "watch_points",
            "follow_up_questions",
        )
    )


def _frontend_implication_from_display_sections(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {}
    sections: dict[str, list[str]] = {}
    for section in value:
        if not isinstance(section, dict):
            continue
        section_type = str(section.get("type") or "").strip()
        items = [
            str(item).strip() for item in (section.get("items") or []) if str(item or "").strip()
        ]
        if section_type and items:
            sections[section_type] = items
    insight_items = sections.get("insight") or []
    action_items = sections.get("action") or []
    if not insight_items and not action_items:
        return {}
    payload: dict[str, Any] = {
        "key_implications": insight_items,
        "peer_implications": insight_items,
        "suggested_actions": action_items,
        "response_directions": action_items,
        "follow_up_questions": [],
    }
    if insight_items:
        payload["potential_impact"] = insight_items[0]
    return payload


def merge_card_news_sources_for_cluster(
    cluster_id: int,
    raw_article_ids: list[int],
    *,
    importance_score: float | None = None,
) -> Optional[str]:
    """기존 클러스터 카드에 새 원문 기사 출처를 병합한다.

    같은 주제의 새 기사가 기존 cluster_id에 편입될 때 새 card_news를 만들지 않고,
    기존 카드의 sources/source_raw_article_ids/source_articles를 보강한다.
    """
    if not raw_article_ids:
        return None

    articles = get_articles_by_ids(raw_article_ids)
    if not articles:
        return None

    try:
        with SessionLocal() as db:
            row = db.execute(
                text("""
                    SELECT id, sources, source_raw_article_ids, source_articles, importance_score
                    FROM card_news
                    WHERE cluster_id = :cluster_id
                      AND status <> 'DELETED'
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"cluster_id": cluster_id},
            ).fetchone()
            if row is None:
                return None

            current = dict(row._mapping)
            existing_sources = _json_or_value(current.get("sources"), [])
            existing_source_articles = _json_or_value(current.get("source_articles"), [])
            existing_ids = [
                int(value)
                for value in (current.get("source_raw_article_ids") or [])
                if value is not None
            ]

            merged_sources = _merge_source_dicts(
                existing_sources if isinstance(existing_sources, list) else [],
                [_source_dict_from_article(article) for article in articles],
            )
            merged_source_articles = _merge_source_dicts(
                existing_source_articles if isinstance(existing_source_articles, list) else [],
                [_source_article_dict_from_article(article) for article in articles],
            )
            merged_ids = sorted({*existing_ids, *(int(article["id"]) for article in articles)})
            merged_importance = max(
                float(current.get("importance_score") or 0.0),
                float(importance_score or 0.0),
            )

            db.execute(
                text("""
                    UPDATE card_news
                    SET sources = CAST(:sources AS jsonb),
                        source_raw_article_ids = CAST(:source_raw_article_ids AS bigint[]),
                        source_articles = CAST(:source_articles AS jsonb),
                        importance_score = GREATEST(
                            COALESCE(importance_score, 0),
                            :importance_score
                        )
                    WHERE id = :id
                """),
                {
                    "id": current["id"],
                    "sources": json.dumps(merged_sources, ensure_ascii=False),
                    "source_raw_article_ids": merged_ids,
                    "source_articles": json.dumps(merged_source_articles, ensure_ascii=False),
                    "importance_score": merged_importance,
                },
            )
            db.commit()
            _sync_card_news_articles(
                card_id=str(current["id"]),
                source_raw_article_ids=merged_ids,
                relation_source="source_raw_article_ids",
            )
            log.info(
                "기존 카드 출처 병합 완료 | card_id=%s cluster_id=%s added=%d",
                current["id"],
                cluster_id,
                len(articles),
            )
            return str(current["id"])
    except Exception as e:
        log.warning(
            "기존 카드 출처 병합 실패 | cluster_id=%s article_ids=%s error=%s",
            cluster_id,
            raw_article_ids,
            e,
        )
        return None


# ──────────────────────────────────────────────────────────────
# evidence_chain 저장
# ──────────────────────────────────────────────────────────────


# V9 (2026-05-12): card_news 테이블 rename 후에도 evidence_chain.issue_card_id 컬럼은
# legacy 이름 유지 (deploy race 회피 — 컬럼 rename 은 V10 분리). Python 변수명은
# card_news_id 로 의미 정렬, SQL column ref 만 issue_card_id 유지.
_INSERT_EVIDENCE = text("""
    INSERT INTO evidence_chain (
        issue_card_id, source_links, provenance, financial_refs,
        mbb_refs, financial_link, evidence_version, pass, missing
    ) VALUES (
        :card_news_id, CAST(:source_links AS jsonb), CAST(:provenance AS jsonb),
        CAST(:financial_refs AS jsonb), CAST(:mbb_refs AS jsonb),
        CAST(:financial_link AS jsonb), :evidence_version, :passed, :missing
    )
    ON CONFLICT (issue_card_id) DO UPDATE SET
        source_links = EXCLUDED.source_links,
        provenance = EXCLUDED.provenance,
        financial_refs = EXCLUDED.financial_refs,
        mbb_refs = EXCLUDED.mbb_refs,
        financial_link = EXCLUDED.financial_link,
        evidence_version = EXCLUDED.evidence_version,
        pass = EXCLUDED.pass,
        missing = EXCLUDED.missing
""")


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


# ──────────────────────────────────────────────────────────────
# pipeline_logs 저장
# ──────────────────────────────────────────────────────────────


_INSERT_PIPELINE_LOG = text("""
    INSERT INTO pipeline_logs (
        pipeline_step, company, input_count, output_count,
        elapsed_ms, llm_tokens_used, error_msg
    ) VALUES (
        :step, :company, :input_count, :output_count,
        :elapsed_ms, :llm_tokens_used, :error_msg
    )
""")


def save_pipeline_log(
    step: str,
    company: Optional[str],
    input_count: int,
    output_count: int,
    elapsed_ms: int,
    llm_tokens_used: int = 0,
    error_msg: Optional[str] = None,
) -> None:
    """파이프라인 단계별 실행 통계를 기록한다.

    backend V30 이후 운영 DB에서는 legacy pipeline_logs 테이블이 제거되었다.
    테이블이 남아 있는 로컬/구버전 DB에서는 기록하고, 없는 DB에서는 조용히 건너뛴다.
    """
    try:
        with SessionLocal() as db:
            exists = db.execute(text("SELECT to_regclass('public.pipeline_logs')")).scalar()
            if exists is None:
                log.debug("pipeline_logs 테이블 없음. 단계 로그 저장 생략 | step=%s", step)
                return
            db.execute(
                _INSERT_PIPELINE_LOG,
                {
                    "step": step,
                    "company": company,
                    "input_count": input_count,
                    "output_count": output_count,
                    "elapsed_ms": elapsed_ms,
                    "llm_tokens_used": llm_tokens_used,
                    "error_msg": error_msg,
                },
            )
            db.commit()
    except Exception as e:
        log.error("pipeline_logs 저장 실패 | step=%s error=%s", step, e)


# ──────────────────────────────────────────────────────────────


_card_id_state: dict[str, int] = {}  # {date_str: last_seq}
_card_id_lock = threading.Lock()


def _generate_card_id(company: str) -> str:
    """IC-YYYYMMDD-NNN 형식의 이슈카드 ID를 생성한다.

    DB 저장 전에 병렬 호출되므로 Lock으로 중복 방지.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    with _card_id_lock:
        if date_str not in _card_id_state:
            with SessionLocal() as db:
                row = db.execute(
                    text("SELECT COUNT(*) FROM card_news WHERE id LIKE :prefix"),
                    {"prefix": f"IC-{date_str}-%"},
                ).fetchone()
                _card_id_state[date_str] = row[0] if row else 0
        _card_id_state[date_str] += 1
        return f"IC-{date_str}-{_card_id_state[date_str]:03d}"


# ──────────────────────────────────────────────────────────────


def _company_for_storage(article: RawArticle) -> list[str]:
    """DB 저장용 company를 반환한다.

    기업이 명시되지 않은 산업 동향 자료는 파이프라인 필터링을 위해
    가상 company bucket으로 저장한다. 일반 무소속 기사는 계속 제외한다.
    """
    if article.company:
        return article.company
    if _is_industry_trend_article(article):
        return [INDUSTRY_TREND_COMPANY]
    return []


def _is_industry_trend_article(article: RawArticle) -> bool:
    if article.source_type in _INDUSTRY_SOURCE_TYPES:
        return True

    extra = article.extra or {}
    report_type = str(extra.get("type") or extra.get("report_type") or "").strip()
    if report_type in _INDUSTRY_REPORT_TYPES:
        return True

    if any(extra.get(key) for key in _INDUSTRY_MARKER_KEYS):
        return True

    return "industry" in (article.source_name or "").lower()


def _is_valid(article: RawArticle, storage_company: Optional[list[str]] = None) -> bool:
    """Gate 1: 최소 품질 필터."""
    if not article.url or not article.title:
        return False
    if not (storage_company if storage_company is not None else article.company):
        return False
    if len(article.content or "") < 10 and len(article.title) < 5:
        return False
    return True


def _source_metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
) -> str:
    meta = _sanitize_jsonish(dict(article.extra))
    if not article.company and storage_company and INDUSTRY_TREND_COMPANY in storage_company:
        meta["topic_scope"] = "industry_trend"
        meta["company_scope"] = "industry"
        meta["company_fallback"] = INDUSTRY_TREND_COMPANY
    elif _is_industry_trend_article(article):
        meta.setdefault("topic_scope", "industry_trend")
    for key in _SOURCE_METADATA_EXCLUDED_KEYS:
        meta.pop(key, None)
    return json.dumps(meta, ensure_ascii=False)


def _metadata_json(
    article: RawArticle,
    storage_company: Optional[list[str]] = None,
    run_context: CrawlRunContext | None = None,
) -> str:
    """Compatibility shim for older tests/callers.

    Crawl run context is now written to crawl_run_articles, not metadata.
    """
    return _source_metadata_json(article, storage_company)


def _upsert_source_metadata(
    db,
    *,
    article_id: int,
    source_type: str,
    source_metadata: str,
) -> None:
    metadata = _metadata_dict(source_metadata)
    parse_payload = _parse_result_payload(metadata)
    collection_metadata = _collection_metadata(metadata)

    if collection_metadata:
        db.execute(
            text("""
                UPDATE raw_articles
                SET metadata = metadata || CAST(:metadata AS jsonb)
                WHERE id = :raw_article_id
            """),
            {
                "raw_article_id": article_id,
                "metadata": json.dumps(collection_metadata, ensure_ascii=False),
            },
        )

    if parse_payload:
        _upsert_parse_result(
            db, article_id=article_id, source_type=source_type, payload=parse_payload
        )


_PARSE_RESULT_METADATA_KEYS = {
    "parser_result",
    "parser_quality_score",
    "parser_quality_label",
    "parser_quality_reason",
    "financial_record",
    "dart_sections",
    "dart_section_tree",
    "dart_document_chunks",
    "dart_classified_tables",
    "dart_financial_statements",
    "ir_sections",
    "ir_document_chunks",
    "securities_report_sections",
    "securities_report_document_chunks",
}


def _collection_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _PARSE_RESULT_METADATA_KEYS and value is not None
    }


def _parse_result_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    parser_result = _metadata_dict(metadata.get("parser_result"))
    if not parser_result and not any(key in metadata for key in _PARSE_RESULT_METADATA_KEYS):
        return {}

    financial_record = (
        metadata.get("financial_record") or parser_result.get("financial_record") or {}
    )
    warnings = parser_result.get("warnings") or metadata.get("warnings") or []
    raw_payload = {
        key: metadata.get(key)
        for key in (
            "parser_quality_score",
            "parser_quality_reason",
            "period",
            "period_year",
            "period_quarter",
            "period_type",
            "topics",
            "topic_signals",
        )
        if metadata.get(key) is not None
    }
    return {
        "parser_version": parser_result.get("parser_version"),
        "parser_ok": bool(parser_result.get("ok")),
        "parser_quality_label": metadata.get("parser_quality_label"),
        "parser_quality_score": metadata.get("parser_quality_score"),
        "parser_quality_reason": metadata.get("parser_quality_reason"),
        "period": metadata.get("period") or parser_result.get("period"),
        "period_year": metadata.get("period_year") or parser_result.get("period_year"),
        "period_quarter": metadata.get("period_quarter") or parser_result.get("period_quarter"),
        "period_type": metadata.get("period_type") or parser_result.get("period_type"),
        "published_at": parser_result.get("published_at"),
        "parser_result": parser_result,
        "financial_record": financial_record if isinstance(financial_record, dict) else {},
        "result_metadata": raw_payload,
        "warnings": warnings if isinstance(warnings, list) else [warnings],
    }


def _upsert_parse_result(
    db,
    *,
    article_id: int,
    source_type: str,
    payload: dict[str, Any],
) -> None:
    db.execute(
        text("""
            INSERT INTO raw_article_parse_results (
                raw_article_id, source_type, parser, parser_ok,
                period, period_year, period_quarter, period_type, published_at,
                parser_quality_score, parser_quality_label, parser_quality_reason,
                financial_record, result_metadata, warnings, raw_result
            ) VALUES (
                :raw_article_id, :source_type, :parser, :parser_ok,
                :period, :period_year, :period_quarter, :period_type, :published_at,
                :parser_quality_score, :parser_quality_label, :parser_quality_reason,
                CAST(:financial_record AS jsonb), CAST(:result_metadata AS jsonb),
                CAST(:warnings AS jsonb), CAST(:raw_result AS jsonb)
            )
            ON CONFLICT (raw_article_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                parser = EXCLUDED.parser,
                parser_ok = EXCLUDED.parser_ok,
                period = EXCLUDED.period,
                period_year = EXCLUDED.period_year,
                period_quarter = EXCLUDED.period_quarter,
                period_type = EXCLUDED.period_type,
                published_at = EXCLUDED.published_at,
                parser_quality_score = EXCLUDED.parser_quality_score,
                parser_quality_label = EXCLUDED.parser_quality_label,
                parser_quality_reason = EXCLUDED.parser_quality_reason,
                financial_record = EXCLUDED.financial_record,
                result_metadata = raw_article_parse_results.result_metadata
                    || EXCLUDED.result_metadata,
                warnings = EXCLUDED.warnings,
                raw_result = EXCLUDED.raw_result,
                updated_at = NOW()
        """),
        {
            "raw_article_id": article_id,
            "source_type": source_type,
            "parser": payload.get("parser_version") or f"{source_type}_parser",
            "parser_ok": payload.get("parser_ok"),
            "period": payload.get("period"),
            "period_year": payload.get("period_year"),
            "period_quarter": payload.get("period_quarter"),
            "period_type": payload.get("period_type"),
            "published_at": payload.get("published_at"),
            "parser_quality_score": payload.get("parser_quality_score"),
            "parser_quality_label": payload.get("parser_quality_label"),
            "parser_quality_reason": payload.get("parser_quality_reason"),
            "financial_record": json.dumps(
                _sanitize_jsonish(payload.get("financial_record") or {}),
                ensure_ascii=False,
            ),
            "result_metadata": json.dumps(
                _sanitize_jsonish(payload.get("result_metadata") or {}),
                ensure_ascii=False,
            ),
            "warnings": json.dumps(
                _sanitize_jsonish(payload.get("warnings") or []),
                ensure_ascii=False,
            ),
            "raw_result": json.dumps(
                _sanitize_jsonish(payload.get("parser_result") or {}),
                ensure_ascii=False,
            ),
        },
    )


def _upsert_crawl_run_article(
    db,
    *,
    article_id: int,
    article: RawArticle,
    run_context: CrawlRunContext | None,
    action: str,
    source_metadata: str,
) -> None:
    if not run_context or not run_context.crawl_run_id:
        return

    raw_payload = {
        "source_name": run_context.source_name or article.source_name,
        "collection_mode": run_context.collection_mode,
        "track": run_context.track,
        "window_start": _iso_or_none(run_context.window_start),
        "window_end": _iso_or_none(run_context.window_end),
        "source_metadata": _json_or_value(source_metadata, {}),
    }
    db.execute(
        _INSERT_CRAWL_RUN_ARTICLE,
        {
            "crawl_run_id": run_context.crawl_run_id,
            "raw_article_id": article_id,
            "url": article.url,
            "url_hash": article.url_hash,
            "discovered_at": article.collected_at,
            "action": action,
            "fetch_status": article.crawl_status,
            "error_message": article.error_message,
            "raw_payload": json.dumps(raw_payload, ensure_ascii=False),
        },
    )


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _sanitize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


# ──────────────────────────────────────────────────────────────────────────
# Global Trends — readers / upsert / trend context cache.
# design: axis-ai/design/30-analysis/global-trends.md §3.2 / §5.2 / §7.
# ──────────────────────────────────────────────────────────────────────────

GLOBAL_NEWSROOM_SOURCE_NAMES: frozenset[str] = frozenset(
    {
        "nvidia_official",
        "microsoft_official",
        "google_official",
        "amazon_official",
        "meta_official",
        "apple_newsroom",
    }
)

GLOBAL_RESEARCH_SOURCE_NAMES: frozenset[str] = frozenset({"spri", "bcg"})

SK_AX_RAW_SOURCE_NAMES: frozenset[str] = frozenset(
    {
        "SK AX Site",
        "SK AX Newsroom",
        "dart",
        "ir_pdf",
    }
)

DEFAULT_PEER_COMPANY_IDS: tuple[str, ...] = (
    "sk_ax",
    "samsung_sds",
    "lg_cns",
    "posco_dx",
    "hyundai_autoever",
)


def fetch_global_trend_inputs(window_days: int = 30) -> list[dict[str, Any]]:
    """글로벌 6사 newsroom + SPRi/BCG raw_articles 를 ITTrendInput 호환 dict 로 반환.

    raw_articles.source_type 이 'official' 인 글로벌 6사 row 는 ITTrendAgent 의
    ``_split_trend_inputs()`` 에서 unsupported_items 로 떨어지므로, 여기서 source_type 을
    'global_newsroom' / 'trend_report' 로 정규화한다 (design §4.1 옵션 B 채택).
    """
    names = list(GLOBAL_NEWSROOM_SOURCE_NAMES | GLOBAL_RESEARCH_SOURCE_NAMES)
    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    """
                SELECT id, source_name, source_type, publisher, title, content, url,
                       published_at, collected_at, metadata, company
                FROM raw_articles
                WHERE source_name = ANY(:names)
                  AND collected_at >= NOW() - make_interval(days => :days)
                ORDER BY collected_at DESC NULLS LAST
                """
                ),
                {"names": names, "days": window_days},
            )
            .mappings()
            .all()
        )

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source_id"] = item["id"]
        source_name = (item.get("source_name") or "").strip().lower()
        if source_name in GLOBAL_NEWSROOM_SOURCE_NAMES:
            item["source_type"] = "global_newsroom"
        elif source_name in GLOBAL_RESEARCH_SOURCE_NAMES:
            item["source_type"] = "trend_report"
            item.setdefault("publisher", source_name)
        items.append(item)
    return items


def _ilike_any_clause(
    col_exprs: Sequence[str],
    terms: Sequence[str],
    prefix: str,
) -> tuple[str, dict[str, str]]:
    """terms 중 하나라도 col_exprs 중 한 곳에 ILIKE 매칭되면 참이 되는 SQL 절을 만든다.

    각 ``col_exprs`` 항목은 ``{k}`` placeholder 를 포함한다 (예: ``"title ILIKE :{k}"``).
    term 1개당 모든 컬럼을 OR 로 묶고, term 들 사이도 OR — 즉 "아무 변형이든 어디서든 걸리면 매칭".
    terms 가 비면 ``("", {})`` 을 돌려 호출부에서 절을 생략하게 한다.

    Args:
        col_exprs: ``{k}`` 를 가진 컬럼 매칭 표현식들.
        terms: ILIKE 로 감쌀 검색어들 (한·영 변형 포함).
        prefix: 바인드 파라미터 이름 접두사 (호출부 간 충돌 방지).

    Returns:
        ``(sql_clause, params)``. ``sql_clause`` 는 바깥을 괄호로 감싼 OR 절.
    """
    if not terms:
        return "", {}
    params: dict[str, str] = {}
    groups: list[str] = []
    for idx, term in enumerate(terms):
        key = f"{prefix}{idx}"
        params[key] = f"%{term}%"
        cols = " OR ".join(expr.format(k=key) for expr in col_exprs)
        groups.append(f"({cols})")
    return "(" + " OR ".join(groups) + ")", params


def fetch_sk_ax_raw_for_alignment(
    *,
    window_days: int,
    match_terms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """SK AX 자사 raw 를 peer alignment 용으로 fetch.

    ``card_news.peer_company_id='sk_ax'`` 가 0 건이라 card 가 아닌 raw_articles 에서
    직접 가져온다 (design §6 Phase 3 분기). ``match_terms`` 는 키워드의 한·영 변형 목록.
    """
    params: dict[str, Any] = {
        "names": list(SK_AX_RAW_SOURCE_NAMES),
        "days": window_days,
    }
    match_sql, match_params = _ilike_any_clause(
        ["title ILIKE :{k}", "content ILIKE :{k}"], match_terms or [], "kw"
    )
    keyword_clause = f"AND {match_sql}" if match_sql else ""
    params.update(match_params)

    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, source_name, source_type, title, content, url,
                       published_at, collected_at, metadata
                FROM raw_articles
                WHERE source_name = ANY(:names)
                  AND collected_at >= NOW() - make_interval(days => :days)
                  {keyword_clause}
                ORDER BY COALESCE(published_at, collected_at) DESC NULLS LAST
                LIMIT 100
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def fetch_peer_cards_for_alignment(
    *,
    peer_id: str,
    window_days: int,
    match_terms: Sequence[str] | None = None,
    importance_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Peer 4사 (samsung_sds / lg_cns / posco_dx / hyundai_autoever) 의 card_news fetch.

    SK AX 는 card_news 0 건이므로 ``fetch_sk_ax_raw_for_alignment`` 사용.
    importance_threshold 미만 카드는 naver_news noise 제거 (design §13).

    ``match_terms`` 는 트렌드 키워드의 한·영 변형 목록 — 트렌드 키워드(theme)는 영문
    정규형이고 국내 피어 card_news 는 한글이라, 변형을 OR ILIKE 로 함께 건다.
    과거의 ``primary_keyword_category`` 동일성 필터는 제거됨: card_news 의 sector
    분류(ax/security/infra/deal)와 트렌드 토픽 카테고리(ai_tech/ai_infra/cloud…)가
    서로 다른 분류 체계라, AND 비교가 AI 계열 트렌드를 항상 0건으로 만들었음.
    """
    if peer_id == "sk_ax":
        return fetch_sk_ax_raw_for_alignment(window_days=window_days, match_terms=match_terms)

    params: dict[str, Any] = {
        "peer": peer_id,
        "days": window_days,
        "importance": importance_threshold,
    }
    # global_search_text 는 gin_trgm_ops 인덱스 — ILIKE 빠름.
    # title + global_search_text + keywords[] (text[]) 3 곳에서 매칭.
    match_sql, match_params = _ilike_any_clause(
        [
            "title ILIKE :{k}",
            "global_search_text ILIKE :{k}",
            "EXISTS (SELECT 1 FROM unnest(keywords) kw WHERE kw ILIKE :{k})",
        ],
        match_terms or [],
        "kw",
    )
    params.update(match_params)

    filter_sql = f" AND {match_sql}" if match_sql else ""

    with SessionLocal() as db:
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, title, summary_lines, primary_keyword_category,
                       peer_company_id, importance_score, created_at,
                       keywords, keyword_categories
                FROM card_news
                WHERE peer_company_id = :peer
                  AND status = 'ACTIVE'
                  AND created_at >= NOW() - make_interval(days => :days)
                  AND COALESCE(importance_score, 0) >= :importance
                  {filter_sql}
                ORDER BY created_at DESC NULLS LAST
                LIMIT 50
                """
                ),
                params,
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


_GLOBAL_TREND_UPSERT_SQL = text(
    """
    INSERT INTO global_industry_trends (
        source_analysis_id, trend_date, industry, region, keyword, keyword_category,
        title, summary, mention_count, impact_score, confidence,
        related_peer_ids, related_card_ids, source_raw_article_ids,
        sk_ax_implication, payload
    ) VALUES (
        :source_analysis_id, :trend_date, :industry, :region, :keyword, :keyword_category,
        :title, :summary, :mention_count, :impact_score, :confidence,
        :related_peer_ids, :related_card_ids, :source_raw_article_ids,
        :sk_ax_implication, CAST(:payload AS JSONB)
    )
    ON CONFLICT (trend_date, industry, region, keyword)
    DO UPDATE SET
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        mention_count = EXCLUDED.mention_count,
        impact_score = EXCLUDED.impact_score,
        confidence = EXCLUDED.confidence,
        related_peer_ids = EXCLUDED.related_peer_ids,
        related_card_ids = EXCLUDED.related_card_ids,
        source_raw_article_ids = EXCLUDED.source_raw_article_ids,
        sk_ax_implication = EXCLUDED.sk_ax_implication,
        payload = EXCLUDED.payload,
        updated_at = NOW()
    """
)


def upsert_global_industry_trends(rows: list[dict[str, Any]]) -> int:
    """V29 ``uq_global_industry_trends_daily_keyword`` 로 idempotent upsert.

    같은 날 cronjob 이 두 번 돌아도 안전. ``source_analysis_id`` 는 첫 INSERT
    시점 보존. ``payload`` 는 dict 면 자동 JSON 직렬화.
    design: global-trends.md §7.
    """
    if not rows:
        return 0
    serialized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        payload = item.get("payload")
        if isinstance(payload, (dict, list)):
            item["payload"] = json.dumps(payload, ensure_ascii=False, default=str)
        elif payload is None:
            item["payload"] = "{}"
        for key in ("related_peer_ids", "related_card_ids", "source_raw_article_ids"):
            item[key] = list(item.get(key) or [])
        serialized.append(item)
    with SessionLocal() as db:
        db.execute(_GLOBAL_TREND_UPSERT_SQL, serialized)
        db.commit()
    return len(serialized)


def fetch_previous_trend_context_for_delta() -> dict[str, Any]:
    """오늘 이전 가장 최근 ``global_industry_trends`` batch → Phase 2 delta 입력.

    design: ``global-trends.md`` §16 — ledger 대신 self-read.
    prior batch 가 없으면 빈 dict (cold start → ``frequency_delta_pct = 0``).
    """
    query = text(
        """
                WITH latest_prior AS (
                    SELECT MAX(trend_date) AS d
                    FROM global_industry_trends
                    WHERE trend_date < CURRENT_DATE
                )
                SELECT g.keyword, g.mention_count, g.payload, g.trend_date
                FROM global_industry_trends g
                INNER JOIN latest_prior lp ON g.trend_date = lp.d
                ORDER BY g.impact_score DESC NULLS LAST, g.keyword ASC
                """
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(query).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if not _is_missing_relation(exc, "global_industry_trends"):
            raise
        rows = []

    if not rows:
        return {}

    keyword_counts: dict[str, int] = {}
    signals: list[dict[str, Any]] = []
    for row in rows:
        kw = str(row["keyword"] or "").strip().lower()
        if not kw:
            continue
        payload = row["payload"] if isinstance(row["payload"], dict) else {}
        mention_count = int(row["mention_count"] or payload.get("mention_count") or 0)
        keyword_counts[kw] = mention_count
        signals.append(
            {
                "signal": kw,
                "mention_count": mention_count,
                "intensity": payload.get("intensity"),
                "leading_companies": payload.get("leading_companies", []),
            }
        )

    latest_date = rows[0]["trend_date"]
    return {
        "period": "prior_batch",
        "keyword_counts": keyword_counts,
        "signals": signals,
        "source_groups": ["global_industry_trends"],
        "updated_at": latest_date.isoformat()
        if hasattr(latest_date, "isoformat")
        else str(latest_date),
        "metadata": {"row_count": len(rows), "prior_trend_date": str(latest_date)},
    }


# trend 는 cronjob 으로 일 1 회만 갱신되므로 60 초 캐시는 정합성 손실 거의 없음.
_TREND_CACHE_TTL_SEC = 60
_trend_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_trend_cache_lock = threading.Lock()


def fetch_latest_trend_context(within_days: int = 7) -> dict[str, Any]:
    """``global_industry_trends`` 의 최근 N 일 row 를 TrendContext shape 로 aggregate.

    StrategicAnalyzer prompt 에 들어갈 글로벌 배경 정보. trend 가 없으면 빈 dict 반환
    (``strategic_analyzer.py`` 가 already-empty-safe). process-level TTL 60 초 캐시.
    """
    import time

    now = time.monotonic()
    with _trend_cache_lock:
        cached = _trend_cache.get(within_days)
        if cached and (now - cached[0]) < _TREND_CACHE_TTL_SEC:
            return cached[1]

    query = text(
        """
                SELECT keyword, summary, payload, trend_date, source_analysis_id
                FROM global_industry_trends
                WHERE trend_date >= CURRENT_DATE - make_interval(days => :days)
                ORDER BY impact_score DESC NULLS LAST, trend_date DESC
                LIMIT 10
                """
    )
    try:
        with SessionLocal() as db:
            rows = db.execute(query, {"days": within_days}).mappings().all()
    except Exception as exc:  # noqa: BLE001
        if not _is_missing_relation(exc, "global_industry_trends"):
            raise
        rows = []

    result: dict[str, Any]
    if not rows:
        result = {}
    else:
        signals: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        trend_lines: list[str] = []
        for row in rows:
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            summary = row["summary"] or ""
            if summary:
                trend_lines.append(summary)
            signals.append(
                {
                    "signal": row["keyword"],
                    "intensity": payload.get("intensity"),
                    "leading_companies": payload.get("leading_companies", []),
                    "source_ids": [row["source_analysis_id"]] if row["source_analysis_id"] else [],
                }
            )
            sources.append(
                {
                    "source_id": row["source_analysis_id"],
                    "title": row["keyword"],
                }
            )
        latest_date = rows[0]["trend_date"]
        result = {
            "period": f"last_{within_days}d",
            "trend_summary": " / ".join(trend_lines[:3]),
            "trend_lines": trend_lines,
            "signals": signals,
            "source_groups": ["global_industry_trends"],
            "sources": sources,
            "reference_issue_ids": [],
            "updated_at": latest_date.isoformat()
            if hasattr(latest_date, "isoformat")
            else str(latest_date),
            "validation": {"pass": True},
            "metadata": {"row_count": len(rows)},
        }

    with _trend_cache_lock:
        _trend_cache[within_days] = (now, result)
    return result


def invalidate_trend_context_cache() -> None:
    """ITTrendAgent 가 cron 으로 새 trend 를 upsert 한 직후 호출하면 즉시 반영."""
    with _trend_cache_lock:
        _trend_cache.clear()


def _is_missing_column(exc: Exception, column_name: str) -> bool:
    """Return True when a read path hit a missing optional DB column."""
    text_value = str(exc)
    return "UndefinedColumn" in text_value and column_name in text_value


def _is_missing_relation(exc: Exception, relation_name: str) -> bool:
    text_value = str(exc)
    return "UndefinedTable" in text_value and relation_name in text_value
