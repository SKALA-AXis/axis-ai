"""article_store_parts _constants — extracted from facade (move-only)."""

from __future__ import annotations

import re

from sqlalchemy import text

INDUSTRY_TREND_COMPANY = "industry_trend"


_INDUSTRY_SOURCE_TYPES = {"trend_report", "search_trend"}


_INDUSTRY_MARKER_KEYS = {"sector", "industry", "upjong_code"}


_INDUSTRY_REPORT_TYPES = {"industry", "industry_report", "sector_report"}


_SELECT_ARTICLE_ID_BY_URL = text("SELECT id FROM raw_articles WHERE url = :url")


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


_INSERT_PIPELINE_LOG = text("""
    INSERT INTO pipeline_logs (
        pipeline_step, company, input_count, output_count,
        elapsed_ms, llm_tokens_used, error_msg
    ) VALUES (
        :step, :company, :input_count, :output_count,
        :elapsed_ms, :llm_tokens_used, :error_msg
    )
""")


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


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
