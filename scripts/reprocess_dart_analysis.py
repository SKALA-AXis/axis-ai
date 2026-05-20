"""DB에 저장된 DART 문서를 재파싱하고 분석용 fact/signal 테이블을 갱신한다.

크롤링/API 호출 없이 raw_articles와 현재 메타데이터/파싱 결과 테이블의 기존 원문과
parser_result를 다시 처리한다.

사용 예:
  uv run python scripts/reprocess_dart_analysis.py --reparse --upsert-metrics
  uv run python scripts/reprocess_dart_analysis.py --upsert-signals --limit 20
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from src.db.article_store import (
    delete_raw_article_business_signals,
    delete_raw_article_financial_metrics,
    update_preprocess_status,
    upsert_raw_article_business_signals,
    upsert_raw_article_financial_metrics,
)
from src.db.postgres import SessionLocal
from src.extractors.dart_analysis_extractor import (
    business_signals_from_dart,
    financial_metrics_from_dart,
)
from src.extractors.dart_llm_analysis_extractor import analyze_dart_with_llm
from src.parsers.dart_parser import DartParser
from src.parsers.parser_quality import analyze_parser_quality_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reprocess_dart_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DART parser 결과 및 분석 fact/signal 재생성")
    parser.add_argument("--limit", type=int, default=0, help="처리할 DART 문서 수 제한")
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="DartParser를 다시 실행해 DART parser 관련 필드를 갱신",
    )
    parser.add_argument(
        "--upsert-metrics",
        action="store_true",
        help="DART 재무제표를 raw_article_financial_metrics에 upsert",
    )
    parser.add_argument(
        "--upsert-signals",
        action="store_true",
        help="DART document chunks를 raw_article_business_signals에 upsert",
    )
    parser.add_argument(
        "--replace-metrics",
        action="store_true",
        help="처리 대상 DART 문서의 기존 financial metrics를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--replace-signals",
        action="store_true",
        help="처리 대상 DART 문서의 기존 business signals를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--llm-analysis",
        action="store_true",
        help="OPENAI_API_KEY를 사용해 DART 회사/사업 섹션 LLM 보조 분석을 실행",
    )
    parser.add_argument(
        "--llm-max-chunks",
        type=int,
        default=0,
        help="LLM 보조 분석에 보낼 최대 DART chunk 수. 0이면 DART_LLM_MAX_CHUNKS 환경값 사용",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.reparse and not args.upsert_metrics and not args.upsert_signals:
        raise SystemExit("--reparse, --upsert-metrics, --upsert-signals 중 하나 이상을 지정하세요.")

    _ensure_parse_results_schema()
    if args.upsert_metrics or args.replace_metrics:
        _ensure_financial_metrics_schema()
    if args.upsert_signals or args.replace_signals:
        _ensure_business_signals_schema()
    articles = _load_dart_articles(limit=args.limit)
    log.info("DART 문서 로드 완료 | count=%d", len(articles))

    all_metrics: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
    for article in articles:
        parser_result = (
            _reparse_dart_article(article) if args.reparse else _stored_parser_result(article)
        )
        if args.llm_analysis:
            parser_result = _attach_llm_analysis(
                article,
                parser_result,
                llm_max_chunks=args.llm_max_chunks or None,
            )
            article["extra"] = {**article["extra"], "parser_result": parser_result}
            update_preprocess_status(
                int(article["id"]),
                "PREPROCESSED_PARSED_DOCUMENT",
                {
                    "parser_result": parser_result,
                    "dart_llm_business_signals": parser_result.get("llm_business_signals"),
                },
            )

        if args.upsert_metrics:
            metrics = financial_metrics_from_dart(article, parser_result)
            all_metrics.extend(metrics)

        if args.upsert_signals:
            signals = business_signals_from_dart(article, parser_result)
            all_signals.extend(signals)

        log.info(
            "DART 처리 완료 | id=%s period=%s metrics=%d signals=%d",
            article["id"],
            parser_result.get("period") or article["extra"].get("period"),
            len(all_metrics),
            len(all_signals),
        )

    article_ids = [int(article["id"]) for article in articles]
    if args.upsert_metrics:
        if args.replace_metrics:
            deleted_count = delete_raw_article_financial_metrics(
                article_ids,
                source_type="dart",
            )
            log.info("기존 DART financial metrics 삭제 완료 | count=%d", deleted_count)
        count = upsert_raw_article_financial_metrics(all_metrics)
        log.info("DART financial metrics upsert 완료 | count=%d", count)

    if args.upsert_signals:
        if args.replace_signals:
            deleted_count = delete_raw_article_business_signals(
                article_ids,
                source_type="dart",
            )
            log.info("기존 DART business signals 삭제 완료 | count=%d", deleted_count)
        count = upsert_raw_article_business_signals(all_signals)
        log.info("DART business signals upsert 완료 | count=%d", count)


def _load_dart_articles(*, limit: int = 0) -> list[dict[str, Any]]:
    limit_sql = "LIMIT :limit" if limit > 0 else ""
    params = {"limit": limit} if limit > 0 else {}
    with SessionLocal() as db:
        rows = db.execute(text(_dart_article_select_sql(limit_sql)), params).fetchall()

    articles = []
    for row in rows:
        article = dict(row._mapping)
        article["extra"] = article.get("metadata") or {}
        articles.append(article)

    return articles


def _dart_article_select_sql(limit_sql: str) -> str:
    """Return a DART article query for legacy, unified, or parse-result schemas."""
    with SessionLocal() as db:
        has_unified = _table_exists(db, "raw_article_metadata_unified")
        has_source_metadata = _table_exists(db, "raw_article_source_metadata")
        has_legacy_dart = _table_exists(db, "raw_article_metadata_dart")
        has_parse_results = _table_exists(db, "raw_article_parse_results")

    if has_unified:
        metadata_join = """
                LEFT JOIN raw_article_metadata_unified md
                  ON md.raw_article_id = ra.id
            """
        metadata_expr = "COALESCE(md.metadata, md.source_metadata, '{}'::jsonb)"
    elif has_source_metadata:
        metadata_join = """
                LEFT JOIN raw_article_source_metadata md
                  ON md.raw_article_id = ra.id
                 AND md.source_type = 'dart'
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    elif has_legacy_dart:
        metadata_join = """
                LEFT JOIN raw_article_metadata_dart md
                  ON md.raw_article_id = ra.id
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    else:
        metadata_join = ""
        metadata_expr = "COALESCE(ra.metadata, '{}'::jsonb)"

    parse_join = ""
    parser_result_expr = "'{}'::jsonb"
    financial_record_expr = "'{}'::jsonb"
    warnings_expr = "'[]'::jsonb"
    if has_parse_results:
        parse_join = """
                LEFT JOIN raw_article_parse_results pr
                  ON pr.raw_article_id = ra.id
            """
        parser_result_expr = "COALESCE(pr.raw_result, '{}'::jsonb)"
        financial_record_expr = "COALESCE(pr.financial_record, '{}'::jsonb)"
        warnings_expr = "COALESCE(pr.warnings, '[]'::jsonb)"

    return f"""
                SELECT
                    ra.id,
                    ra.company,
                    ra.title,
                    ra.content,
                    ra.url,
                    ra.source_type,
                    ra.source_name,
                    ra.content_type,
                    ra.published_at,
                    ra.collected_at,
                    (
                        {metadata_expr}
                        || jsonb_build_object(
                            'parser_result', {parser_result_expr},
                            'financial_record', {financial_record_expr},
                            'warnings', {warnings_expr}
                        )
                    ) AS metadata
                FROM raw_articles ra
                {metadata_join}
                {parse_join}
                WHERE ra.source_type = 'dart'
                ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                {limit_sql}
            """


def _table_exists(db: Any, table_name: str) -> bool:
    return bool(
        db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": f"public.{table_name}"},
        ).scalar_one_or_none()
    )


def _ensure_parse_results_schema() -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS raw_article_parse_results (
                    raw_article_id BIGINT PRIMARY KEY
                        REFERENCES raw_articles(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    parser TEXT,
                    parser_ok BOOLEAN,
                    period TEXT,
                    period_year INTEGER,
                    period_quarter INTEGER,
                    period_type TEXT,
                    published_at TIMESTAMPTZ,
                    parser_quality_score DOUBLE PRECISION,
                    parser_quality_label TEXT,
                    parser_quality_reason TEXT,
                    financial_record JSONB NOT NULL DEFAULT '{}'::jsonb,
                    result_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    raw_result JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_raw_article_parse_results_source_period
                ON raw_article_parse_results (source_type, period_year, period_quarter)
            """)
        )
        db.commit()


def _ensure_financial_metrics_schema() -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS raw_article_financial_metrics (
                    id BIGSERIAL PRIMARY KEY,
                    raw_article_id BIGINT NOT NULL
                        REFERENCES raw_articles(id) ON DELETE CASCADE,
                    metric_uid TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT,
                    peer_id TEXT,
                    period TEXT,
                    period_year INTEGER,
                    period_quarter INTEGER,
                    period_type TEXT,
                    metric_name TEXT NOT NULL,
                    metric_label TEXT,
                    metric_scope TEXT,
                    business_area TEXT,
                    value_numeric NUMERIC,
                    value_krwbn NUMERIC,
                    value_krw NUMERIC,
                    unit TEXT,
                    currency TEXT,
                    source_page INTEGER,
                    source_table_uid TEXT,
                    source_chunk_uid TEXT,
                    confidence DOUBLE PRECISION,
                    extraction_method TEXT,
                    evidence_text TEXT,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (raw_article_id, metric_uid)
                )
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_raw_article_financial_metrics_peer_period
                ON raw_article_financial_metrics (peer_id, period_year, period_quarter)
            """)
        )
        db.commit()


def _ensure_business_signals_schema() -> None:
    with SessionLocal() as db:
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS raw_article_business_signals (
                    id BIGSERIAL PRIMARY KEY,
                    raw_article_id BIGINT NOT NULL
                        REFERENCES raw_articles(id) ON DELETE CASCADE,
                    signal_uid TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT,
                    peer_id TEXT,
                    period TEXT,
                    period_year INTEGER,
                    period_quarter INTEGER,
                    period_type TEXT,
                    business_area TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    sentiment TEXT,
                    summary TEXT NOT NULL,
                    evidence_text TEXT,
                    source_page INTEGER,
                    source_chunk_uid TEXT,
                    confidence DOUBLE PRECISION,
                    extraction_method TEXT,
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (raw_article_id, signal_uid)
                )
            """)
        )
        db.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_raw_article_business_signals_peer_period
                ON raw_article_business_signals (peer_id, period_year, period_quarter)
            """)
        )
        db.commit()


def _reparse_dart_article(article: dict[str, Any]) -> dict[str, Any]:
    parser = DartParser()
    parsed = parser.parse_article(article)
    item, ok, reason = analyze_parser_quality_article(article)
    parser_result = item.get("parser_result") or parsed

    metadata_patch = {
        "parser_result": parser_result,
        "parser_quality_score": item.get("parser_quality_score"),
        "parser_quality_label": item.get("parser_quality_label"),
        "parser_quality_reason": item.get("parser_quality_reason"),
        "period": parser_result.get("period"),
        "period_year": parser_result.get("period_year"),
        "period_quarter": parser_result.get("period_quarter"),
        "period_type": parser_result.get("period_type"),
        "financial_record": parser_result.get("financial_record"),
        "topics": parser_result.get("topics"),
        "topic_signals": parser_result.get("topic_signals"),
        "dart_sections": parser_result.get("sections"),
        "dart_section_tree": parser_result.get("section_tree"),
        "dart_document_chunks": parser_result.get("document_chunks"),
        "dart_classified_tables": parser_result.get("classified_tables"),
        "dart_financial_statements": parser_result.get("financial_statements"),
    }

    update_preprocess_status(
        int(article["id"]),
        "PREPROCESSED_PARSED_DOCUMENT" if ok else "SKIPPED_PARSER_QUALITY",
        metadata_patch,
        error_message=None if ok else reason,
    )
    article["extra"] = {**article["extra"], **metadata_patch}
    return parser_result


def _attach_llm_analysis(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    llm_max_chunks: int | None = None,
) -> dict[str, Any]:
    analysis = analyze_dart_with_llm(article, parser_result, max_chunks=llm_max_chunks)
    llm_signals = analysis.get("llm_business_signals") or []
    return {
        **parser_result,
        "llm_business_signals": llm_signals,
    }


def _stored_parser_result(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("extra") or {}
    parser_result = metadata.get("parser_result")
    return parser_result if isinstance(parser_result, dict) else {}


if __name__ == "__main__":
    main()
