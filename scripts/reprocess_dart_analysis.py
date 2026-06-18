# 작성일: 2026-05-18
# 작성자: 박지원
# 변경이력:
#   2026-05-18 박지원 — DART/IR 파서 작성 및 재파싱·분석 테이블 갱신 스크립트 추가·수정
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
import json
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
from src.parsers.dart_parser import DartParser, extract_dart_storage_content
from src.parsers.parser_quality import analyze_parser_quality_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reprocess_dart_analysis")

_PERIODIC_REPORT_NAME_SQL = """
                    AND (
                        ra.title LIKE '%사업보고서%'
                        OR ra.title LIKE '%반기보고서%'
                        OR ra.title LIKE '%분기보고서%'
                    )
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DART parser 결과 및 분석 fact/signal 재생성")
    parser.add_argument("--limit", type=int, default=0, help="처리할 DART 문서 수 제한")
    parser.add_argument(
        "--peer-id",
        default="",
        help="특정 peer_id/company만 처리. 예: sk_ax",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="metrics/signals 삭제·적재를 몇 개 문서 단위로 flush할지 지정",
    )
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
        "--index-vector",
        action="store_true",
        help="DART I/II document chunks를 Qdrant axis_documents에 upsert",
    )
    parser.add_argument(
        "--compact-content",
        action="store_true",
        help="raw_articles.content를 DART I/II 섹션만 남기도록 갱신",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        not args.reparse
        and not args.upsert_metrics
        and not args.upsert_signals
        and not args.index_vector
        and not args.compact_content
    ):
        raise SystemExit(
            "--reparse, --upsert-metrics, --upsert-signals, "
            "--index-vector, --compact-content 중 하나 이상을 지정하세요."
        )

    _ensure_parse_results_schema()
    if args.upsert_metrics or args.replace_metrics:
        _ensure_financial_metrics_schema()
    if args.upsert_signals or args.replace_signals:
        _ensure_business_signals_schema()
    articles = _load_dart_articles(limit=args.limit, peer_id=args.peer_id or None)
    log.info("DART 문서 로드 완료 | count=%d", len(articles))

    batch_size = max(1, int(args.batch_size))
    batch_article_ids: list[int] = []
    batch_metrics: list[dict[str, Any]] = []
    batch_signals: list[dict[str, Any]] = []
    total_metrics = 0
    total_signals = 0

    for index, article in enumerate(articles, start=1):
        article_id = int(article["id"])
        parser_result = (
            _reparse_dart_article(article) if args.reparse else _stored_parser_result(article)
        )

        if args.upsert_metrics:
            metrics = financial_metrics_from_dart(article, parser_result)
            batch_metrics.extend(metrics)

        if args.upsert_signals:
            signals = business_signals_from_dart(article, parser_result)
            batch_signals.extend(signals)

        if args.index_vector:
            _index_dart_chunks(article, parser_result)

        if args.compact_content:
            _compact_article_content(article)

        batch_article_ids.append(article_id)
        log.info(
            "DART 문서 처리 완료 | id=%s period=%s metrics=%d signals=%d batch=%d/%d",
            article["id"],
            parser_result.get("period") or article["extra"].get("period"),
            len(metrics) if args.upsert_metrics else 0,
            len(signals) if args.upsert_signals else 0,
            len(batch_article_ids),
            batch_size,
        )

        if len(batch_article_ids) >= batch_size:
            flushed_metrics, flushed_signals = _flush_analysis_batch(
                article_ids=batch_article_ids,
                metrics=batch_metrics,
                signals=batch_signals,
                upsert_metrics=args.upsert_metrics,
                replace_metrics=args.replace_metrics,
                upsert_signals=args.upsert_signals,
                replace_signals=args.replace_signals,
            )
            total_metrics += flushed_metrics
            total_signals += flushed_signals
            batch_article_ids = []
            batch_metrics = []
            batch_signals = []
            log.info(
                "DART batch flush 완료 | processed=%d/%d total_metrics=%d total_signals=%d",
                index,
                len(articles),
                total_metrics,
                total_signals,
            )

    flushed_metrics, flushed_signals = _flush_analysis_batch(
        article_ids=batch_article_ids,
        metrics=batch_metrics,
        signals=batch_signals,
        upsert_metrics=args.upsert_metrics,
        replace_metrics=args.replace_metrics,
        upsert_signals=args.upsert_signals,
        replace_signals=args.replace_signals,
    )
    total_metrics += flushed_metrics
    total_signals += flushed_signals
    log.info(
        "DART 재전처리 완료 | articles=%d metrics=%d signals=%d batch_size=%d",
        len(articles),
        total_metrics,
        total_signals,
        batch_size,
    )


def _flush_analysis_batch(
    *,
    article_ids: list[int],
    metrics: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    upsert_metrics: bool,
    replace_metrics: bool,
    upsert_signals: bool,
    replace_signals: bool,
) -> tuple[int, int]:
    if not article_ids:
        return 0, 0

    metric_count = 0
    signal_count = 0
    if upsert_metrics:
        if replace_metrics:
            deleted_count = delete_raw_article_financial_metrics(
                article_ids,
                source_type="dart",
            )
            log.info("기존 DART financial metrics batch 삭제 완료 | count=%d", deleted_count)
        metric_count = upsert_raw_article_financial_metrics(metrics)
        log.info("DART financial metrics batch upsert 완료 | count=%d", metric_count)

    if upsert_signals:
        if replace_signals:
            deleted_count = delete_raw_article_business_signals(
                article_ids,
                source_type="dart",
            )
            log.info("기존 DART business signals batch 삭제 완료 | count=%d", deleted_count)
        signal_count = upsert_raw_article_business_signals(signals)
        log.info("DART business signals batch upsert 완료 | count=%d", signal_count)

    return metric_count, signal_count


def _load_dart_articles(*, limit: int = 0, peer_id: str | None = None) -> list[dict[str, Any]]:
    limit_sql = "LIMIT :limit" if limit > 0 else ""
    params: dict[str, Any] = {}
    if limit > 0:
        params["limit"] = limit
    if peer_id:
        params["peer_id"] = peer_id
    with SessionLocal() as db:
        rows = db.execute(
            text(_dart_article_select_sql(limit_sql, peer_id=peer_id)),
            params,
        ).fetchall()

    articles = []
    for row in rows:
        article = dict(row._mapping)
        article["extra"] = article.get("metadata") or {}
        articles.append(article)

    return articles


def _dart_article_select_sql(limit_sql: str, *, peer_id: str | None = None) -> str:
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

    peer_filter_sql = (
        """
                AND (
                    ra.company ? :peer_id
                    OR {metadata_expr}->>'peer_id' = :peer_id
                    OR {metadata_expr}->>'company' = :peer_id
                    OR {metadata_expr}->>'corp_name' = :peer_id
                )
    """.format(metadata_expr=metadata_expr)
        if peer_id
        else ""
    )

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
                {_PERIODIC_REPORT_NAME_SQL}
                {peer_filter_sql}
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
    storage_parser_result = _compact_parser_result_for_storage(parser_result)

    metadata_patch = {
        "parser_result": storage_parser_result,
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
        "dart_document_chunks": storage_parser_result.get("document_chunks"),
        "dart_classified_tables": parser_result.get("classified_tables"),
        "dart_financial_statements": parser_result.get("financial_statements"),
    }

    update_preprocess_status(
        int(article["id"]),
        "PROCESSED" if ok else "SKIPPED",
        {
            **metadata_patch,
            "status_detail": "parsed_document" if ok else "parser_quality_failed",
            **({} if ok else {"skip_reason": reason}),
        },
        error_message=None if ok else reason,
    )
    article["extra"] = {**article["extra"], **metadata_patch}
    return parser_result


def _compact_parser_result_for_storage(parser_result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(parser_result)
    chunks = compact.get("document_chunks")
    if isinstance(chunks, list):
        compact["document_chunks"] = [_chunk_without_text(chunk) for chunk in chunks]
    return compact


def _chunk_without_text(chunk: Any) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}
    text = str(chunk.get("text") or "")
    return {key: value for key, value in chunk.items() if key != "text"} | {
        "text_chars": len(text) or chunk.get("text_chars"),
        "text_omitted_for_metadata": True,
    }


def _index_dart_chunks(article: dict[str, Any], parser_result: dict[str, Any]) -> None:
    from src.rag.document_index import index_dart_chunks

    if not _has_chunk_text(parser_result.get("document_chunks")):
        reparsed = DartParser().parse_article(article)
        parser_result = {
            **parser_result,
            "document_chunks": reparsed.get("document_chunks") or [],
            "sections": reparsed.get("sections") or parser_result.get("sections"),
            "section_index": reparsed.get("section_index") or parser_result.get("section_index"),
            "topic_signals": reparsed.get("topic_signals") or parser_result.get("topic_signals"),
            "topics": reparsed.get("topics") or parser_result.get("topics"),
        }
    point_ids = index_dart_chunks(article=article, parser_result=parser_result)
    log.info("DART vector index 완료 | id=%s chunks=%d", article.get("id"), len(point_ids))


def _has_chunk_text(chunks: Any) -> bool:
    if not isinstance(chunks, list):
        return False
    return any(isinstance(chunk, dict) and chunk.get("text") for chunk in chunks)


def _compact_article_content(article: dict[str, Any]) -> None:
    content, stored_sections = extract_dart_storage_content(str(article.get("content") or ""))
    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET content = :content,
                    metadata = metadata || CAST(:metadata AS jsonb)
                WHERE id = :id
            """),
            {
                "id": int(article["id"]),
                "content": content,
                "metadata": json.dumps(
                    {
                        "stored_sections": stored_sections,
                        "content_chars": len(content),
                        "content_compacted_for_rdb": True,
                    },
                    ensure_ascii=False,
                ),
            },
        )
        db.commit()
    article["content"] = content
    article["extra"] = {
        **article.get("extra", {}),
        "stored_sections": stored_sections,
        "content_chars": len(content),
        "content_compacted_for_rdb": True,
    }


def _stored_parser_result(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("extra") or {}
    parser_result = metadata.get("parser_result")
    return parser_result if isinstance(parser_result, dict) else {}


if __name__ == "__main__":
    main()
