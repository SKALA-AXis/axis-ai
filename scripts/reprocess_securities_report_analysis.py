# 작성일: 2026-05-18
# 작성자: 박지원
# 변경이력:
#   2026-05-18 박지원 — 증권사 리포트 재파싱 및 fact/signal 갱신 스크립트 작성
"""DB에 저장된 증권사 리포트를 재파싱하고 분석용 fact/signal 테이블을 갱신한다.

사용 예:
  uv run python scripts/reprocess_securities_report_analysis.py --reparse --upsert-metrics
  uv run python scripts/reprocess_securities_report_analysis.py --upsert-signals --limit 20
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import re
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
from src.extractors.securities_report_analysis_extractor import (
    business_signals_from_securities_report,
    financial_metrics_from_securities_report,
)
from src.parsers.parser_quality import analyze_parser_quality_article
from src.parsers.securities_report_parser import SecuritiesReportParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reprocess_securities_report_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="증권사 리포트 분석 fact/signal 재생성")
    parser.add_argument("--limit", type=int, default=0, help="처리할 문서 수 제한")
    parser.add_argument(
        "--article-id",
        dest="article_ids",
        action="append",
        type=int,
        default=[],
        help="특정 raw_article_id만 선택 처리. 여러 번 지정 가능",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="SecuritiesReportParser를 다시 실행해 metadata parser 관련 필드를 갱신",
    )
    parser.add_argument(
        "--upsert-metrics",
        action="store_true",
        help="투자의견/목표가/재무 수치를 raw_article_financial_metrics에 upsert",
    )
    parser.add_argument(
        "--upsert-signals",
        action="store_true",
        help="리포트 문장/청크를 raw_article_business_signals에 upsert",
    )
    parser.add_argument(
        "--replace-metrics",
        action="store_true",
        help="처리 대상 문서의 기존 financial metrics를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--replace-signals",
        action="store_true",
        help="처리 대상 문서의 기존 business signals를 삭제한 뒤 다시 적재",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.reparse and not args.upsert_metrics and not args.upsert_signals:
        raise SystemExit("--reparse, --upsert-metrics, --upsert-signals 중 하나 이상을 지정하세요.")

    articles = _load_articles(limit=args.limit, article_ids=args.article_ids)
    log.info("증권사 리포트 로드 완료 | count=%d", len(articles))

    all_metrics: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
    for article in articles:
        parser_result = (
            _reparse_article(article) if args.reparse else _stored_parser_result(article)
        )

        if args.upsert_metrics:
            all_metrics.extend(financial_metrics_from_securities_report(article, parser_result))
        if args.upsert_signals:
            all_signals.extend(business_signals_from_securities_report(article, parser_result))

        log.info(
            "증권사 리포트 처리 완료 | id=%s period=%s metrics=%d signals=%d",
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
                source_type="securities_report",
            )
            log.info("기존 securities_report financial metrics 삭제 완료 | count=%d", deleted_count)
        all_metrics = _dedupe_securities_report_metrics(all_metrics)
        count = upsert_raw_article_financial_metrics(all_metrics)
        log.info("securities_report financial metrics upsert 완료 | count=%d", count)

    if args.upsert_signals:
        if args.replace_signals:
            deleted_count = delete_raw_article_business_signals(
                article_ids,
                source_type="securities_report",
            )
            log.info("기존 securities_report business signals 삭제 완료 | count=%d", deleted_count)
        all_signals = _dedupe_securities_report_signals(all_signals)
        count = upsert_raw_article_business_signals(all_signals)
        log.info("securities_report business signals upsert 완료 | count=%d", count)


def _load_articles(*, limit: int = 0, article_ids: list[int] | None = None) -> list[dict[str, Any]]:
    limit_sql = "LIMIT :limit" if limit > 0 else ""
    params = {"limit": limit} if limit > 0 else {}
    article_ids = [article_id for article_id in article_ids or [] if article_id > 0]
    article_filter_sql = "AND ra.id = ANY(:article_ids)" if article_ids else ""
    if article_ids:
        params["article_ids"] = article_ids
    with SessionLocal() as db:
        has_unified = _table_exists(db, "raw_article_metadata_unified")
        has_source_metadata = _table_exists(db, "raw_article_source_metadata")
        has_legacy_securities = _table_exists(db, "raw_article_metadata_securities_report")

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
                 AND md.source_type = 'securities_report'
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    elif has_legacy_securities:
        metadata_join = """
                LEFT JOIN raw_article_metadata_securities_report md
                  ON md.raw_article_id = ra.id
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    else:
        metadata_join = ""
        metadata_expr = "'{}'::jsonb"

    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
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
                    {metadata_expr} AS metadata
                FROM raw_articles ra
                {metadata_join}
                WHERE ra.source_type = 'securities_report'
                  {article_filter_sql}
                ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                {limit_sql}
            """),
            params,
        ).fetchall()

    articles = []
    for row in rows:
        article = dict(row._mapping)
        article["extra"] = article.get("metadata") or {}
        articles.append(article)

    return articles


def _table_exists(db: Any, table_name: str) -> bool:
    return bool(
        db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": f"public.{table_name}"},
        ).scalar_one_or_none()
    )


def _dedupe_securities_report_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    peer_ids = sorted({str(row.get("peer_id") or "") for row in rows if row.get("peer_id")})
    existing_keys = _load_existing_metric_keys(peer_ids)
    deduped: list[dict[str, Any]] = []
    seen = set(existing_keys)
    skipped = 0
    for row in rows:
        key = _metric_dedupe_key(row)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        deduped.append(row)

    if skipped:
        log.info("securities_report metrics 중복 스킵 | skipped=%d kept=%d", skipped, len(deduped))
    return deduped


def _dedupe_securities_report_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    peer_ids = sorted({str(row.get("peer_id") or "") for row in rows if row.get("peer_id")})
    existing_keys = _load_existing_signal_keys(peer_ids)
    deduped: list[dict[str, Any]] = []
    seen = set(existing_keys)
    skipped = 0
    for row in rows:
        key = _signal_dedupe_key(row)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        deduped.append(row)

    if skipped:
        log.info("securities_report signals 중복 스킵 | skipped=%d kept=%d", skipped, len(deduped))
    return deduped


def _load_existing_metric_keys(peer_ids: list[str]) -> set[tuple[Any, ...]]:
    if not peer_ids:
        return set()
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT peer_id, period, metric_name, business_area, value_numeric, unit
                FROM raw_article_financial_metrics
                WHERE source_type = 'securities_report'
                  AND peer_id = ANY(:peer_ids)
            """),
            {"peer_ids": peer_ids},
        ).fetchall()
    return {_metric_dedupe_key(dict(row._mapping)) for row in rows}


def _load_existing_signal_keys(peer_ids: list[str]) -> set[tuple[Any, ...]]:
    if not peer_ids:
        return set()
    with SessionLocal() as db:
        rows = db.execute(
            text("""
                SELECT peer_id, period, business_area, signal_type, evidence_text
                FROM raw_article_business_signals
                WHERE source_type = 'securities_report'
                  AND peer_id = ANY(:peer_ids)
            """),
            {"peer_ids": peer_ids},
        ).fetchall()
    return {_signal_dedupe_key(dict(row._mapping)) for row in rows}


def _metric_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("peer_id"),
        row.get("period"),
        row.get("metric_name"),
        row.get("business_area"),
        _normalize_numeric_key(row.get("value_numeric")),
        row.get("unit"),
    )


def _signal_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("peer_id"),
        row.get("period"),
        row.get("business_area"),
        row.get("signal_type"),
        _normalize_text_key(row.get("evidence_text")),
    )


def _normalize_text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _normalize_numeric_key(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _reparse_article(article: dict[str, Any]) -> dict[str, Any]:
    parser = SecuritiesReportParser()
    parsed = parser.parse_article(article)
    item, ok, reason = analyze_parser_quality_article(article)
    parser_result = item.get("parser_result") or parsed

    metadata_patch = {
        "parser_result": parser_result,
        "parser_quality_score": item.get("parser_quality_score"),
        "parser_quality_label": item.get("parser_quality_label"),
        "parser_quality_reason": item.get("parser_quality_reason"),
        "period": parser_result.get("period"),
        "report_firm": parser_result.get("report_firm"),
        "investment_opinion": parser_result.get("investment_opinion"),
        "target_price_krw": parser_result.get("target_price_krw"),
        "current_price_krw": parser_result.get("current_price_krw"),
        "topics": parser_result.get("topics"),
        "topic_signals": parser_result.get("topic_signals"),
        "securities_report_sections": parser_result.get("sections"),
        "securities_report_document_chunks": parser_result.get("document_chunks"),
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


def _stored_parser_result(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("extra") or {}
    parser_result = metadata.get("parser_result")
    if isinstance(parser_result, dict):
        return parser_result
    return SecuritiesReportParser().parse_article(article)


if __name__ == "__main__":
    main()
