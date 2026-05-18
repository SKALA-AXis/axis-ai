"""DB에 저장된 DART 문서를 재파싱하고 분석용 fact/signal 테이블을 갱신한다.

크롤링/API 호출 없이 raw_articles + raw_article_metadata_dart 안의 기존 원문과
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
        help="DartParser를 다시 실행해 raw_article_metadata_dart parser 관련 필드를 갱신",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.reparse and not args.upsert_metrics and not args.upsert_signals:
        raise SystemExit("--reparse, --upsert-metrics, --upsert-signals 중 하나 이상을 지정하세요.")

    articles = _load_dart_articles(limit=args.limit)
    log.info("DART 문서 로드 완료 | count=%d", len(articles))

    all_metrics: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
    for article in articles:
        parser_result = (
            _reparse_dart_article(article) if args.reparse else _stored_parser_result(article)
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
                    COALESCE(md.source_metadata, '{{}}'::jsonb) AS metadata
                FROM raw_articles ra
                LEFT JOIN raw_article_metadata_dart md
                  ON md.raw_article_id = ra.id
                WHERE ra.source_type = 'dart'
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


def _stored_parser_result(article: dict[str, Any]) -> dict[str, Any]:
    metadata = article.get("extra") or {}
    parser_result = metadata.get("parser_result")
    return parser_result if isinstance(parser_result, dict) else {}


if __name__ == "__main__":
    main()
