"""Materialize parsed document facts/signals into analysis tables.

The crawler stores raw documents first.  Preprocessing then parses DART, IR,
and securities reports into ``raw_article_parse_results`` / metadata.  This
module is the bridge that turns those parser results into normalized analysis
rows used by downstream cards and peer comparison.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from src.db.article_store import (
    delete_raw_article_business_signals,
    delete_raw_article_financial_metrics,
    get_articles_by_ids,
    upsert_raw_article_business_signals,
    upsert_raw_article_financial_metrics,
)
from src.extractors.dart_analysis_extractor import (
    business_signals_from_dart,
    financial_metrics_from_dart,
)
from src.extractors.securities_report_analysis_extractor import (
    business_signals_from_securities_report,
    financial_metrics_from_securities_report,
)

log = logging.getLogger(__name__)

DOCUMENT_ANALYSIS_SOURCE_TYPES = {"dart", "ir", "securities_report"}


def materialize_document_analysis(
    raw_article_ids: list[int],
    *,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Upsert metrics/signals for parsed document articles.

    Returns lightweight counts so ingestion logs can show whether realtime
    crawls produced analysis rows.
    """
    if not raw_article_ids:
        return _empty_result()

    articles = [
        _normalize_article(row)
        for row in get_articles_by_ids(raw_article_ids)
        if _source_type(row) in DOCUMENT_ANALYSIS_SOURCE_TYPES
    ]
    if not articles:
        return _empty_result()

    metrics: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    errors: list[str] = []

    for article in articles:
        source_type = _source_type(article)
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        parser_result = _parser_result(article)
        if not parser_result:
            errors.append(f"{article.get('id')}: empty parser_result")
            continue

        try:
            article_metrics, article_signals = _rows_for_article(
                article,
                parser_result,
                source_type=source_type,
            )
        except Exception as exc:
            message = f"{article.get('id')}:{source_type}:{type(exc).__name__}: {exc}"
            errors.append(message)
            log.exception("문서 분석 row 생성 실패 | %s", message)
            continue

        metrics.extend(article_metrics)
        signals.extend(article_signals)

    article_ids = [int(article["id"]) for article in articles if article.get("id")]
    if replace_existing and article_ids:
        for source_type in DOCUMENT_ANALYSIS_SOURCE_TYPES:
            source_article_ids = [
                int(article["id"])
                for article in articles
                if _source_type(article) == source_type and article.get("id")
            ]
            if not source_article_ids:
                continue
            delete_raw_article_financial_metrics(source_article_ids, source_type=source_type)
            delete_raw_article_business_signals(source_article_ids, source_type=source_type)

    metric_count = upsert_raw_article_financial_metrics(metrics)
    signal_count = upsert_raw_article_business_signals(signals)

    log.info(
        "문서 분석 materialize 완료 | articles=%d metrics=%d signals=%d sources=%s errors=%d",
        len(articles),
        metric_count,
        signal_count,
        source_counts,
        len(errors),
    )
    return {
        "analysis_document_ids": article_ids,
        "analysis_source_counts": source_counts,
        "analysis_metric_count": metric_count,
        "analysis_signal_count": signal_count,
        "analysis_errors": errors,
    }


def _rows_for_article(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    *,
    source_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source_type == "dart":
        return (
            financial_metrics_from_dart(article, parser_result),
            business_signals_from_dart(article, parser_result),
        )

    if source_type == "securities_report":
        return (
            financial_metrics_from_securities_report(article, parser_result),
            business_signals_from_securities_report(article, parser_result),
        )

    if source_type == "ir":
        # The IR extraction logic currently lives with the reprocess CLI.  Load it
        # lazily to avoid making crawler/preprocessing startup depend on it.
        ir_analysis = importlib.import_module("scripts.reprocess_ir_analysis")
        financial_record = ir_analysis._financial_record(article, parser_result)
        return (
            ir_analysis._metrics_from_parser_result(article, parser_result, financial_record),
            ir_analysis._business_signals_from_parser_result(
                article,
                parser_result,
                financial_record,
            ),
        )

    return [], []


def _normalize_article(row: dict[str, Any]) -> dict[str, Any]:
    article = dict(row)
    metadata = _dict_or_empty(article.get("metadata"))
    parser_result = _dict_or_empty(article.get("parser_result") or metadata.get("parser_result"))
    financial_record = _dict_or_empty(
        article.get("financial_record")
        or metadata.get("financial_record")
        or parser_result.get("financial_record")
    )
    article["extra"] = {
        **metadata,
        "parser_result": parser_result,
        "financial_record": financial_record,
    }
    return article


def _parser_result(article: dict[str, Any]) -> dict[str, Any]:
    extra = _dict_or_empty(article.get("extra"))
    return _dict_or_empty(extra.get("parser_result") or article.get("parser_result"))


def _source_type(article: dict[str, Any]) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _empty_result() -> dict[str, Any]:
    return {
        "analysis_document_ids": [],
        "analysis_source_counts": {},
        "analysis_metric_count": 0,
        "analysis_signal_count": 0,
        "analysis_errors": [],
    }


__all__ = ["DOCUMENT_ANALYSIS_SOURCE_TYPES", "materialize_document_analysis"]
