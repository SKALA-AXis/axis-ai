"""DB에 저장된 IR 문서를 재파싱하고 financial metric 테이블을 갱신한다.

크롤링/API 호출 없이 raw_articles + raw_article_metadata_ir 안의 기존 원문/PDF 텍스트만
다시 처리한다.

사용 예:
  uv run python run_ir_analysis_reprocess.py --reparse --upsert-metrics
  uv run python run_ir_analysis_reprocess.py --upsert-metrics --limit 20
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from sqlalchemy import text

from src.db.article_store import (
    delete_raw_article_financial_metrics,
    update_preprocess_status,
    upsert_raw_article_financial_metrics,
)
from src.db.postgres import SessionLocal
from src.parsers.ir_parser import IRParser
from src.parsers.parser_quality import analyze_parser_quality_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_ir_analysis_reprocess")

_METRIC_SPECS = {
    "revenue_total": {
        "record_key": "revenue_total_krwbn",
        "label": "매출",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "operating_profit": {
        "record_key": "operating_profit_krwbn",
        "label": "영업이익",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "net_income": {
        "record_key": "net_income_krwbn",
        "label": "순이익",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "ebitda": {
        "record_key": "ebitda_krwbn",
        "label": "EBITDA",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "backlog": {
        "record_key": "backlog_krwbn",
        "label": "수주잔고",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "capex": {
        "record_key": "capex_krwbn",
        "label": "CapEx",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "operating_margin": {
        "record_key": "operating_margin_pct",
        "label": "영업이익률",
        "unit": "%",
        "metric_scope": "company_total",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IR parser 결과 및 financial metrics 재생성")
    parser.add_argument("--limit", type=int, default=0, help="처리할 IR 문서 수 제한")
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="IRParser를 다시 실행해 raw_article_metadata_ir parser 관련 필드를 갱신",
    )
    parser.add_argument(
        "--upsert-metrics",
        action="store_true",
        help="financial_record를 raw_article_financial_metrics에 upsert",
    )
    parser.add_argument(
        "--replace-metrics",
        action="store_true",
        help="처리 대상 IR 문서의 기존 financial metrics를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--include-unknown-metrics",
        action="store_true",
        help="scope가 unknown인 IR 숫자 후보도 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.reparse and not args.upsert_metrics:
        raise SystemExit("--reparse 또는 --upsert-metrics 중 하나 이상을 지정하세요.")

    articles = _load_ir_articles(limit=args.limit)
    log.info("IR 문서 로드 완료 | count=%d", len(articles))

    all_metrics: list[dict[str, Any]] = []
    for article in articles:
        parser_result = (
            _reparse_ir_article(article) if args.reparse else _stored_parser_result(article)
        )
        financial_record = _financial_record(article, parser_result)

        if args.upsert_metrics:
            metrics = _metrics_from_parser_result(
                article,
                parser_result,
                financial_record,
                include_unknown=args.include_unknown_metrics,
            )
            all_metrics.extend(metrics)

        log.info(
            "IR 처리 완료 | id=%s period=%s candidates=%d metrics=%d",
            article["id"],
            financial_record.get("period") or parser_result.get("period"),
            len(parser_result.get("candidates") or []),
            len(all_metrics),
        )

    if args.upsert_metrics:
        if args.replace_metrics:
            deleted_count = delete_raw_article_financial_metrics(
                [int(article["id"]) for article in articles],
                source_type="ir",
            )
            log.info("기존 IR financial metrics 삭제 완료 | count=%d", deleted_count)
        count = upsert_raw_article_financial_metrics(all_metrics)
        log.info("financial metrics upsert 완료 | count=%d", count)


def _load_ir_articles(*, limit: int = 0) -> list[dict[str, Any]]:
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
                LEFT JOIN raw_article_metadata_ir md
                  ON md.raw_article_id = ra.id
                WHERE ra.source_type = 'ir'
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


def _reparse_ir_article(article: dict[str, Any]) -> dict[str, Any]:
    parser = IRParser()
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
        "ir_sections": parser_result.get("sections"),
        "ir_document_chunks": parser_result.get("document_chunks"),
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


def _financial_record(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> dict[str, Any]:
    metadata = article.get("extra") or {}
    financial_record = parser_result.get("financial_record") or metadata.get("financial_record")
    return financial_record if isinstance(financial_record, dict) else {}


def _metrics_from_financial_record(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    financial_record: dict[str, Any],
    *,
    include_unknown: bool = False,
) -> list[dict[str, Any]]:
    article_id = int(article["id"])
    period = (
        financial_record.get("period")
        or parser_result.get("period")
        or article["extra"].get("period")
    )
    peer_id = financial_record.get("peer_id") or _company_peer_id(article.get("company"))
    period_year = parser_result.get("period_year") or article["extra"].get("period_year")
    period_quarter = parser_result.get("period_quarter") or article["extra"].get("period_quarter")
    period_type = parser_result.get("period_type") or article["extra"].get("period_type")
    metric_details = financial_record.get("metric_details")
    if not isinstance(metric_details, dict):
        metric_details = {}

    metrics: list[dict[str, Any]] = []
    for metric_name, spec in _METRIC_SPECS.items():
        value = financial_record.get(spec["record_key"])
        if value is None:
            continue

        detail = metric_details.get(metric_name)
        if not isinstance(detail, dict):
            detail = {}

        metric_scope = str(detail.get("metric_scope") or "unknown")
        if metric_scope == "portfolio_company":
            continue
        if metric_scope == "unknown" and not include_unknown:
            continue

        business_area = detail.get("business_area")
        scope_key = business_area if metric_scope == "segment" and business_area else "total"
        is_percentage = spec["unit"] == "%"
        confidence = detail.get("confidence")
        if not isinstance(confidence, int | float):
            confidence = 0.85 if metric_scope == "company_total" else 0.7
        metrics.append(
            {
                "raw_article_id": article_id,
                "metric_uid": (
                    f"ir:{metric_name}:{metric_scope}:{scope_key}:{period or 'unknown'}"
                ),
                "source_type": "ir",
                "source_name": article.get("source_name"),
                "peer_id": peer_id,
                "period": period,
                "period_year": period_year,
                "period_quarter": period_quarter,
                "period_type": period_type,
                "metric_name": metric_name,
                "metric_label": spec["label"],
                "metric_scope": metric_scope,
                "business_area": business_area,
                "value_numeric": value,
                "value_krwbn": None if is_percentage else value,
                "value_krw": None if is_percentage else float(value) * 100_000_000,
                "unit": spec["unit"],
                "currency": None if is_percentage else "KRW",
                "source_page": detail.get("page") or financial_record.get("ir_page"),
                "confidence": confidence,
                "extraction_method": "ir_parser.financial_record",
                "evidence_text": detail.get("raw"),
                "payload": {
                    "financial_record": financial_record,
                    "metric_detail": detail,
                    "title": article.get("title"),
                    "url": article.get("url"),
                },
            }
        )

    return metrics


def _metrics_from_parser_result(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    financial_record: dict[str, Any],
    *,
    include_unknown: bool = False,
) -> list[dict[str, Any]]:
    candidates = parser_result.get("candidates")
    if not isinstance(candidates, list):
        return _metrics_from_financial_record(
            article,
            parser_result,
            financial_record,
            include_unknown=include_unknown,
        )

    article_id = int(article["id"])
    period = (
        financial_record.get("period")
        or parser_result.get("period")
        or article["extra"].get("period")
    )
    peer_id = financial_record.get("peer_id") or _company_peer_id(article.get("company"))
    period_year = parser_result.get("period_year") or article["extra"].get("period_year")
    period_quarter = parser_result.get("period_quarter") or article["extra"].get("period_quarter")
    period_type = parser_result.get("period_type") or article["extra"].get("period_type")

    metrics: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue

        metric_name = str(candidate.get("type") or "")
        spec = _METRIC_SPECS.get(metric_name)
        if spec is None:
            continue

        metric_scope = str(candidate.get("metric_scope") or "unknown")
        if metric_scope == "portfolio_company":
            continue
        if metric_scope == "unknown" and not include_unknown:
            continue

        value = candidate.get("value_pct") if spec["unit"] == "%" else candidate.get("value_krwbn")
        if not isinstance(value, int | float):
            continue

        business_area = candidate.get("business_area")
        entity_name = candidate.get("entity_name")
        scope_key = _metric_scope_key(metric_scope, business_area, entity_name)
        confidence = candidate.get("confidence")
        if not isinstance(confidence, int | float):
            confidence = 0.85 if metric_scope == "company_total" else 0.7

        is_percentage = spec["unit"] == "%"
        metrics.append(
            {
                "raw_article_id": article_id,
                "metric_uid": (
                    f"ir:{metric_name}:{metric_scope}:{scope_key}:"
                    f"p{candidate.get('page') or 'x'}:{idx}:{period or 'unknown'}"
                ),
                "source_type": "ir",
                "source_name": article.get("source_name"),
                "peer_id": peer_id,
                "period": period,
                "period_year": period_year,
                "period_quarter": period_quarter,
                "period_type": period_type,
                "metric_name": metric_name,
                "metric_label": spec["label"],
                "metric_scope": metric_scope,
                "business_area": business_area,
                "value_numeric": value,
                "value_krwbn": None if is_percentage else value,
                "value_krw": None if is_percentage else float(value) * 100_000_000,
                "unit": spec["unit"],
                "currency": None if is_percentage else "KRW",
                "source_page": candidate.get("page") or financial_record.get("ir_page"),
                "confidence": confidence,
                "extraction_method": "ir_parser.candidates",
                "evidence_text": candidate.get("raw"),
                "payload": {
                    "financial_record": financial_record,
                    "metric_candidate": candidate,
                    "title": article.get("title"),
                    "url": article.get("url"),
                },
            }
        )

    return metrics


def _metric_scope_key(
    metric_scope: str,
    business_area: Any,
    entity_name: Any,
) -> str:
    if metric_scope == "segment" and business_area:
        return str(business_area)
    if metric_scope == "portfolio_company" and entity_name:
        return str(entity_name)
    return "total"


def _company_peer_id(company: Any) -> str | None:
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


if __name__ == "__main__":
    main()
