# 작성일: 2026-05-18
# 작성자: 박지원
# 변경이력:
#   2026-05-18 박지원 — DART/IR 문서 재파싱 및 IR fact/signal 테이블 갱신 스크립트 작성, 이후 IR 지표 기간·시그널 보정
"""DB에 저장된 IR 문서를 재파싱하고 분석용 fact/signal 테이블을 갱신한다.

크롤링/API 호출 없이 raw_articles + raw_article_metadata_ir 안의 기존 원문/PDF 텍스트만
다시 처리한다.

사용 예:
  uv run python scripts/reprocess_ir_analysis.py --reparse --upsert-metrics
  uv run python scripts/reprocess_ir_analysis.py --upsert-signals --limit 20
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

from src.config.env_loader import load_profile

load_profile()

from src.db.article_store import (
    delete_raw_article_business_signals,
    delete_raw_article_financial_metrics,
    update_preprocess_status,
    upsert_raw_article_business_signals,
    upsert_raw_article_financial_metrics,
)
from src.db.postgres import SessionLocal
from src.parsers.ir_parser import IRParser
from src.parsers.parser_quality import analyze_parser_quality_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reprocess_ir_analysis")

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
    "gross_profit": {
        "record_key": "gross_profit_krwbn",
        "label": "매출총이익",
        "unit": "억원",
        "metric_scope": "company_total",
    },
    "gross_margin": {
        "record_key": "gross_margin_pct",
        "label": "매출총이익률",
        "unit": "%",
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
    "orders": {
        "record_key": "orders_krwbn",
        "label": "수주",
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
_COMPARISON_LABELS = {"qoq": "QoQ", "yoy": "YoY"}
for _base_metric_name, _base_spec in list(_METRIC_SPECS.items()):
    for _comparison_key, _comparison_label in _COMPARISON_LABELS.items():
        _METRIC_SPECS[f"{_base_metric_name}_{_comparison_key}"] = {
            "record_key": f"{_base_metric_name}_{_comparison_key}_pct",
            "label": f"{_base_spec['label']} {_comparison_label}",
            "unit": "%",
            "metric_scope": _base_spec["metric_scope"],
        }

_BUSINESS_AREA_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cloud",
        (
            "cloud",
            "클라우드",
            "msp",
            "csp",
            "aws",
            "azure",
            "gcp",
            "데이터센터",
            "data center",
            "gpu",
        ),
    ),
    (
        "ai_ax",
        (
            "ai",
            "ax",
            "genai",
            "생성형",
            "llm",
            "agent",
            "fabrix",
            "brity",
            "erp ai",
            "scm",
            "자동화",
        ),
    ),
    ("logistics", ("logistics", "물류", "cello", "scl")),
    ("smart_factory", ("smart factory", "스마트팩토리", "mes", "factory", "제조")),
    ("vehicle_sw", ("vehicle", "차량", "sdv", "내비게이션", "navigation")),
    ("enterprise_it", ("enterprise", "erp", "ito", "si", "그룹사", "it서비스")),
    ("robotics", ("robot", "로봇", "automation")),
)
_SECTION_AREA_MAP = {
    "cloud": "cloud",
    "ai": "ai_ax",
    "digital_transformation": "enterprise_it",
    "orders_pipeline": "company_total",
    "financial": "company_total",
    "summary": "company_total",
    "outlook": "company_total",
    "shareholder": "company_total",
}
_SIGNAL_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("orders_pipeline", ("수주", "backlog", "계약", "잔고")),
    ("investment", ("투자", "capex", "설비", "데이터센터", "gpu", "구축")),
    ("growth", ("성장", "확대", "증가", "개선", "상승", "호조", "profitability")),
    ("strategy", ("전략", "추진", "고도화", "출시", "제휴", "협력", "mou", "개편")),
    ("efficiency", ("효율", "최적화", "자동화", "비용 절감", "생산성")),
    ("risk", ("리스크", "위험", "하락", "감소", "둔화", "부진", "비용 증가")),
    (
        "business_update",
        ("프로젝트", "서비스", "사업", "고객", "운영", "매출 인식", "전망", "기대"),
    ),
)
_SIGNAL_REQUIRED_CONTEXT = {
    "orders_pipeline": ("수주", "backlog", "계약", "잔고"),
    "investment": ("투자", "capex", "설비", "데이터센터", "gpu", "구축"),
    "growth": ("성장", "확대", "증가", "개선", "상승", "호조"),
    "strategy": ("전략", "추진", "고도화", "출시", "제휴", "협력", "mou", "개편"),
    "efficiency": ("효율", "최적화", "자동화", "비용 절감", "생산성"),
    "risk": ("리스크", "위험", "하락", "감소", "둔화", "부진", "비용 증가"),
    "business_update": ("프로젝트", "서비스", "사업", "고객", "운영", "매출 인식", "전망", "기대"),
}
_SIGNAL_TYPE_PRIORITY = {
    "risk": 0,
    "orders_pipeline": 1,
    "growth": 2,
    "strategy": 3,
    "investment": 4,
    "efficiency": 5,
    "business_update": 6,
}
_IR_SIGNAL_EXCLUDE_TERMS = (
    "appendix",
    "주요비상장자회사",
    "주요비상장사합산",
    "비상장자회사",
    "자회사분기별실적",
    "rebalancing",
    "중간배당",
    "배당금",
    "sk이노베이션",
    "sk innovation",
    "forward-looking",
    "본 자료는",
    "무단 복제",
)
_SK_AFFILIATE_EXCLUDE_TERMS = (
    "sk이노베이션",
    "sk innovation",
    "sk텔레콤",
    "sk telecom",
    "skt",
    "에이닷",
    "sk하이닉스",
    "sk hynix",
    "sk스퀘어",
    "sk square",
    "sk바이오팜",
    "sk biopharmaceuticals",
    "sk e&s",
    "sk온",
    "sk on",
    "sk에코플랜트",
    "sk ecoplant",
    "sk팜테코",
    "pharmteco",
)
_NEGATIVE_TERMS = ("하락", "감소", "둔화", "부진", "리스크", "위험", "비용 증가")
_POSITIVE_TERMS = ("성장", "확대", "증가", "개선", "강화", "고도화", "수주", "계약")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IR parser 결과 및 financial metrics 재생성")
    parser.add_argument("--limit", type=int, default=0, help="처리할 IR 문서 수 제한")
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
        help="IRParser를 다시 실행해 raw_article_metadata_ir parser 관련 필드를 갱신",
    )
    parser.add_argument(
        "--upsert-metrics",
        action="store_true",
        help="financial_record를 raw_article_financial_metrics에 upsert",
    )
    parser.add_argument(
        "--upsert-signals",
        action="store_true",
        help="IR document chunks를 raw_article_business_signals에 upsert",
    )
    parser.add_argument(
        "--replace-metrics",
        action="store_true",
        help="처리 대상 IR 문서의 기존 financial metrics를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--replace-signals",
        action="store_true",
        help="처리 대상 IR 문서의 기존 business signals를 삭제한 뒤 다시 적재",
    )
    parser.add_argument(
        "--include-unknown-metrics",
        action="store_true",
        help="scope가 unknown인 IR 숫자 후보도 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.reparse and not args.upsert_metrics and not args.upsert_signals:
        raise SystemExit("--reparse, --upsert-metrics, --upsert-signals 중 하나 이상을 지정하세요.")

    articles = _load_ir_articles(limit=args.limit, article_ids=args.article_ids)
    log.info("IR 문서 로드 완료 | count=%d", len(articles))

    all_metrics: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
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

        if args.upsert_signals:
            signals = _business_signals_from_parser_result(
                article,
                parser_result,
                financial_record,
            )
            all_signals.extend(signals)

        log.info(
            "IR 처리 완료 | id=%s period=%s candidates=%d metrics=%d signals=%d",
            article["id"],
            financial_record.get("period") or parser_result.get("period"),
            len(parser_result.get("candidates") or []),
            len(all_metrics),
            len(all_signals),
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

    if args.upsert_signals:
        if args.replace_signals:
            deleted_count = delete_raw_article_business_signals(
                [int(article["id"]) for article in articles],
                source_type="ir",
            )
            log.info("기존 IR business signals 삭제 완료 | count=%d", deleted_count)
        count = upsert_raw_article_business_signals(all_signals)
        log.info("business signals upsert 완료 | count=%d", count)


def _load_ir_articles(
    *,
    limit: int = 0,
    article_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    limit_sql = "LIMIT :limit" if limit > 0 else ""
    article_ids = [article_id for article_id in article_ids or [] if article_id > 0]
    article_filter_sql = "AND ra.id = ANY(:article_ids)" if article_ids else ""
    params: dict[str, Any] = {}
    if limit > 0:
        params["limit"] = limit
    if article_ids:
        params["article_ids"] = article_ids
    with SessionLocal() as db:
        rows = db.execute(
            text(_ir_article_select_sql(limit_sql, article_filter_sql)),
            params,
        ).fetchall()

    articles = []
    for row in rows:
        article = dict(row._mapping)
        article["extra"] = article.get("metadata") or {}
        articles.append(article)

    return articles


def _ir_article_select_sql(limit_sql: str, article_filter_sql: str) -> str:
    """Return an IR article query for either legacy or unified metadata schemas."""
    with SessionLocal() as db:
        has_unified = _table_exists(db, "raw_article_metadata_unified")
        has_source_metadata = _table_exists(db, "raw_article_source_metadata")
        has_legacy_ir = _table_exists(db, "raw_article_metadata_ir")

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
                 AND md.source_type = 'ir'
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    elif has_legacy_ir:
        metadata_join = """
                LEFT JOIN raw_article_metadata_ir md
                  ON md.raw_article_id = ra.id
            """
        metadata_expr = "COALESCE(md.source_metadata, '{}'::jsonb)"
    else:
        metadata_join = ""
        metadata_expr = "COALESCE(ra.metadata, '{}'::jsonb)"

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
                    {metadata_expr} AS metadata
                FROM raw_articles ra
                {metadata_join}
                WHERE ra.source_type = 'ir'
                  {article_filter_sql}
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
        if not business_area and metric_scope == "company_total":
            business_area = "company_total"
        metric_scope, business_area = _normalize_self_business_area(
            peer_id=peer_id,
            metric_scope=metric_scope,
            business_area=business_area,
        )
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
                "evidence_text": _metric_evidence_text(detail, fallback=detail.get("raw")),
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

    selected_candidates = _select_ir_metric_candidates(
        candidates,
        report_period=period,
        peer_id=peer_id,
    )

    metrics: list[dict[str, Any]] = []
    for idx, candidate in enumerate(selected_candidates, start=1):
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

        has_candidate_period = "period" in candidate
        candidate_period = candidate.get("period") or period
        candidate_period_year = (
            candidate.get("period_year") if has_candidate_period else period_year
        )
        candidate_period_quarter = (
            candidate.get("period_quarter") if has_candidate_period else period_quarter
        )
        candidate_period_type = (
            candidate.get("period_type") if has_candidate_period else period_type
        )
        business_area = candidate.get("business_area")
        if not business_area and metric_scope == "company_total":
            business_area = "company_total"
        metric_scope, business_area = _normalize_self_business_area(
            peer_id=peer_id,
            metric_scope=metric_scope,
            business_area=business_area,
        )
        entity_name = candidate.get("entity_name")
        scope_key = _metric_scope_key(metric_scope, business_area, entity_name)
        confidence = candidate.get("confidence")
        if not isinstance(confidence, int | float):
            confidence = 0.85 if metric_scope == "company_total" else 0.7

        is_percentage = spec["unit"] == "%"
        extraction_method = (
            "ir_parser.table_matrix"
            if candidate.get("source") == "ir_table_matrix"
            else "ir_parser.candidates"
        )
        metrics.append(
            {
                "raw_article_id": article_id,
                "metric_uid": (
                    f"ir:{metric_name}:{metric_scope}:{scope_key}:"
                    f"p{candidate.get('page') or 'x'}:{idx}:{candidate_period or 'unknown'}"
                ),
                "source_type": "ir",
                "source_name": article.get("source_name"),
                "peer_id": peer_id,
                "period": candidate_period,
                "period_year": candidate_period_year,
                "period_quarter": candidate_period_quarter,
                "period_type": candidate_period_type,
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
                "source_table_uid": candidate.get("source_table_uid"),
                "source_chunk_uid": candidate.get("source_chunk_uid"),
                "confidence": confidence,
                "extraction_method": extraction_method,
                "evidence_text": _metric_evidence_text(
                    candidate,
                    fallback=candidate.get("evidence_text") or candidate.get("raw"),
                ),
                "payload": {
                    "financial_record": financial_record,
                    "metric_candidate": candidate,
                    "title": article.get("title"),
                    "url": article.get("url"),
                },
            }
        )

    return metrics


def _select_ir_metric_candidates(
    candidates: list[Any],
    *,
    report_period: Any,
    peer_id: str | None = None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for order, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        metric_name = str(candidate.get("type") or "")
        if metric_name not in _METRIC_SPECS:
            continue
        candidate_period, period_patch = _candidate_period_for_report(candidate, report_period)
        if not _candidate_period_allowed_for_report(candidate_period, report_period):
            continue
        metric_scope = str(candidate.get("metric_scope") or "unknown")
        if metric_scope == "portfolio_company" or metric_scope == "unknown":
            continue
        business_area = candidate.get("business_area")
        if not business_area and metric_scope == "company_total":
            business_area = "company_total"
        metric_scope, business_area = _normalize_self_business_area(
            peer_id=peer_id,
            metric_scope=metric_scope,
            business_area=business_area,
        )
        scope_key = _metric_scope_key(metric_scope, business_area, candidate.get("entity_name"))
        key = (metric_name, metric_scope, scope_key, str(candidate_period or "unknown"))
        enriched = {
            **candidate,
            **period_patch,
            "_candidate_order": order,
            "period_matches_report": bool(
                report_period and candidate_period and str(candidate_period) == str(report_period)
            ),
        }
        current = selected.get(key)
        if current is None or _metric_candidate_rank(enriched) > _metric_candidate_rank(current):
            selected[key] = enriched
    return list(selected.values())


def _metric_candidate_rank(candidate: dict[str, Any]) -> tuple[int, int, int, int, float, int, int]:
    source_priority = 3 if candidate.get("source") == "ir_table_matrix" else 2
    report_period_score = 1 if candidate.get("period_matches_report") else 0
    period_score = 0 if candidate.get("period_inferred_from_report") else 1
    has_evidence = 1 if candidate.get("evidence_text") or candidate.get("raw") else 0
    confidence = candidate.get("confidence")
    confidence_score = float(confidence) if isinstance(confidence, int | float) else 0.0
    page = candidate.get("page")
    page_score = -int(page) if isinstance(page, int) else -9999
    order_score = -int(candidate.get("_candidate_order") or 0)
    return (
        report_period_score,
        source_priority,
        period_score,
        has_evidence,
        confidence_score,
        page_score,
        order_score,
    )


def _candidate_period_for_report(
    candidate: dict[str, Any],
    report_period: Any,
) -> tuple[Any, dict[str, Any]]:
    return candidate.get("period") or report_period, {}


def _candidate_period_allowed_for_report(candidate_period: Any, report_period: Any) -> bool:
    if not report_period or not candidate_period:
        return True
    candidate_period_value = str(candidate_period)
    report_period_value = str(report_period)
    if candidate_period_value == report_period_value:
        return True

    annual_match = re.fullmatch(r"(20\d{2})", report_period_value)
    candidate_year, _candidate_quarter = _quarter_period_parts(candidate_period_value)
    return bool(annual_match and candidate_year == int(annual_match.group(1)))


def _quarter_period_parts(period: Any) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"(20\d{2})Q([1-4])", str(period or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _metric_evidence_text(metric_source: dict[str, Any], *, fallback: Any = None) -> str | None:
    parts = [
        str(fallback or "").strip(),
        (
            f"분류 근거: {metric_source.get('classification_reason')}"
            if metric_source.get("classification_reason")
            else ""
        ),
        (
            f"판단 문맥: {metric_source.get('context_evidence')}"
            if metric_source.get("context_evidence")
            else ""
        ),
    ]
    value = " | ".join(part for part in parts if part)
    return value[:2000] if value else None


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


def _business_signals_from_parser_result(
    article: dict[str, Any],
    parser_result: dict[str, Any],
    financial_record: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks = _document_chunks(article, parser_result)
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

    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if not chunks:
        return signals
    page_contexts = _page_contexts_from_chunks(chunks)

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        text_value = str(chunk.get("text") or "").strip()
        if len(text_value) < 30:
            continue
        page_context = page_contexts.get(chunk.get("page")) or text_value
        if not _is_ir_chunk_target_for_peer(
            peer_id=peer_id,
            text_value=text_value,
            page_context=page_context,
        ):
            continue
        if _is_ir_chunk_excluded_for_peer(peer_id, text_value):
            continue

        source_chunk_uid = str(chunk.get("chunk_id") or chunk.get("chunk_index") or "")
        for sentence_index, evidence_text, business_area, signal_type in _signals_from_chunk(
            chunk,
            text_value,
            peer_id=peer_id,
            page_context=page_context,
        ):
            _metric_scope, business_area = _normalize_self_business_area(
                peer_id=peer_id,
                metric_scope="segment",
                business_area=business_area,
            )
            business_area = _business_area_from_evidence_override(
                peer_id=peer_id,
                business_area=business_area,
                evidence_text=evidence_text,
            )
            dedupe_key = (business_area, evidence_text[:160])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            signal_uid = (
                f"ir:{business_area}:{signal_type}:"
                f"p{chunk.get('page') or 'x'}:{chunk.get('chunk_index') or len(signals) + 1}:"
                f"s{sentence_index}"
            )
            signals.append(
                {
                    "raw_article_id": article_id,
                    "signal_uid": signal_uid,
                    "source_type": "ir",
                    "source_name": article.get("source_name"),
                    "peer_id": peer_id,
                    "period": period,
                    "period_year": period_year,
                    "period_quarter": period_quarter,
                    "period_type": period_type,
                    "business_area": business_area,
                    "signal_type": signal_type,
                    "sentiment": _sentiment_from_text(evidence_text),
                    "summary": _summary_from_evidence(evidence_text),
                    "evidence_text": evidence_text,
                    "source_page": chunk.get("page"),
                    "source_chunk_uid": source_chunk_uid or None,
                    "confidence": _signal_confidence(business_area, signal_type, evidence_text),
                    "extraction_method": "ir_parser.document_chunks.rule_based.v2",
                    "payload": {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "sentence_index": sentence_index,
                        "matched_business_terms": _matched_terms_for_business_area(
                            business_area,
                            evidence_text,
                        ),
                        "matched_signal_terms": _matched_terms_for_signal_type(
                            signal_type,
                            evidence_text,
                        ),
                        "chunk": {
                            "chunk_id": chunk.get("chunk_id"),
                            "chunk_index": chunk.get("chunk_index"),
                            "section_key": chunk.get("section_key"),
                            "section_title": chunk.get("section_title"),
                            "topics": chunk.get("topics"),
                        },
                    },
                }
            )

    return signals


def _document_chunks(
    article: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[Any]:
    chunks = parser_result.get("document_chunks")
    if isinstance(chunks, list):
        return chunks

    metadata = article.get("extra") or {}
    chunks = metadata.get("ir_document_chunks")
    return chunks if isinstance(chunks, list) else []


def _signals_from_chunk(
    chunk: dict[str, Any],
    text_value: str,
    *,
    peer_id: str | None,
    page_context: str,
) -> list[tuple[int, str, str, str]]:
    section_area = _business_area_from_section(chunk)
    signals: list[tuple[int, str, str, str]] = []
    normalized_text_value = _normalize_ocr_signal_text(text_value)
    normalized_page_context = _normalize_ocr_signal_text(page_context)
    sentences = _sentences(normalized_text_value)
    for sentence_index, sentence in enumerate(sentences, start=1):
        if len(sentence) < 30 or _is_low_value_signal_sentence(sentence):
            continue
        if _is_ir_sentence_excluded_for_peer(
            peer_id, sentence, page_context=normalized_page_context
        ):
            continue
        if not _looks_like_narrative_signal_sentence(sentence):
            continue

        context_text = _signal_context_text(sentences, sentence_index - 1)
        if _is_ir_sentence_excluded_for_peer(
            peer_id, context_text, page_context=normalized_page_context
        ):
            continue
        signal_types = _detect_signal_types(sentence)
        if not signal_types:
            signal_types = _detect_signal_types(context_text)
        if not signal_types:
            continue
        business_area = _resolve_business_area(
            peer_id=peer_id,
            sentence=sentence,
            context_text=context_text,
            page_context=normalized_page_context,
            section_area=section_area,
            chunk_text=normalized_text_value,
        )
        evidence_text = _signal_evidence_text(sentence=sentence, context_text=context_text)
        if _is_degraded_ocr_signal_text(evidence_text):
            continue

        eligible_signal_types = [
            signal_type
            for signal_type in signal_types
            if signal_type == "business_update"
            or _has_required_signal_context(context_text, signal_type)
        ]
        if not eligible_signal_types:
            continue
        primary_signal_type = _primary_signal_type(eligible_signal_types)
        signals.append((sentence_index, evidence_text[:800], business_area, primary_signal_type))
    return signals


def _is_ir_chunk_excluded_for_peer(peer_id: str | None, text_value: str) -> bool:
    if peer_id != "sk_ax":
        return False

    lowered = text_value.lower()
    has_sk_ax_context = _has_sk_ax_context(lowered)
    if has_sk_ax_context:
        return False

    return _is_low_value_signal_sentence(text_value)


def _is_ir_chunk_target_for_peer(
    peer_id: str | None,
    text_value: str,
    *,
    page_context: str,
) -> bool:
    if peer_id != "sk_ax":
        return True

    text_lowered = text_value.lower()
    page_lowered = page_context.lower()
    if _has_strong_sk_ax_context(text_lowered) or _has_strong_sk_ax_context(page_lowered):
        return True

    return False


def _is_ir_sentence_excluded_for_peer(
    peer_id: str | None,
    text_value: str,
    *,
    page_context: str,
) -> bool:
    if peer_id != "sk_ax":
        return False

    lowered = text_value.lower()
    if _has_sk_ax_context(lowered):
        return False
    if _has_sk_ax_context(page_context.lower()):
        return False

    return any(term in lowered for term in _SK_AFFILIATE_EXCLUDE_TERMS)


def _business_area_from_chunk(chunk: dict[str, Any], text_value: str) -> str | None:
    detected = _detect_business_area(text_value)
    if detected:
        return detected

    return _business_area_from_section(chunk)


def _business_area_from_section(chunk: dict[str, Any]) -> str | None:
    section_key = str(chunk.get("section_key") or "")
    mapped = _SECTION_AREA_MAP.get(section_key)
    if mapped:
        return mapped

    return None


def _resolve_business_area(
    *,
    peer_id: str | None,
    sentence: str,
    context_text: str,
    page_context: str,
    section_area: str | None,
    chunk_text: str,
) -> str:
    page_preferred_area = _page_preferred_business_area(peer_id=peer_id, page_context=page_context)
    if page_preferred_area:
        if page_preferred_area == "Enterprise IT":
            return "company_total"

    return (
        _detect_business_area(sentence)
        or _detect_business_area(context_text)
        or _section_area_fallback_for_sentence(
            section_area=section_area,
            sentence=sentence,
            chunk_text=chunk_text,
        )
        or "company_total"
    )


def _page_preferred_business_area(
    *,
    peer_id: str | None,
    page_context: str,
) -> str | None:
    lowered = page_context.lower()
    if peer_id == "sk_ax":
        if any(
            term in lowered
            for term in (
                "it서비스부문(sk ax)",
                "it서비스부문",
                "it서비스ebitda",
                "its 사업",
                "it 예산",
                "enterprise it",
            )
        ):
            return "Enterprise IT"
    return None


def _section_area_fallback_for_sentence(
    *,
    section_area: str | None,
    sentence: str,
    chunk_text: str,
) -> str | None:
    if not section_area:
        return None
    if section_area != "cloud":
        return section_area

    sentence_lowered = sentence.lower()
    chunk_lowered = chunk_text.lower()
    if _has_explicit_cloud_context(sentence_lowered):
        return "cloud"

    # `section_key=cloud`만으로는 부족하다. 청크 전체에 클라우드 문맥이 없으면
    # 반도체/Hi-tech/IT서비스 문장까지 cloud로 흘러가는 오분류를 막는다.
    if _has_explicit_cloud_context(chunk_lowered):
        return "cloud"

    return "other"


def _detect_business_area(text_value: str) -> str | None:
    lowered = text_value.lower()
    for business_area, terms in _BUSINESS_AREA_RULES:
        if any(term.lower() in lowered for term in terms):
            return business_area
    return None


def _detect_business_areas(text_value: str) -> set[str]:
    lowered = text_value.lower()
    return {
        business_area
        for business_area, terms in _BUSINESS_AREA_RULES
        if any(term.lower() in lowered for term in terms)
    }


def _has_explicit_cloud_context(lowered_text: str) -> bool:
    return any(
        term in lowered_text
        for term in (
            "cloud",
            "클라우드",
            "msp",
            "csp",
            "aws",
            "azure",
            "gcp",
            "데이터센터",
            "data center",
            "gpu",
        )
    )


def _has_sk_ax_context(lowered_text: str) -> bool:
    return any(
        term in lowered_text
        for term in (
            "sk ax",
            "sk c&c",
            "c&c",
            "씨앤씨",
            "it서비스",
            "it 서비스",
            "enterprise it",
            "digital transformation",
            "ai transformation",
        )
    )


def _has_strong_sk_ax_context(lowered_text: str) -> bool:
    return any(
        term in lowered_text
        for term in (
            "sk ax",
            "sk c&c",
            "it서비스부문(sk ax)",
            "it서비스부문",
            "it서비스ebitda",
            "it서비스",
            "it 서비스",
            "its 사업",
            "ai transformation",
            "dt 기반",
        )
    )


def _signal_type_from_text(text_value: str) -> str | None:
    signal_types = _detect_signal_types(text_value)
    return signal_types[0] if signal_types else None


def _detect_signal_types(text_value: str) -> list[str]:
    lowered = text_value.lower()
    scores: dict[str, int] = {}
    for signal_type, terms in _SIGNAL_TYPE_RULES:
        score = sum(1 for term in terms if term.lower() in lowered)
        if score > 0:
            scores[signal_type] = score

    if not scores:
        return []

    if "business_update" in scores and len(scores) > 1:
        scores.pop("business_update", None)
    if "strategy" in scores and "efficiency" in scores:
        scores.pop("efficiency", None)

    positive_score = sum(1 for term in _POSITIVE_TERMS if term in lowered)
    negative_score = sum(1 for term in _NEGATIVE_TERMS if term in lowered)
    if positive_score and negative_score:
        if positive_score >= negative_score:
            scores.pop("risk", None)
        if negative_score >= positive_score:
            scores.pop("growth", None)

    if not scores:
        return []

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _signal_type_priority(item[0])),
    )
    top_score = ranked[0][1]
    selected = [
        signal_type for signal_type, score in ranked if score == top_score or len(ranked) == 1
    ][:2]
    if (
        "orders_pipeline" in scores
        and any(term in lowered for term in ("수주", "계약", "backlog", "잔고"))
        and "orders_pipeline" not in selected
    ):
        selected.append("orders_pipeline")
    if (
        "strategy" in scores
        and any(term in lowered for term in ("개편", "전환", "추진", "고도화", "전략"))
        and "strategy" not in selected
    ):
        selected.append("strategy")
    selected = selected[:2]
    return selected


def _primary_signal_type(signal_types: list[str]) -> str:
    ranked = sorted(signal_types, key=_signal_type_priority)
    return ranked[0]


def _signal_type_priority(signal_type: str) -> int:
    return _SIGNAL_TYPE_PRIORITY.get(signal_type, 99)


def _is_low_value_signal_sentence(text_value: str) -> bool:
    lowered = text_value.lower()
    return any(term.lower() in lowered for term in _IR_SIGNAL_EXCLUDE_TERMS) or (
        _is_degraded_ocr_signal_text(text_value)
        and not _has_high_value_ocr_signal_terms(text_value)
    )


def _looks_like_narrative_signal_sentence(text_value: str) -> bool:
    value = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not value:
        return False
    if _looks_like_table_like_signal_text(value):
        return False

    hangul_tokens = re.findall(r"[가-힣A-Za-z]{2,}", value)
    if len(hangul_tokens) < 3:
        return False

    if re.search(r"[.!?。]\s*$", value):
        return True

    return bool(
        re.search(
            r"(습니다|했다|한다|된다|있다|없다|보인다|전망이다|예상된다|이어지고 있다|"
            r"개선됐다|감소했다|증가했다|확대됐다|축소됐다|"
            r"증가|감소|개선|확대|축소|지속|견조|호조|둔화|부진|약세|강화|고도화|"
            r"진행중|진행 중|전환|유지|확보|상승|하락|집중|축소)",
            value,
        )
    )


def _looks_like_table_like_signal_text(text_value: str) -> bool:
    value = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not value:
        return False

    lowered = value.lower()
    numeric_tokens = re.findall(r"[△]?\d[\d,./]*(?:%|억원|조원|bn|mn)?", value)
    digit_count = sum(char.isdigit() for char in value)

    if value.startswith("[단위") or value.startswith("(단위"):
        return True
    if "ebitda margin" in lowered or "op margin" in lowered:
        return True
    if len(numeric_tokens) >= 4 or digit_count >= 14:
        return True
    table_metric_pattern = (
        r"(매출|영업이익|ebitda|이익률)\s+[△]?\d[\d,./]*(?:%|억원|조원|bn|mn)?"
        r"(?:\s+[△]?\d[\d,./]*(?:%|억원|조원|bn|mn)?){1,}"
    )
    if re.search(table_metric_pattern, value, re.IGNORECASE):
        return True

    return False


def _has_required_signal_context(text_value: str, signal_type: str) -> bool:
    lowered = text_value.lower()
    required_terms = _SIGNAL_REQUIRED_CONTEXT.get(signal_type, ())
    return any(term.lower() in lowered for term in required_terms)


def _evidence_text(
    text_value: str,
    business_area: str,
    signal_type: str,
) -> str | None:
    sentences = _sentences(text_value)
    terms = _terms_for_evidence(business_area, signal_type)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term.lower() in lowered for term in terms):
            return sentence[:800]

    return sentences[0][:800] if sentences else None


def _terms_for_evidence(business_area: str, signal_type: str) -> tuple[str, ...]:
    business_terms = next(
        (terms for area, terms in _BUSINESS_AREA_RULES if area == business_area),
        (),
    )
    signal_terms = next(
        (terms for kind, terms in _SIGNAL_TYPE_RULES if kind == signal_type),
        (),
    )
    return (*business_terms, *signal_terms)


def _matched_terms_for_business_area(business_area: str, text_value: str) -> list[str]:
    terms = next((terms for area, terms in _BUSINESS_AREA_RULES if area == business_area), ())
    return _matched_terms(terms, text_value)


def _matched_terms_for_signal_type(signal_type: str, text_value: str) -> list[str]:
    terms = _SIGNAL_REQUIRED_CONTEXT.get(signal_type) or next(
        (terms for kind, terms in _SIGNAL_TYPE_RULES if kind == signal_type),
        (),
    )
    return _matched_terms(terms, text_value)


def _matched_terms(terms: tuple[str, ...], text_value: str) -> list[str]:
    lowered = text_value.lower()
    return [term for term in terms if term.lower() in lowered]


def _sentences(text_value: str) -> list[str]:
    lines = [line.strip(" -•\t") for line in str(text_value or "").splitlines() if line.strip()]
    if not lines:
        value = re.sub(r"\s+", " ", text_value).strip()
        return [value] if value else []

    pieces: list[str] = []
    for line in lines:
        split_line = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+", line)
        split_line = [piece.strip(" -•\t") for piece in split_line if piece.strip()]
        if not split_line:
            continue
        pieces.extend(split_line)

    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        normalized = re.sub(r"\s+", " ", piece).strip()
        if not normalized:
            continue
        if not buffer:
            buffer = normalized
            continue
        if (
            len(buffer) < 48
            and len(normalized) < 72
            and not re.search(r"[.!?。]\s*$", buffer)
            and not _looks_like_table_like_signal_text(normalized)
        ):
            buffer = f"{buffer} {normalized}".strip()
            continue
        merged.append(buffer)
        buffer = normalized

    if buffer:
        merged.append(buffer)

    filtered = [
        piece for piece in merged if len(piece) >= 24 and not _is_numeric_heavy_signal_text(piece)
    ]
    return filtered or merged


def _page_contexts_from_chunks(chunks: list[Any]) -> dict[Any, str]:
    page_texts: dict[Any, list[str]] = {}
    seen_by_page: dict[Any, set[str]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        page = chunk.get("page")
        text_value = str(chunk.get("text") or "").strip()
        if not text_value:
            continue
        seen = seen_by_page.setdefault(page, set())
        if text_value in seen:
            continue
        seen.add(text_value)
        page_texts.setdefault(page, []).append(text_value)

    page_contexts: dict[Any, str] = {}
    for page, texts in page_texts.items():
        sentences: list[str] = []
        for text_value in texts:
            sentences.extend(
                sentence
                for sentence in _sentences(text_value)
                if _looks_like_narrative_signal_sentence(sentence)
                and not _is_low_value_signal_sentence(sentence)
            )
        page_contexts[page] = " ".join(sentences)[:2400].strip()
    return page_contexts


def _signal_context_text(sentences: list[str], index: int) -> str:
    start = max(0, index - 1)
    end = min(len(sentences), index + 2)
    context_sentences = [
        sentence
        for sentence in sentences[start:end]
        if _looks_like_narrative_signal_sentence(sentence)
        and not _is_low_value_signal_sentence(sentence)
    ]
    return " ".join(context_sentences).strip()


def _signal_evidence_text(*, sentence: str, context_text: str) -> str:
    if _looks_like_table_like_signal_text(sentence) and len(context_text) > len(sentence):
        return context_text
    if len(sentence) < 48 and len(context_text) <= 800:
        if _detect_signal_types(sentence):
            return sentence
        return context_text
    return sentence


def _sentiment_from_text(text_value: str) -> str:
    lowered = text_value.lower()
    negative_hits = sum(1 for term in _NEGATIVE_TERMS if term in lowered)
    positive_hits = sum(1 for term in _POSITIVE_TERMS if term in lowered)
    if negative_hits and positive_hits:
        return "neutral"
    if negative_hits:
        return "negative"
    if positive_hits:
        return "positive"
    return "neutral"


def _summary_from_evidence(evidence_text: str) -> str:
    value = re.sub(r"\s+", " ", _normalize_ocr_signal_text(evidence_text)).strip()
    if _is_numeric_heavy_signal_text(value):
        cleaned = _clean_numeric_heavy_summary(value)
        if len(cleaned) >= 24:
            value = cleaned
    if len(value) <= 160:
        return value
    return f"{value[:157]}..."


def _is_numeric_heavy_signal_text(value: str) -> bool:
    numeric_tokens = re.findall(r"[△]?\d[\d,./]*(?:%|억원|조원|bn|mn)?", value)
    digit_count = sum(char.isdigit() for char in value)
    return len(numeric_tokens) >= 5 or digit_count >= 18


def _clean_numeric_heavy_summary(value: str) -> str:
    cleaned = re.sub(r"\[[^\]]*단위[^\]]*\]", " ", value)
    cleaned = re.sub(r"\([^)]*단위[^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[△]?\d[\d,./]*(?:%|억원|조원|bn|mn)?", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _signal_confidence(
    business_area: str,
    signal_type: str,
    text_value: str,
) -> float:
    text_value = _normalize_ocr_signal_text(text_value)
    detected_area = _detect_business_area(text_value)
    detected_type = _signal_type_from_text(text_value)
    penalty = 0.1 if _has_ocr_marker_or_noise(text_value) else 0.0
    if detected_area == business_area and detected_type == signal_type:
        return max(0.58, 0.78 - penalty)
    if business_area == "company_total":
        return max(0.55, 0.68 - penalty)
    return max(0.55, 0.72 - penalty)


def _normalize_ocr_signal_text(text_value: str) -> str:
    value = str(text_value or "")
    value = re.sub(r"\[PAGE\s+\d+\]", " ", value, flags=re.IGNORECASE)
    value = value.replace("[OCR]", " ")
    replacements = (
        (r"\bAl\b", "AI"),
        (r"\bAl(?=Ops\b)", "AI"),
        (r"\bAl(?=\s*(?:Native|Full|Agent|Orchestrator|Data|Machine)\b)", "AI"),
        (r"\bAl(?=\s*(?:인프라|플랫폼|서비스|솔루션|클라우드|데이터|컴퓨팅))", "AI"),
        (r"\b시\s*/\s*클라우드", "AI/클라우드"),
        (r"\b시\s*컴퓨팅", "AI 컴퓨팅"),
        (r"\b시\s*데이터", "AI 데이터"),
        (r"\bAx\b", "AX"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r"AI\s*-\s*AI", "AI", value)
    value = re.sub(r"AX\s*-\s*AI", "AX-AI", value)
    return value


def _has_ocr_marker_or_noise(text_value: str) -> bool:
    return "[OCR]" in text_value or _ocr_noise_ratio(text_value) >= 0.18


def _is_degraded_ocr_signal_text(text_value: str) -> bool:
    value = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not value:
        return True
    semantic_chars = len(re.findall(r"[0-9A-Za-z가-힣]", value))
    if semantic_chars < 24:
        return True
    if _ocr_noise_ratio(value) >= 0.28:
        return True
    noisy_tokens = len(re.findall(r"[{}<>|\\^~]{1,}|[A-Za-z]{1,}\d{2,}[A-Za-z]*", value))
    semantic_tokens = len(re.findall(r"[A-Za-z가-힣]{2,}", value))
    return noisy_tokens >= 3 and noisy_tokens > semantic_tokens


def _ocr_noise_ratio(text_value: str) -> float:
    value = str(text_value or "")
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return 1.0
    noisy = re.sub(r"[0-9A-Za-z가-힣%&/.,:;+\-()·ㆍ\[\]]", "", compact)
    return len(noisy) / max(len(compact), 1)


def _has_high_value_ocr_signal_terms(text_value: str) -> bool:
    lowered = _normalize_ocr_signal_text(text_value).lower()
    return any(
        term in lowered
        for term in (
            "full stack",
            "inorganic",
            "gpuaas",
            "npuaas",
            "vertical ai",
            "ai native",
            "ai 인프라",
            "ai 플랫폼",
            "ai 컴퓨팅",
            "ax-ai",
            "ai/클라우드",
            "aiops",
            "mlops",
        )
    )


def _company_peer_id(company: Any) -> str | None:
    if isinstance(company, list) and company:
        return str(company[0])
    if isinstance(company, str):
        return company
    return None


def _normalize_self_business_area(
    *,
    peer_id: str | None,
    metric_scope: str,
    business_area: Any,
) -> tuple[str, Any]:
    if not business_area:
        return metric_scope, business_area
    business_area = _canonical_business_area(str(business_area))
    if not peer_id:
        return metric_scope, business_area

    if str(business_area) == "company_total":
        return "company_total", "company_total"

    area_key = _business_area_alias_key(str(business_area))
    self_keys = {_business_area_alias_key(peer_id)}
    if peer_id == "sk_ax":
        self_keys.update(
            {
                "skax",
                "sk에이엑스",
                "sk주식회사사업부문",
                "sk사업부문",
                "it서비스부문skax",
            }
        )

    if area_key in self_keys:
        return "company_total", "company_total"

    return metric_scope, business_area


def _business_area_from_evidence_override(
    *,
    peer_id: str | None,
    business_area: Any,
    evidence_text: str,
) -> Any:
    if peer_id != "sk_ax":
        return business_area

    lowered = evidence_text.lower()
    has_it_service_context = any(
        term in lowered
        for term in (
            "it서비스",
            "it 서비스",
            "it service",
            "it services",
            "enterprise it",
        )
    )
    has_explicit_cloud_context = _has_explicit_cloud_context(lowered)
    if has_it_service_context and not has_explicit_cloud_context:
        return "company_total"

    if str(business_area) != "cloud":
        return business_area

    explicit_areas = _detect_business_areas(evidence_text)
    explicit_non_cloud_areas = [area for area in explicit_areas if area != "cloud"]
    if explicit_non_cloud_areas and not _has_explicit_cloud_context(lowered):
        if len(explicit_non_cloud_areas) == 1:
            area = explicit_non_cloud_areas[0]
            return _canonical_business_area(area)
        return "other"

    if not has_explicit_cloud_context:
        return "other"

    return business_area


def _canonical_business_area(value: str) -> str:
    key = _business_area_alias_key(value)
    aliases = {
        "si": "SI",
        "systemintegration": "SI",
        "ito": "ITO",
        "itoutsourcing": "ITO",
        "차량sw": "vehicle_sw",
        "차량용sw": "vehicle_sw",
        "vehiclesw": "vehicle_sw",
        "automotivesw": "vehicle_sw",
        "enterpriseit": "Enterprise IT",
        "엔터프라이즈it": "Enterprise IT",
        "엔터프라이즈아이티": "Enterprise IT",
    }
    return aliases.get(key, value)


def _business_area_alias_key(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", value.lower())


if __name__ == "__main__":
    main()
