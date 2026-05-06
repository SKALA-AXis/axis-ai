"""DART 크롤링 결과를 재무 후보 레코드로 변환하는 파서 에이전트."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from src.agents.ir_parser_agent import (
    _OPERATING_PROFIT_PATTERNS,
    _REVENUE_PATTERNS,
    _extract_period,
    _first_amount,
)

log = logging.getLogger(__name__)

_REPORT_PERIOD_PATTERN = re.compile(r"\((20\d{2})\.(0[369]|12)\)")


def _article_get(article: Any, key: str, default: Any = None) -> Any:
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _article_extra(article: Any) -> dict[str, Any]:
    extra = _article_get(article, "extra", {}) or {}
    return extra if isinstance(extra, dict) else {}


def _article_peer_id(article: Any) -> str | None:
    peer_id = _article_get(article, "peer_id")
    if peer_id:
        return str(peer_id)

    company = _article_get(article, "company", []) or []
    if isinstance(company, list) and company:
        return str(company[0])

    return None


def _article_published_at(article: Any, extra: dict[str, Any]) -> str | None:
    published_at = _article_get(article, "published_at")
    if isinstance(published_at, datetime):
        return published_at.isoformat()
    if published_at:
        return str(published_at)

    rcept_dt = extra.get("rcept_dt")
    if isinstance(rcept_dt, str) and len(rcept_dt) == 8:
        return f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"

    return None


def _period_from_report_name(report_name: str) -> tuple[str | None, str | None]:
    name = report_name or ""
    match = _REPORT_PERIOD_PATTERN.search(name)

    if not match:
        return _extract_period(name), None

    year = match.group(1)
    month = int(match.group(2))

    if "사업보고서" in name:
        return f"{year}Q4", "annual"
    if "반기보고서" in name:
        return f"{year}Q2", "half"
    if "분기보고서" in name:
        return f"{year}Q{month // 3}", "quarter"

    return f"{year}Q{month // 3}", None


def _candidate_page(candidates: list[dict[str, Any]], metric_type: str) -> int | None:
    for candidate in candidates:
        if candidate.get("type") == metric_type:
            page = candidate.get("page")
            return int(page) if isinstance(page, int) else None
    return None


class DartParserAgent:
    """DartCrawler가 만든 RawArticle 또는 dict 결과를 파싱한다."""

    def parse_article(self, article: Any) -> dict[str, Any]:
        extra = _article_extra(article)
        text = str(_article_get(article, "content", "") or "")
        peer_id = _article_peer_id(article)
        title = str(_article_get(article, "title", "") or "")
        url = str(_article_get(article, "url", "") or "")
        published_at = _article_published_at(article, extra)

        report_name = str(extra.get("report_name") or extra.get("report_nm") or title)
        period, period_type = _period_from_report_name(report_name)

        if not period:
            period = _extract_period(" ".join([report_name, text[:5000]]))

        warnings: list[str] = []
        candidates: list[dict[str, Any]] = []

        revenue_total, revenue_raw = _first_amount(text, _REVENUE_PATTERNS)
        if revenue_total is not None:
            candidates.append(
                {
                    "page": None,
                    "type": "revenue_total",
                    "value_krwbn": revenue_total,
                    "raw": revenue_raw,
                }
            )

        operating_profit, operating_profit_raw = _first_amount(
            text,
            _OPERATING_PROFIT_PATTERNS,
        )
        if operating_profit is not None:
            candidates.append(
                {
                    "page": None,
                    "type": "operating_profit",
                    "value_krwbn": operating_profit,
                    "raw": operating_profit_raw,
                }
            )

        if not text:
            warnings.append("content 없음")
        if not period:
            warnings.append("period 추출 실패")
        if revenue_total is None:
            warnings.append("revenue_total 추출 실패")
        if operating_profit is None:
            warnings.append("operating_profit 추출 실패")
        if not extra.get("document_fetched"):
            warnings.append("DART 원문 미수집 상태일 수 있음")

        rcept_no = str(extra.get("rcept_no") or extra.get("receipt_no") or "")
        financial_record = {
            "peer_id": peer_id,
            "period": period,
            "period_type": period_type,
            "revenue_total_krwbn": revenue_total,
            "operating_profit_krwbn": operating_profit,
            "source": "dart",
            "title": title,
            "url": url,
            "published_at": published_at,
            "dart_rcept_no": rcept_no or None,
            "dart_report_name": report_name,
            "dart_page": _candidate_page(candidates, "revenue_total")
            or _candidate_page(candidates, "operating_profit"),
        }

        result = {
            "ok": bool(text),
            "source": "dart",
            "peer_id": peer_id,
            "title": title,
            "url": url,
            "published_at": published_at,
            "period": period,
            "period_type": period_type,
            "rcept_no": rcept_no or None,
            "corp_code": extra.get("corp_code"),
            "corp_name": extra.get("corp_name"),
            "stock_code": extra.get("stock_code"),
            "report_name": report_name,
            "disclosure_type": extra.get("disclosure_type"),
            "disclosure_type_label": extra.get("disclosure_type_label"),
            "revenue_total_krwbn": revenue_total,
            "operating_profit_krwbn": operating_profit,
            "candidates": candidates,
            "document": {
                "document_fetched": extra.get("document_fetched"),
                "document_fetch_error": extra.get("document_fetch_error"),
                "document_text_length": extra.get("document_text_length"),
                "content_chars": extra.get("content_chars") or len(text),
                "content_truncated": extra.get("content_truncated"),
                "contains_tables": extra.get("contains_tables"),
                "table_count": extra.get("table_count"),
                "table_parse_strategy": extra.get("table_parse_strategy"),
                "contains_images": extra.get("contains_images"),
                "image_count": extra.get("image_count"),
            },
            "financial_record": financial_record,
            "warnings": warnings,
        }

        log.info(
            "DART article 파싱 완료 | peer_id=%s rcept_no=%s period=%s rev=%s op=%s",
            peer_id,
            rcept_no,
            period,
            revenue_total,
            operating_profit,
        )
        return result


__all__ = ["DartParserAgent"]
