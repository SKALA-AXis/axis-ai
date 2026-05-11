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
_DART_STATEMENT_ANCHOR_PATTERN = re.compile(r"(?:연\s*결\s*)?(?:포\s*괄\s*)?손\s*익\s*계\s*산\s*서")
_DART_AMOUNT_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
_DART_ROW_STOP_PATTERN = re.compile(
    r"매출원가|매출총이익|판매비와관리비|영업이익|기타수익|기타비용|금융수익|금융비용|"
    r"법인세|당기순이익|기타포괄손익|주당이익"
)


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


def _normalize_dart_amount_krwbn(value: str, unit: str) -> float | None:
    amount_text = value.strip()
    negative = amount_text.startswith("(") and amount_text.endswith(")")
    amount_text = amount_text.strip("()").replace(",", "")

    try:
        amount = float(amount_text)
    except ValueError:
        return None

    if negative:
        amount = -amount

    compact_unit = re.sub(r"\s+", "", unit)
    if compact_unit == "원":
        return amount / 100_000_000
    if compact_unit == "천원":
        return amount / 100_000
    if compact_unit == "백만원":
        return amount / 100
    if compact_unit in {"억원", "억"}:
        return amount
    if compact_unit in {"조원", "조"}:
        return amount * 10_000

    return None


def _dart_statement_unit(section: str) -> str | None:
    match = re.search(r"단위\s*:\s*(원|천원|백만원|억원|억|조원|조)", section)
    return match.group(1) if match else None


def _extract_dart_statement_amount(
    section: str, label: str, unit: str
) -> tuple[float, str] | tuple[None, None]:
    label_match = re.search(label, section)
    if not label_match:
        return None, None

    row = section[label_match.end() : label_match.end() + 350]
    stop_match = _DART_ROW_STOP_PATTERN.search(row)
    if stop_match:
        row = row[: stop_match.start()]

    for match in _DART_AMOUNT_PATTERN.finditer(row):
        raw_amount = match.group(0)
        normalized = _normalize_dart_amount_krwbn(raw_amount, unit)
        if normalized is None:
            continue

        # DART rows often include footnote numbers before the actual amount.
        if abs(normalized) < 1:
            continue

        return normalized, f"{label_match.group(0)} {raw_amount} ({unit})"

    return None, None


def _extract_dart_statement_metrics(text: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    anchors = list(_DART_STATEMENT_ANCHOR_PATTERN.finditer(text or ""))

    for idx, anchor in enumerate(anchors):
        next_start = anchors[idx + 1].start() if idx + 1 < len(anchors) else anchor.start() + 3000
        section = text[anchor.start() : min(next_start, anchor.start() + 3000)]
        if "매출액" not in section or "영업이익" not in section:
            continue

        unit = _dart_statement_unit(section)
        if not unit:
            continue

        revenue_total, revenue_raw = _extract_dart_statement_amount(section, "매출액", unit)
        operating_profit, operating_profit_raw = _extract_dart_statement_amount(
            section,
            "영업이익",
            unit,
        )
        if revenue_total is None and operating_profit is None:
            continue

        score = 1
        if re.search(r"연\s*결", section[:500]):
            score += 10

        candidate = {
            "score": score,
            "revenue_total": revenue_total,
            "revenue_raw": revenue_raw,
            "operating_profit": operating_profit,
            "operating_profit_raw": operating_profit_raw,
        }
        if not best or candidate["score"] > best["score"]:
            best = candidate

    return best


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
        document_fetched = bool(extra.get("document_fetched"))

        statement_metrics = _extract_dart_statement_metrics(text)

        revenue_total = statement_metrics.get("revenue_total")
        revenue_raw = statement_metrics.get("revenue_raw")
        if revenue_total is None:
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

        operating_profit = statement_metrics.get("operating_profit")
        operating_profit_raw = statement_metrics.get("operating_profit_raw")
        if operating_profit is None:
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
        if not document_fetched:
            warnings.append("DART 목록 메타데이터만 수집됨: 원문 HTML/XBRL 본문은 미수집")
        elif revenue_total is None:
            warnings.append("revenue_total 추출 실패")
        if document_fetched and operating_profit is None:
            warnings.append("operating_profit 추출 실패")

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
            "financial_metrics_source": (
                "dart_document_text" if document_fetched else "not_available_without_document"
            ),
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
                "document_fetched": document_fetched,
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
