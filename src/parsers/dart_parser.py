"""DART 크롤링 결과를 재무 후보 레코드로 변환하는 파서 에이전트."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from src.config.sectors import SECTOR_KEYWORDS
from src.parsers.ir_parser import (
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
_DART_MAJOR_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("company_overview", "I", "회사의 개요"),
    ("business", "II", "사업의 내용"),
    ("financial", "III", "재무에 관한 사항"),
    ("management_discussion", "IV", "이사의 경영진단 및 분석의견"),
    ("auditor", "V", "회계감사인의 감사의견"),
    ("governance", "VI", "이사회 등 회사의 기관에 관한 사항"),
    ("shareholder", "VII", "주주에 관한 사항"),
    ("executives", "VIII", "임원 및 직원 등에 관한 사항"),
    ("affiliates", "IX", "계열회사 등에 관한 사항"),
    ("major_shareholder", "X", "대주주 등과의 거래내용"),
    ("other", "XI", "그 밖에 투자자 보호를 위하여 필요한 사항"),
)
_ROMAN_TO_KEY = {roman: (section_key, title) for section_key, roman, title in _DART_MAJOR_SECTIONS}
_ROMAN_VARIANTS = {
    "I": "Ⅰ",
    "II": "Ⅱ",
    "III": "Ⅲ",
    "IV": "Ⅳ",
    "V": "Ⅴ",
    "VI": "Ⅵ",
    "VII": "Ⅶ",
    "VIII": "Ⅷ",
    "IX": "Ⅸ",
    "X": "Ⅹ",
    "XI": "Ⅺ",
}
_DART_MAJOR_HEADING_PATTERN = re.compile(
    r"(?:(?<=\n)|^)\s*"
    r"(?P<roman>XI|IX|VIII|VII|VI|IV|III|II|X|V|I|Ⅺ|Ⅸ|Ⅷ|Ⅶ|Ⅵ|Ⅳ|Ⅲ|Ⅱ|Ⅹ|Ⅴ|Ⅰ)"
    r"\s*[.\)]?\s*"
    r"(?P<title>"
    r"회사의\s*개요|사업의\s*내용|재무에\s*관한\s*사항|"
    r"이사의\s*경영진단\s*및\s*분석의견|회계감사인의\s*감사의견|"
    r"이사회\s*등\s*회사의\s*기관에\s*관한\s*사항|주주에\s*관한\s*사항|"
    r"임원\s*및\s*직원\s*등에\s*관한\s*사항|계열회사\s*등에\s*관한\s*사항|"
    r"대주주\s*등과의\s*거래내용|그\s*밖에\s*투자자\s*보호를\s*위하여\s*필요한\s*사항"
    r")",
    re.MULTILINE,
)
_DART_SUB_HEADING_PATTERN = re.compile(
    r"(?:(?<=\n)|^)\s*(?P<label>(?:\d{1,2}|[가-힣]|[A-Z])\s*[.\)]|\(\s*\d{1,2}\s*\))\s*"
    r"(?P<title>[^\n]{2,80})"
)
_DART_CHUNK_MAX_CHARS = 3500
_DART_CHUNK_OVERLAP_CHARS = 250


def _article_get(article: Any, key: str, default: Any = None) -> Any:
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _article_extra(article: Any) -> dict[str, Any]:
    extra = _article_get(article, "extra", {}) or {}
    if not extra:
        extra = _article_get(article, "metadata", {}) or {}
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


def _period_parts(period: str | None) -> dict[str, int | None]:
    if not period:
        return {"period_year": None, "period_quarter": None}
    match = re.match(r"(20\d{2})Q([1-4])", period)
    if not match:
        return {"period_year": None, "period_quarter": None}
    return {
        "period_year": int(match.group(1)),
        "period_quarter": int(match.group(2)),
    }


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


def _extract_table_statement_metrics(tables: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}

    for table in tables:
        table_text = str(table.get("text") or "")
        if "매출액" not in table_text and "영업이익" not in table_text:
            continue

        unit = _dart_statement_unit(table_text) or _infer_table_unit(table)
        if not unit:
            continue

        revenue_total, revenue_raw = _extract_metric_from_table_rows(table, "매출액", unit)
        operating_profit, operating_profit_raw = _extract_metric_from_table_rows(
            table,
            "영업이익",
            unit,
        )

        if revenue_total is None and operating_profit is None:
            continue

        score = 5
        title_text = " ".join(
            [
                str(table.get("title") or ""),
                table_text[:500],
            ]
        )
        if re.search(r"연\s*결", title_text):
            score += 10
        if "손익계산서" in title_text or "포괄손익계산서" in title_text:
            score += 5

        candidate: dict[str, Any] = {
            "score": score,
            "source": "structured_table",
            "table_index": table.get("table_index"),
            "table_title": table.get("title"),
            "revenue_total": revenue_total,
            "revenue_raw": revenue_raw,
            "operating_profit": operating_profit,
            "operating_profit_raw": operating_profit_raw,
        }
        if not best or score > int(best.get("score") or 0):
            best = candidate

    return best


def _extract_metric_from_table_rows(
    table: dict[str, Any],
    label: str,
    unit: str,
) -> tuple[float, str] | tuple[None, None]:
    rows = table.get("rows")
    if not isinstance(rows, list):
        return None, None

    for row in rows:
        if not isinstance(row, list):
            continue
        row_values = [str(value) for value in row if value is not None]
        row_text = " | ".join(row_values)
        if label not in row_text:
            continue

        amounts: list[tuple[float, str]] = []
        for match in _DART_AMOUNT_PATTERN.finditer(row_text):
            raw = match.group(0)
            value = _normalize_dart_amount_krwbn(raw, unit)
            if value is not None and abs(value) >= 1:
                amounts.append((value, raw))
        if not amounts:
            continue

        value, raw = amounts[0]
        return value, f"{label} {raw} ({unit}) table={table.get('table_index')}"

    return None, None


def _infer_table_unit(table: dict[str, Any]) -> str | None:
    text = " ".join(
        [
            str(table.get("title") or ""),
            str(table.get("text") or "")[:1000],
        ]
    )
    return _dart_statement_unit(text)


def _compact_tables_for_parser_result(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for table in tables[:20]:
        compacted.append(
            {
                "table_index": table.get("table_index"),
                "title": table.get("title"),
                "row_count": table.get("row_count"),
                "column_count": table.get("column_count"),
                "text": str(table.get("text") or "")[:2000],
            }
        )
    return compacted


def _extract_sections_and_chunks(
    text: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not text:
        return {}, []

    anchors = _major_section_anchors(text)
    if not anchors:
        cleaned = _clean_section_text(text)
        chunk = _build_chunk(
            text=cleaned,
            section_key="unclassified",
            section_title="분류되지 않은 본문",
            section_order=0,
            subsection_title=None,
            subsection_label=None,
            chunk_index=1,
        )
        return (
            {
                "unclassified": {
                    "order": 0,
                    "title": "분류되지 않은 본문",
                    "text_chars": len(cleaned),
                    "snippet": cleaned[:1200],
                    "chunk_count": 1 if cleaned else 0,
                }
            },
            [chunk] if chunk else [],
        )

    sections: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors):
        end = anchors[index + 1]["start"] if index + 1 < len(anchors) else len(text)
        raw_section_text = text[anchor["start"] : end]
        section_text = _clean_section_text(raw_section_text)
        subsection_chunks = _split_section_into_chunks(
            raw_section_text=raw_section_text,
            section_key=anchor["key"],
            section_title=anchor["title"],
            section_order=index + 1,
        )
        sections[anchor["key"]] = {
            "order": index + 1,
            "title": anchor["title"],
            "text_chars": len(section_text),
            "snippet": section_text[:1200],
            "chunk_count": len(subsection_chunks),
        }
        chunks.extend(subsection_chunks)

    return sections, chunks


def _major_section_anchors(text: str) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for match in _DART_MAJOR_HEADING_PATTERN.finditer(text):
        roman = _normalize_roman(match.group("roman"))
        mapped = _ROMAN_TO_KEY.get(roman)
        if not mapped:
            continue
        section_key, canonical_title = mapped
        if section_key in seen_keys:
            continue
        seen_keys.add(section_key)
        anchors.append(
            {
                "key": section_key,
                "roman": roman,
                "title": canonical_title,
                "matched_title": _clean_section_text(match.group(0)),
                "start": match.start(),
            }
        )

    anchors.sort(key=lambda item: int(item["start"]))
    return anchors


def _normalize_roman(value: str) -> str:
    value = value.strip()
    for ascii_roman, unicode_roman in _ROMAN_VARIANTS.items():
        if value == unicode_roman:
            return ascii_roman
    return value


def _split_section_into_chunks(
    *,
    raw_section_text: str,
    section_key: str,
    section_title: str,
    section_order: int,
) -> list[dict[str, Any]]:
    subsection_anchors = _subsection_anchors(raw_section_text)
    chunks: list[dict[str, Any]] = []

    if not subsection_anchors:
        for chunk_text in _chunk_text(_clean_section_text(raw_section_text)):
            chunk = _build_chunk(
                text=chunk_text,
                section_key=section_key,
                section_title=section_title,
                section_order=section_order,
                subsection_title=None,
                subsection_label=None,
                chunk_index=len(chunks) + 1,
            )
            if chunk:
                chunks.append(chunk)
        return chunks

    for index, anchor in enumerate(subsection_anchors):
        end = (
            subsection_anchors[index + 1]["start"]
            if index + 1 < len(subsection_anchors)
            else len(raw_section_text)
        )
        subsection_text = _clean_section_text(raw_section_text[anchor["start"] : end])
        for chunk_text in _chunk_text(subsection_text):
            chunk = _build_chunk(
                text=chunk_text,
                section_key=section_key,
                section_title=section_title,
                section_order=section_order,
                subsection_title=anchor["title"],
                subsection_label=anchor["label"],
                chunk_index=len(chunks) + 1,
            )
            if chunk:
                chunks.append(chunk)

    return chunks


def _subsection_anchors(raw_section_text: str) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for match in _DART_SUB_HEADING_PATTERN.finditer(raw_section_text):
        label = _clean_section_text(match.group("label"))
        title = _clean_section_text(match.group("title"))
        if _looks_like_noise_heading(title):
            continue
        anchors.append(
            {
                "label": label,
                "title": title,
                "start": match.start(),
            }
        )
    return anchors


def _looks_like_noise_heading(title: str) -> bool:
    lowered = title.lower()
    if len(title) < 2:
        return True
    if any(token in lowered for token in ("http", "dart", "전자공시", "목차")):
        return True
    return False


def _chunk_text(text: str) -> list[str]:
    if not text:
        return []
    if len(text) <= _DART_CHUNK_MAX_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + _DART_CHUNK_MAX_CHARS)
        if end < len(text):
            sentence_end = max(text.rfind(". ", start, end), text.rfind("다. ", start, end))
            if sentence_end > start + 1000:
                end = sentence_end + 2
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - _DART_CHUNK_OVERLAP_CHARS, start + 1)
    return [chunk for chunk in chunks if chunk]


def _build_chunk(
    *,
    text: str,
    section_key: str,
    section_title: str,
    section_order: int,
    subsection_title: str | None,
    subsection_label: str | None,
    chunk_index: int,
) -> dict[str, Any] | None:
    cleaned = _clean_section_text(text)
    if len(cleaned) < 40:
        return None
    topic_signals = _extract_topic_signals(cleaned)
    return {
        "chunk_id": f"{section_key}:{chunk_index}",
        "section_key": section_key,
        "section_title": section_title,
        "section_order": section_order,
        "subsection_label": subsection_label,
        "subsection_title": subsection_title,
        "chunk_index": chunk_index,
        "text_chars": len(cleaned),
        "text": cleaned,
        "topics": [signal["topic"] for signal in topic_signals],
        "topic_signals": topic_signals,
    }


def _clean_section_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _extract_topic_signals(text: str) -> list[dict[str, Any]]:
    normalized = (text or "").lower()
    signals: list[dict[str, Any]] = []

    for sector_id, sector_info in SECTOR_KEYWORDS.items():
        terms = tuple(sector_info["keywords"])
        matches: list[str] = []
        count = 0
        for term in terms:
            term_lower = term.lower()
            term_count = normalized.count(term_lower)
            if term_count:
                count += term_count
                matches.append(term)

        if count:
            signals.append(
                {
                    "topic": sector_id,
                    "topic_name_ko": sector_info["name_ko"],
                    "count": count,
                    "matched_terms": matches[:8],
                    "snippet": _topic_snippet(text, matches[0]) if matches else "",
                }
            )

    signals.sort(key=lambda item: int(item["count"]), reverse=True)
    return signals


def _topic_snippet(text: str, term: str, radius: int = 180) -> str:
    if not text or not term:
        return ""
    match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return _clean_section_text(text[start:end])


class DartParser:
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

        period_parts = _period_parts(period)
        sections, document_chunks = _extract_sections_and_chunks(text)
        topic_signals = _extract_topic_signals(text)
        warnings: list[str] = []
        candidates: list[dict[str, Any]] = []
        document_fetched = bool(extra.get("document_fetched"))
        raw_tables = extra.get("tables")
        tables: list[dict[str, Any]] = (
            [table for table in raw_tables if isinstance(table, dict)]
            if isinstance(raw_tables, list)
            else []
        )

        table_metrics = _extract_table_statement_metrics(tables)
        statement_metrics = table_metrics or _extract_dart_statement_metrics(text)

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
                    "source": statement_metrics.get("source", "document_text"),
                    "table_index": statement_metrics.get("table_index"),
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
                    "source": statement_metrics.get("source", "document_text"),
                    "table_index": statement_metrics.get("table_index"),
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
            **period_parts,
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
                statement_metrics.get("source")
                or ("dart_document_text" if document_fetched else "not_available_without_document")
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
            **period_parts,
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
                "structured_table_count": len(tables),
                "tables": _compact_tables_for_parser_result(tables),
                "table_parse_strategy": extra.get("table_parse_strategy"),
                "contains_images": extra.get("contains_images"),
                "image_count": extra.get("image_count"),
            },
            "sections": sections,
            "document_chunks": document_chunks,
            "topic_signals": topic_signals,
            "topics": [signal["topic"] for signal in topic_signals],
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


__all__ = ["DartParser"]
