# 작성일: 2026-05-13
# 작성자: 박지원
# 변경이력:
#   2026-05-13 박지원 — 증권사 리포트 파서 작성 및 네이버 리포트 파싱 개선
"""Naver research securities report parser."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.config.sectors import SECTOR_KEYWORDS

_INVESTMENT_OPINION_PATTERNS = (
    re.compile(
        r"(?:투자의견|Investment Opinion|Opinion)\s*[:：]?\s*"
        r"(BUY|HOLD|SELL|OUTPERFORM|NEUTRAL|매수|중립|비중확대|시장수익률)",
        re.I,
    ),
    re.compile(r"\b(BUY|HOLD|SELL|OUTPERFORM|NEUTRAL)\b", re.I),
    re.compile(r"\b(매수|중립|비중확대|시장수익률)\b"),
)
_TARGET_PRICE_PATTERNS = (
    re.compile(r"(?:목표주가|Target Price)\s*[:：]?\s*([0-9][0-9,]*)\s*원?"),
    re.compile(r"(?:TP)\s*[:：]?\s*([0-9][0-9,]*)\s*원?", re.I),
)
_CURRENT_PRICE_PATTERNS = (
    re.compile(r"(?:현재주가|Current Price)\s*[:：]?\s*([0-9][0-9,]*)\s*원?"),
    re.compile(r"(?:CP)\s*[:：]?\s*([0-9][0-9,]*)\s*원?", re.I),
)
_PERIOD_PATTERNS = (
    re.compile(r"\b([1-4])\s*Q\s*(\d{2})\s*(?:P|E)?\b", re.I),
    re.compile(r"(20\d{2})\s*Q\s*([1-4])", re.I),
    re.compile(r"(\d{2})\s*년\s*([1-4])\s*분기"),
    re.compile(r"(20\d{2})\s*년\s*([1-4])\s*분기"),
    re.compile(r"(20\d{2})E"),
)
_HIGHLIGHT_SPLIT = re.compile(r"(?:\r?\n)+")
_PAGE_MARKER = re.compile(r"(?m)^\s*\[PAGE\s+(\d+)\]\s*$")
_TRAILING_REPORT_SECTIONS = re.compile(
    r"목표주가\s*변동\s*추이|투자의견\s*변동\s*내역|투자의견\s*및\s*목표주가|"
    r"Compliance|Disclaimer|컴플라이언스|고지사항|등급분포|괴리율",
    re.I,
)
_PRICE_BAD_CONTEXT = re.compile(
    r"목표주가\s*변동\s*추이|투자의견\s*변동|괴리율|평균|최고|최저|등급분포|"
    r"Compliance|Disclaimer|고지사항",
    re.I,
)
_PRICE_MIN_KRW = 1_000
_PRICE_MAX_KRW = 10_000_000
_FRONT_MATTER_PAGES = 2
_SECTION_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("summary", "Summary", ("summary", "investment points", "핵심", "요약", "투자포인트")),
    ("investment_view", "Investment View", ("투자의견", "investment opinion", "buy", "hold")),
    ("valuation", "Valuation", ("목표주가", "target price", "valuation", "per", "pbr")),
    ("earnings_review", "Earnings Review", ("실적", "review", "earnings", "매출", "영업이익")),
    ("forecast", "Forecast", ("전망", "forecast", "guidance", "추정", "2026e", "2025e")),
    ("risks", "Risk", ("risk", "리스크", "우려", "downside")),
)
_CHUNK_MAX_CHARS = 2600
_CHUNK_OVERLAP_CHARS = 160


def _article_get(article: Any, key: str, default: Any = None) -> Any:
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _article_extra(article: Any) -> dict[str, Any]:
    extra = _article_get(article, "extra", {}) or {}
    if not extra:
        extra = _article_get(article, "metadata", {}) or {}
    return extra if isinstance(extra, dict) else {}


def _article_published_at(article: Any, extra: dict[str, Any]) -> str | None:
    published_at = _article_get(article, "published_at")
    if isinstance(published_at, datetime):
        return published_at.isoformat()
    if published_at:
        return str(published_at)
    value = extra.get("published_at")
    return str(value) if value else None


def _article_peer_id(article: Any) -> str | None:
    peer_id = _article_get(article, "peer_id")
    if peer_id:
        return str(peer_id)
    company = _article_get(article, "company", []) or []
    if isinstance(company, list) and company:
        return str(company[0])
    return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_price(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    for line in _candidate_lines(text):
        if _PRICE_BAD_CONTEXT.search(line):
            continue
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            digits = re.sub(r"[^0-9]", "", match.group(1))
            if not digits:
                continue
            value = int(digits)
            if _PRICE_MIN_KRW <= value <= _PRICE_MAX_KRW:
                return value
    return None


def _parse_investment_opinion(text: str) -> str | None:
    for line in _candidate_lines(text):
        if _PRICE_BAD_CONTEXT.search(line):
            continue
        for pattern in _INVESTMENT_OPINION_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            return _normalize_opinion(match.group(1))
    return None


def _parse_period(text: str) -> str | None:
    for pattern in _PERIOD_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if pattern.pattern.startswith(r"\b([1-4]"):
            return f"20{match.group(2)}Q{match.group(1)}"
        if len(match.groups()) == 2 and match.group(2):
            year = match.group(1)
            if len(year) == 2:
                year = f"20{year}"
            return f"{year}Q{match.group(2)}"
        return f"{match.group(1)}E"
    return None


def _candidate_lines(text: str) -> list[str]:
    return [_normalize_text(line) for line in (text or "").splitlines() if _normalize_text(line)]


def _normalize_opinion(value: str) -> str:
    opinion = _normalize_text(value).upper()
    mapping = {
        "BUY": "BUY",
        "HOLD": "HOLD",
        "SELL": "SELL",
        "OUTPERFORM": "OUTPERFORM",
        "NEUTRAL": "NEUTRAL",
    }
    if opinion in mapping:
        return mapping[opinion]
    return _normalize_text(value)


def _pages_from_article(text: str, extra: dict[str, Any]) -> list[dict[str, Any]]:
    page_blocks = extra.get("pdf_page_blocks") or []
    pages: list[dict[str, Any]] = []
    if isinstance(page_blocks, list):
        for page in page_blocks:
            if not isinstance(page, dict):
                continue
            page_no = page.get("page")
            blocks = page.get("blocks") or []
            page_text = "\n".join(
                str(block.get("text") or "")
                for block in blocks
                if isinstance(block, dict) and block.get("text")
            )
            if page_text.strip():
                pages.append({"page": page_no, "text": page_text, "blocks": blocks})
    if pages:
        return pages

    matches = list(_PAGE_MARKER.finditer(text or ""))
    if not matches:
        return [{"page": 1, "text": text or ""}] if text else []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append({"page": int(match.group(1)), "text": page_text})
    return pages


def _front_matter_text(pages: list[dict[str, Any]], fallback_text: str) -> str:
    if not pages:
        return fallback_text[:6000]
    front = "\n".join(
        str(page.get("text") or "")
        for page in pages
        if isinstance(page.get("page"), int) and int(page["page"]) <= _FRONT_MATTER_PAGES
    )
    if not front:
        front = "\n".join(str(page.get("text") or "") for page in pages[:_FRONT_MATTER_PAGES])
    return _strip_trailing_report_sections(front or fallback_text[:6000])


def _strip_trailing_report_sections(text: str) -> str:
    match = _TRAILING_REPORT_SECTIONS.search(text or "")
    return text[: match.start()] if match else text


def _match_topics(text: str) -> tuple[list[str], dict[str, list[str]]]:
    lowered = text.lower()
    topic_signals: dict[str, list[str]] = {}
    for topic, info in SECTOR_KEYWORDS.items():
        hits = [keyword for keyword in info["keywords"] if keyword.lower() in lowered]
        if hits:
            topic_signals[topic] = hits[:8]
    return sorted(topic_signals), topic_signals


def _section_for_text(text: str) -> tuple[str, str]:
    lowered = text.lower()
    for section_key, title, keywords in _SECTION_RULES:
        if any(keyword.lower() in lowered for keyword in keywords):
            return section_key, title
    return "other", "Other"


def _split_chunks(text: str) -> list[str]:
    value = text.strip()
    if not value:
        return []
    if len(value) <= _CHUNK_MAX_CHARS:
        return [value]

    chunks: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + _CHUNK_MAX_CHARS, len(value))
        boundary = value.rfind("\n", start, end)
        if boundary <= start + _CHUNK_MAX_CHARS // 2:
            boundary = value.rfind(". ", start, end)
        if boundary <= start:
            boundary = end
        chunk = value[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        if boundary >= len(value):
            break
        start = max(boundary - _CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _extract_sections_and_chunks(
    text: str,
    pages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    parts_with_page = _section_parts(text, pages)

    if not parts_with_page:
        return [], []

    for part, page_no in parts_with_page:
        if _TRAILING_REPORT_SECTIONS.search(part):
            continue
        section_key, section_title = _section_for_text(part)
        topics, topic_signals = _match_topics(part)
        section = sections.setdefault(
            section_key,
            {
                "section_key": section_key,
                "section_title": section_title,
                "text_chars": 0,
                "chunk_count": 0,
                "topics": [],
                "snippet": "",
            },
        )
        section["text_chars"] += len(part)
        section["topics"] = sorted(set(section["topics"]) | set(topics))
        if not section["snippet"]:
            section["snippet"] = part[:1200]

        for chunk_text in _split_chunks(part):
            chunks.append(
                {
                    "chunk_id": f"{section_key}:{len(chunks) + 1}",
                    "section_key": section_key,
                    "section_title": section_title,
                    "chunk_index": len(chunks) + 1,
                    "page": page_no,
                    "text_chars": len(chunk_text),
                    "text": chunk_text,
                    "topics": topics,
                    "topic_signals": topic_signals,
                }
            )
            section["chunk_count"] += 1

    ordered_sections = list(sections.values())
    for index, section in enumerate(ordered_sections, start=1):
        section["section_order"] = index

    return ordered_sections, chunks


def _section_parts(
    text: str,
    pages: list[dict[str, Any]] | None,
) -> list[tuple[str, int | None]]:
    if pages:
        parts: list[tuple[str, int | None]] = []
        for page in pages:
            page_no = page.get("page")
            page_index = int(page_no) if isinstance(page_no, int) else None
            page_text = _strip_trailing_report_sections(str(page.get("text") or ""))
            for part in _HIGHLIGHT_SPLIT.split(page_text):
                cleaned = part.strip()
                if cleaned:
                    parts.append((cleaned, page_index))
        return parts
    return [(part.strip(), None) for part in _HIGHLIGHT_SPLIT.split(text or "") if part.strip()]


def _extract_highlights(text: str) -> list[str]:
    highlights: list[str] = []
    for part in _HIGHLIGHT_SPLIT.split(text or ""):
        cleaned = _normalize_text(part)
        if len(cleaned) < 12:
            continue
        highlights.append(cleaned)
        if len(highlights) >= 5:
            break
    return highlights


class SecuritiesReportParser:
    def parse_article(self, article: Any) -> dict[str, Any]:
        extra = _article_extra(article)
        text = str(_article_get(article, "content", "") or "")
        title = str(_article_get(article, "title", "") or "")
        url = str(_article_get(article, "url", "") or extra.get("pdf_url", "") or "")
        peer_id = _article_peer_id(article)
        published_at = _article_published_at(article, extra)
        pages = _pages_from_article(text, extra)
        front_matter = _front_matter_text(pages, text)
        period = _parse_period(" ".join([title, front_matter, text[:2000]]))
        topics, topic_signals = _match_topics(text)
        sections, document_chunks = _extract_sections_and_chunks(text, pages)

        metadata = {
            "report_firm": extra.get("firm") or _article_get(article, "publisher"),
            "pdf_url": extra.get("pdf_url"),
            "item_code": extra.get("item_code"),
            "pdf_pages": extra.get("pdf_pages"),
            "pdf_parsed_pages": extra.get("pdf_parsed_pages"),
            "pdf_text_chars": extra.get("pdf_text_chars") or len(text),
            "pdf_parse_strategy": extra.get("pdf_parse_strategy"),
            "table_parse_strategy": extra.get("table_parse_strategy"),
            "chart_parse_strategy": extra.get("chart_parse_strategy"),
            "contains_images": extra.get("contains_images"),
            "image_count": extra.get("image_count"),
            "contains_tables": extra.get("contains_tables"),
            "table_count": extra.get("table_count"),
            "tables": extra.get("tables"),
        }

        return {
            "ok": bool(text),
            "parser": "securities_report_parser",
            "source": "securities_report",
            "source_type": "securities_report",
            "source_name": _article_get(article, "source_name"),
            "peer_id": peer_id,
            "title": title,
            "url": url,
            "published_at": published_at,
            "report_firm": metadata["report_firm"],
            "investment_opinion": _parse_investment_opinion(front_matter),
            "target_price_krw": _parse_price(_TARGET_PRICE_PATTERNS, front_matter),
            "current_price_krw": _parse_price(_CURRENT_PRICE_PATTERNS, front_matter),
            "period": period,
            "highlights": _extract_highlights(front_matter or text),
            "sections": sections,
            "document_chunks": document_chunks,
            "topics": topics,
            "topic_signals": topic_signals,
            "metadata": metadata,
            "warnings": [] if text else ["content 없음"],
        }


__all__ = ["SecuritiesReportParser"]
