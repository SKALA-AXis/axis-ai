"""ir _document_processing — extracted from facade (move-only)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from src.config.sectors import SECTOR_KEYWORDS
from src.parsers.ir._constants import (  # noqa: F401
    _BACKLOG_PATTERNS,
    _CAPEX_PATTERNS,
    _EBITDA_PATTERNS,
    _IR_CHUNK_MAX_CHARS,
    _IR_CHUNK_MIN_CHARS,
    _IR_CHUNK_OVERLAP_CHARS,
    _IR_CHUNK_TARGET_CHARS,
    _IR_COMPANY_SECTION_HINTS,
    _IR_COMPANY_TOTAL_TERMS,
    _IR_FINANCIAL_METRIC_RULES,
    _IR_LOW_VALUE_PATTERNS,
    _IR_PORTFOLIO_TERMS,
    _IR_SECTION_RULES,
    _IR_SEGMENT_CONTEXT_RULES,
    _IR_SIGNAL_TERMS,
    _IR_TABLE_COMPARISON_PATTERN,
    _IR_TABLE_METRIC_ALIASES,
    _IR_TABLE_NOISE_LABEL_PATTERN,
    _IR_TABLE_NON_METRIC_LABEL_PATTERN,
    _IR_TABLE_NUMBER_PATTERN,
    _IR_TABLE_PERIOD_PATTERN,
    _NET_INCOME_PATTERNS,
    _OPERATING_MARGIN_PATTERNS,
    _OPERATING_PROFIT_PATTERNS,
    _ORDERS_PATTERNS,
    _PAGE_SPLIT_PATTERN,
    _PERIOD_PATTERNS,
    _REVENUE_PATTERNS,
    _SK_AX_PAGE_BUSINESS_TERMS,
    _SK_AX_PAGE_EXCLUDE_TERMS,
    _SK_AX_PAGE_STRONG_TERMS,
)
from src.parsers.ir._utils import (  # noqa: F401
    _candidate_value_key,
    _classify_metric_context,
    _detect_business_area,
    _detect_portfolio_entity,
    _extract_period,
    _first_amount,
    _first_metric_value,
    _has_same_table_metric_candidate,
    _inline_comparison_candidates,
    _metric_context_evidence,
    _metric_context_text,
    _metric_context_window,
    _metric_current_line_text,
    _metric_line_context,
    _metric_values,
    _normalize_amount_krwbn,
    _normalize_percentage,
    _period_parts,
    _signed_percentage,
)

log = logging.getLogger(__name__)


def _contains_token(text: str, term: str) -> bool:
    lowered_term = term.lower()
    if len(lowered_term) <= 3 and re.fullmatch(r"[a-z0-9&]+", lowered_term):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])",
                text,
            )
        )
    return lowered_term in text


def _is_sk_ax_page(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False

    has_strong_term = any(_contains_token(lowered, term) for term in _SK_AX_PAGE_STRONG_TERMS)
    if has_strong_term:
        return True

    has_excluded_affiliate = any(
        _contains_token(lowered, term) for term in _SK_AX_PAGE_EXCLUDE_TERMS
    )
    if has_excluded_affiliate:
        return False

    business_hits = sum(1 for term in _SK_AX_PAGE_BUSINESS_TERMS if _contains_token(lowered, term))
    return business_hits >= 2


def _filter_pages_for_peer(
    pages: list[dict[str, Any]], peer_id: str | None
) -> list[dict[str, Any]]:
    if peer_id != "sk_ax":
        return pages

    filtered = [page for page in pages if _is_sk_ax_page(str(page.get("text") or ""))]
    if filtered:
        log.info(
            "SK AX IR 관련 페이지 필터 적용 | before=%d after=%d pages=%s",
            len(pages),
            len(filtered),
            [page.get("page") for page in filtered],
        )
        return filtered

    log.warning(
        "SK AX IR 관련 페이지를 찾지 못해 원본 페이지 전체로 fallback | pages=%d", len(pages)
    )
    return pages


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

    value = extra.get("published_at")
    return str(value) if value else None


def _period_from_ir_article(article: Any, extra: dict[str, Any], text: str) -> str | None:
    title_period = _extract_period(str(_article_get(article, "title", "") or ""))
    text_period = _extract_period(text[:5000])
    if text_period:
        return text_period
    if title_period:
        return title_period

    date_info = extra.get("date_info") or {}
    if isinstance(date_info, dict):
        year = date_info.get("year")
        quarter = date_info.get("quarter")
        if year and quarter:
            return f"{year}Q{quarter}"

    return _extract_period(
        " ".join(
            [
                str(_article_get(article, "title", "") or ""),
                text[:5000],
            ]
        )
    )


def _pages_from_ir_article(article: Any, extra: dict[str, Any]) -> list[dict[str, Any]]:
    page_blocks = extra.get("pdf_page_blocks") or []
    pages: list[dict[str, Any]] = []

    if isinstance(page_blocks, list):
        for idx, page in enumerate(page_blocks, start=1):
            if not isinstance(page, dict):
                continue

            blocks = page.get("blocks") or []
            text = _page_text_from_pdf_blocks(blocks)
            page_no = int(page.get("page") or idx)

            if text:
                pages.append({"page": page_no, "text": text, "blocks": blocks})

    content_pages = _pages_from_article_content(str(_article_get(article, "content", "") or ""))
    if _should_use_content_pages(content_pages, pages):
        return content_pages

    if pages:
        return pages

    return content_pages


def _pages_from_article_content(text: str) -> list[dict[str, Any]]:
    matches = list(_PAGE_SPLIT_PATTERN.finditer(text))

    if not matches:
        return [{"page": None, "text": text}] if text else []

    pages: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        page_no = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append({"page": page_no, "text": page_text})

    return pages


def _should_use_content_pages(
    content_pages: list[dict[str, Any]],
    block_pages: list[dict[str, Any]],
) -> bool:
    if not content_pages:
        return False
    if not block_pages:
        return True

    content_text = "\n".join(str(page.get("text") or "") for page in content_pages)
    block_text = "\n".join(str(page.get("text") or "") for page in block_pages)
    if "[OCR]" in content_text:
        return True

    content_chars = len(re.findall(r"[0-9A-Za-z가-힣]", content_text))
    block_chars = len(re.findall(r"[0-9A-Za-z가-힣]", block_text))
    return content_chars > block_chars + 500 and content_chars > block_chars * 1.2


def _page_text_from_pdf_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return ""

    positioned: list[dict[str, Any]] = []
    fallback_texts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        if not text:
            continue
        fallback_texts.append(text)
        bbox = block.get("bbox")
        if not (
            isinstance(bbox, list | tuple)
            and len(bbox) >= 4
            and all(isinstance(value, int | float) for value in bbox[:4])
        ):
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        positioned.append(
            {
                "text": text,
                "x0": x0,
                "y_center": (y0 + y1) / 2,
                "height": max(y1 - y0, 1.0),
            }
        )

    if not positioned:
        return "\n".join(fallback_texts).strip()

    positioned.sort(key=lambda item: (item["y_center"], item["x0"]))
    median_height = sorted(item["height"] for item in positioned)[len(positioned) // 2]
    tolerance = max(4.0, min(10.0, median_height * 0.75))

    rows: list[list[dict[str, Any]]] = []
    for item in positioned:
        if not rows:
            rows.append([item])
            continue
        current_row = rows[-1]
        row_center = sum(cell["y_center"] for cell in current_row) / len(current_row)
        if abs(item["y_center"] - row_center) <= tolerance:
            current_row.append(item)
        else:
            rows.append([item])

    lines = [
        " ".join(cell["text"] for cell in sorted(row, key=lambda item: item["x0"])).strip()
        for row in rows
    ]
    return "\n".join(line for line in lines if line).strip()


def _company_section_rules(peer_id: str | None) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return _IR_COMPANY_SECTION_HINTS.get(peer_id or "", ()) + _IR_SECTION_RULES


def _classify_page_section(
    text: str,
    *,
    peer_id: str | None,
) -> tuple[str, str, list[str]]:
    normalized = text.lower()
    matches: list[str] = []

    for section_key, section_title, keywords in _company_section_rules(peer_id):
        hits = [keyword for keyword in keywords if keyword.lower() in normalized]
        if hits:
            return section_key, section_title, hits[:8]

    return "other", "Other IR Content", matches


def _match_topics(text: str) -> tuple[list[str], dict[str, list[str]]]:
    lowered = text.lower()
    topic_signals: dict[str, list[str]] = {}

    for topic, keywords in SECTOR_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword.lower() in lowered]
        if hits:
            topic_signals[topic] = hits[:8]

    return sorted(topic_signals), topic_signals


def _split_text_chunks(text: str, max_chars: int = _IR_CHUNK_MAX_CHARS) -> list[str]:
    value = _normalize_ir_chunk_text(text)
    if not value:
        return []
    if len(value) <= max_chars:
        return [value] if _is_informative_ir_chunk(value) else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in _ir_paragraphs(value):
        if len(paragraph) > max_chars:
            if current:
                _append_ir_chunk(chunks, "\n".join(current))
                current = []
                current_len = 0
            for split in _split_long_ir_paragraph(paragraph, max_chars=max_chars):
                _append_ir_chunk(chunks, split)
            continue

        next_len = current_len + len(paragraph) + (1 if current else 0)
        if current and next_len > max_chars:
            _append_ir_chunk(chunks, "\n".join(current))
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + (1 if current_len else 0)

        if current_len >= _IR_CHUNK_TARGET_CHARS:
            _append_ir_chunk(chunks, "\n".join(current))
            current = []
            current_len = 0

    if current:
        _append_ir_chunk(chunks, "\n".join(current))

    return chunks


def _normalize_ir_chunk_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", line):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def _ir_paragraphs(text: str) -> list[str]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for paragraph in paragraphs:
        if buffer and buffer_len + len(paragraph) > _IR_CHUNK_TARGET_CHARS:
            merged.append("\n".join(buffer))
            buffer = []
            buffer_len = 0

        buffer.append(paragraph)
        buffer_len += len(paragraph)

    if buffer:
        merged.append("\n".join(buffer))

    return merged


def _split_long_ir_paragraph(text: str, *, max_chars: int) -> list[str]:
    value = text.strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [value]

    chunks: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + max_chars, len(value))
        boundary = value.rfind("\n", start, end)
        if boundary <= start + max_chars // 2:
            boundary = value.rfind(". ", start, end)
        if boundary <= start:
            boundary = end

        chunk = value[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        if boundary >= len(value):
            break
        start = max(boundary - _IR_CHUNK_OVERLAP_CHARS, start + 1)

    return chunks


def _append_ir_chunk(chunks: list[str], text: str) -> None:
    value = text.strip()
    if _is_informative_ir_chunk(value):
        chunks.append(value)


def _is_informative_ir_chunk(text: str) -> bool:
    value = text.strip()
    if not value:
        return False

    lowered = value.lower()
    has_signal = _has_ir_signal(value)
    has_metric = _has_financial_metric_text(value)

    if any(pattern.search(value) for pattern in _IR_LOW_VALUE_PATTERNS) and not has_metric:
        return False

    if len(value) < _IR_CHUNK_MIN_CHARS and not has_signal and not has_metric:
        return False

    alpha_numeric_count = len(re.findall(r"[0-9A-Za-z가-힣]", value))
    if alpha_numeric_count < 30 and not has_metric:
        return False

    if lowered in {"disclaimer", "contents", "목차"}:
        return False

    return True


def _has_ir_signal(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in _IR_SIGNAL_TERMS)


def _has_financial_metric_text(text: str) -> bool:
    return any(
        pattern.search(text) for pattern in (*_REVENUE_PATTERNS, *_OPERATING_PROFIT_PATTERNS)
    )


def _extract_sections_and_chunks(
    pages: list[dict[str, Any]],
    *,
    peer_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, list[str]]]:
    sections_by_key: dict[str, dict[str, Any]] = {}
    document_chunks: list[dict[str, Any]] = []
    all_topic_signals: dict[str, list[str]] = {}

    for page in pages:
        page_no = page.get("page")
        page_text = str(page.get("text", "") or "").strip()
        if not page_text:
            continue

        section_key, section_title, section_signals = _classify_page_section(
            page_text,
            peer_id=peer_id,
        )
        topics, topic_signals = _match_topics(page_text)

        section = sections_by_key.setdefault(
            section_key,
            {
                "section_key": section_key,
                "section_title": section_title,
                "pages": [],
                "text_chars": 0,
                "signals": [],
                "topics": [],
            },
        )
        if page_no not in section["pages"]:
            section["pages"].append(page_no)
        section["text_chars"] += len(page_text)
        section["signals"] = sorted(set(section["signals"]) | set(section_signals))
        section["topics"] = sorted(set(section["topics"]) | set(topics))

        for topic, hits in topic_signals.items():
            merged = set(all_topic_signals.get(topic, []))
            merged.update(hits)
            all_topic_signals[topic] = sorted(merged)[:12]

        for local_idx, chunk_text in enumerate(_split_text_chunks(page_text), start=1):
            chunk_section_key, chunk_section_title, chunk_section_signals = _classify_page_section(
                chunk_text, peer_id=peer_id
            )
            chunk_topics, chunk_topic_signals = _match_topics(chunk_text)
            document_chunks.append(
                {
                    "chunk_id": f"ir-p{page_no or 'x'}-{local_idx}",
                    "page": page_no,
                    "section_key": chunk_section_key,
                    "section_title": chunk_section_title,
                    "section_signals": chunk_section_signals,
                    "chunk_index": len(document_chunks) + 1,
                    "text_chars": len(chunk_text),
                    "text": chunk_text,
                    "topics": chunk_topics,
                    "topic_signals": chunk_topic_signals,
                }
            )

    sections = list(sections_by_key.values())
    for idx, section in enumerate(sections, start=1):
        section["section_order"] = idx

    return sections, document_chunks, sorted(all_topic_signals), all_topic_signals


def _build_page_index(
    pages: list[dict[str, Any]],
    document_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compact page-level index so agents can target pages before reading chunks."""
    chunks_by_page: dict[Any, list[dict[str, Any]]] = {}
    for chunk in document_chunks:
        chunks_by_page.setdefault(chunk.get("page"), []).append(chunk)

    page_index: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        page_no = page.get("page") or index
        text = str(page.get("text") or "")
        page_chunks = chunks_by_page.get(page_no, [])
        section_keys = sorted(
            {str(chunk.get("section_key")) for chunk in page_chunks if chunk.get("section_key")}
        )
        topics = sorted(
            {str(topic) for chunk in page_chunks for topic in (chunk.get("topics") or []) if topic}
        )
        page_index.append(
            {
                "page": page_no,
                "text_chars": len(text),
                "chunk_count": len(page_chunks),
                "section_keys": section_keys,
                "topics": topics,
            }
        )
    return page_index
