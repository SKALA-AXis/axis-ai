"""PDF 텍스트/블록/표 후보 메타데이터 추출 유틸."""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def extract_pdf_payload(pdf_bytes: bytes, *, max_text_chars: int = 200_000) -> dict[str, Any]:
    """PDF bytes에서 텍스트, 페이지 블록, 표 후보 메타데이터를 추출한다."""

    try:
        import pymupdf  # type: ignore

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            chunks: list[str] = []
            pages_payload: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []
            current_length = 0
            image_count = 0

            for page_index, page in enumerate(doc, start=1):
                page_text = _clean_pdf_text(page.get_text("text", sort=True) or "")
                page_blocks = _extract_pdf_page_blocks(page)
                page_images = len(page.get_images(full=True))
                page_tables = _detect_table_candidates(page_index, page_blocks)

                image_count += page_images
                tables.extend(page_tables)
                chunks.append(f"\n[PAGE {page_index}]\n{page_text}")
                current_length += len(page_text)
                pages_payload.append(
                    {
                        "page": page_index,
                        "text_chars": len(page_text),
                        "image_count": page_images,
                        "blocks": page_blocks,
                        "table_candidates": page_tables,
                    }
                )

                if current_length >= max_text_chars:
                    break

            text = "\n".join(chunks).strip()[:max_text_chars]

            return {
                "text": text,
                "page_count": len(doc),
                "parsed_page_count": len(pages_payload),
                "image_count": image_count,
                "contains_images": image_count > 0,
                "pages": pages_payload,
                "tables": tables,
                "table_count": len(tables),
                "contains_tables": bool(tables),
                "pdf_parse_strategy": "text_blocks_table_candidates",
                "table_parse_strategy": "numeric_text_block_candidates",
                "chart_parse_strategy": "not_parsed",
            }

    except Exception as e:
        log.debug("PDF payload 추출 실패 | error=%s", e)
        return {
            "text": "",
            "page_count": 0,
            "parsed_page_count": 0,
            "image_count": 0,
            "contains_images": False,
            "pages": [],
            "tables": [],
            "table_count": 0,
            "contains_tables": False,
            "pdf_parse_strategy": "failed",
            "table_parse_strategy": "not_parsed",
            "chart_parse_strategy": "not_parsed",
        }


def _clean_pdf_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
    text = text.replace("\\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"(?<=\dQ\d{2})(?=\dQ\d{2})", " ", text)
    text = re.sub(r"(?<=\d)(?=Q\d{2})", " ", text)
    text = re.sub(r"([A-Za-z])(?=[가-힣])", r"\1 ", text)
    text = re.sub(r"([가-힣])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([가-힣])(?=\d)", r"\1 ", text)
    text = re.sub(r"(\d)(?=[가-힣])", r"\1 ", text)
    return text.strip()


def _extract_pdf_page_blocks(page) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    for block in page.get_text("blocks", sort=True):
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            continue

        blocks.append(
            {
                "bbox": [
                    round(float(x0), 2),
                    round(float(y0), 2),
                    round(float(x1), 2),
                    round(float(y1), 2),
                ],
                "text": text,
            }
        )

    return blocks


_TABLE_HEADER_RE = re.compile(
    r"(매출|영업이익|순이익|EBITDA|OP|QoQ|YoY|분기|연간|억원|십억원|백만원|원)"
)
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")


def _detect_table_candidates(
    page_index: int,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for block in blocks:
        text = str(block.get("text") or "")
        numbers = _NUMBER_RE.findall(text)
        has_header = bool(_TABLE_HEADER_RE.search(text))
        if len(numbers) < 2 and not (has_header and numbers):
            continue

        rows.append(
            {
                "bbox": block.get("bbox"),
                "text": text,
                "number_count": len(numbers),
            }
        )

    if len(rows) < 2:
        return []

    return [
        {
            "page": page_index,
            "strategy": "numeric_text_block_candidates",
            "row_count": len(rows),
            "rows": rows,
        }
    ]


__all__ = ["extract_pdf_payload"]
