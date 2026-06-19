# 작성일: 2026-05-08
# 작성자: 박지원
# 변경이력:
#   2026-05-08 박지원 — 크롤러 전처리 구현
"""PDF 텍스트/블록/표 후보 메타데이터 추출 유틸."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

log = logging.getLogger(__name__)

PDF_OCR_ENABLED = os.getenv("PDF_OCR_ENABLED", "1").lower() not in {"0", "false", "no"}
PDF_OCR_MIN_TEXT_CHARS = int(os.getenv("PDF_OCR_MIN_TEXT_CHARS", "80"))
PDF_OCR_LANGUAGE = os.getenv("PDF_OCR_LANGUAGE", os.getenv("IR_PDF_OCR_LANGUAGE", "kor+eng"))
PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", os.getenv("IR_PDF_OCR_DPI", "200")))
PDF_OCR_PSM = os.getenv("PDF_OCR_PSM", "6")
PDF_OCR_TIMEOUT_SECONDS = int(os.getenv("PDF_OCR_TIMEOUT_SECONDS", "30"))


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
            drawing_count = 0
            ocr_pages: list[int] = []

            for page_index, page in enumerate(doc, start=1):
                page_text = _clean_pdf_text(page.get_text("text", sort=True) or "")
                page_blocks = _extract_pdf_page_blocks(page)
                page_images = len(page.get_images(full=True))
                page_drawings = len(page.get_drawings())
                ocr_used = False

                if _needs_ocr_page(page_text, page_images, page_drawings):
                    ocr_text = _extract_pdf_page_ocr_text(page)
                    if _is_useful_ocr_text(ocr_text, page_text):
                        page_text = _merge_page_text_with_ocr(page_text, ocr_text)
                        page_blocks.extend(_ocr_text_blocks(ocr_text))
                        ocr_used = True
                        ocr_pages.append(page_index)

                page_tables = _detect_table_candidates(page_index, page_blocks)

                image_count += page_images
                drawing_count += page_drawings
                tables.extend(page_tables)
                chunks.append(f"\n[PAGE {page_index}]\n{page_text}")
                current_length += len(page_text)
                pages_payload.append(
                    {
                        "page": page_index,
                        "text": page_text,
                        "text_chars": len(page_text),
                        "image_count": page_images,
                        "drawing_count": page_drawings,
                        "ocr_used": ocr_used,
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
                "drawing_count": drawing_count,
                "contains_images": image_count > 0,
                "ocr_applied": bool(ocr_pages),
                "ocr_pages": ocr_pages,
                "pages": pages_payload,
                "tables": tables,
                "table_count": len(tables),
                "contains_tables": bool(tables),
                "pdf_parse_strategy": "text_blocks_table_candidates_with_optional_ocr",
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
            "drawing_count": 0,
            "contains_images": False,
            "ocr_applied": False,
            "ocr_pages": [],
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


def _needs_ocr_page(page_text: str, image_count: int, drawing_count: int) -> bool:
    if not PDF_OCR_ENABLED:
        return False
    if image_count <= 0 and drawing_count < 20:
        return False

    visible_chars = len(re.findall(r"[0-9A-Za-z가-힣]", page_text or ""))
    return visible_chars < PDF_OCR_MIN_TEXT_CHARS


def _extract_pdf_page_ocr_text(page) -> str:
    text = _extract_pdf_page_ocr_text_with_pymupdf(page)
    if text:
        return text
    return _extract_pdf_page_ocr_text_with_tesseract(page)


def _extract_pdf_page_ocr_text_with_pymupdf(page) -> str:
    try:
        textpage = page.get_textpage_ocr(
            flags=0,
            language=PDF_OCR_LANGUAGE,
            dpi=PDF_OCR_DPI,
            full=False,
        )
        return _normalize_ocr_text(
            _clean_pdf_text(page.get_text("text", sort=True, textpage=textpage))
        )
    except Exception as exc:
        log.debug("PyMuPDF OCR fallback 실패 | page=%s error=%s", getattr(page, "number", "?"), exc)
        return ""


def _extract_pdf_page_ocr_text_with_tesseract(page) -> str:
    if not shutil.which("tesseract"):
        return ""

    tmp_path = ""
    try:
        pix = page.get_pixmap(dpi=PDF_OCR_DPI, alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            pix.save(tmp_path)

        result = subprocess.run(
            [
                "tesseract",
                tmp_path,
                "stdout",
                "-l",
                PDF_OCR_LANGUAGE,
                "--psm",
                PDF_OCR_PSM,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PDF_OCR_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            log.debug(
                "tesseract OCR 실패 | page=%s returncode=%s stderr=%s",
                getattr(page, "number", "?"),
                result.returncode,
                result.stderr[:300],
            )
            return ""
        return _normalize_ocr_text(_clean_pdf_text(result.stdout))
    except Exception as exc:
        log.debug(
            "tesseract OCR fallback 실패 | page=%s error=%s", getattr(page, "number", "?"), exc
        )
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _is_useful_ocr_text(ocr_text: str, existing_text: str) -> bool:
    ocr_chars = len(re.findall(r"[0-9A-Za-z가-힣]", ocr_text or ""))
    existing_chars = len(re.findall(r"[0-9A-Za-z가-힣]", existing_text or ""))
    return ocr_chars >= PDF_OCR_MIN_TEXT_CHARS and ocr_chars > existing_chars


def _normalize_ocr_text(text: str) -> str:
    """OCR에서 자주 깨지는 AI/AX 관련 토큰을 보수적으로 보정한다."""
    value = text or ""
    replacements = (
        (r"\bAl\b", "AI"),
        (r"\bAl(?=Ops\b)", "AI"),
        (r"\bAl(?=\s*(?:Native|Full|Agent|Orchestrator|Data|Machine)\b)", "AI"),
        (r"\bAl(?=\s*(?:인프라|플랫폼|서비스|솔루션|클라우드|데이터|컴퓨팅))", "AI"),
        (r"\bAIOps\s*/\s*MLOps", "AIOps/MLOps"),
        (r"\b시\s*/\s*클라우드", "AI/클라우드"),
        (r"\b시\s*컴퓨팅", "AI 컴퓨팅"),
        (r"\b시\s*데이터", "AI 데이터"),
        (r"\bAx\b", "AX"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

    lines = []
    for line in value.splitlines():
        line = re.sub(r"AI\s*-\s*AI", "AI", line)
        line = re.sub(r"AX\s*-\s*AI", "AX-AI", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _merge_page_text_with_ocr(existing_text: str, ocr_text: str) -> str:
    if not existing_text:
        return ocr_text
    return f"{existing_text}\n[OCR]\n{ocr_text}"


def _ocr_text_blocks(ocr_text: str) -> list[dict[str, Any]]:
    return [
        {
            "bbox": [],
            "text": line.strip(),
            "source": "ocr",
        }
        for line in ocr_text.splitlines()
        if line.strip()
    ]


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
