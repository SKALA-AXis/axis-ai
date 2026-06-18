# 작성일: 2026-05-12
# 작성자: 박지원
# 변경이력:
#   2026-05-12 박지원 — 문서형 article 파서 라우터 작성 및 peer profile 스냅샷 파이프라인 반영
"""수집된 문서형 article을 source별 deterministic parser로 라우팅한다."""

from __future__ import annotations

from typing import Any

from src.parsers.dart_parser import DartParser
from src.parsers.ir_parser import IRParser
from src.parsers.securities_report_parser import SecuritiesReportParser


class DocumentParserRouter:
    """DART, IR, 리포트/산업문서를 공통 인터페이스로 파싱한다.

    크롤러가 이미 추출한 본문 텍스트와 메타데이터를 표준 parsed_document로 정리한다.
    PDF 파일 재다운로드나 원문 재수집은 수행하지 않는다.
    """

    def parse_article(self, article: dict[str, Any]) -> dict[str, Any]:
        source_type = _source_type(article)
        source_name = _source_name(article)

        if source_type == "dart":
            return DartParser().parse_article(article)

        if source_type == "ir":
            return IRParser().parse_article(article)

        if source_type == "securities_report":
            return SecuritiesReportParser().parse_article(article)

        if source_type == "trend_report":
            return _parse_document_article(
                article,
                parser_name=f"{source_name or 'trend_report'}_parser",
                source="industry_trend",
            )

        return _parse_document_article(article, parser_name="generic_document_parser")


def _parse_document_article(
    article: dict[str, Any],
    *,
    parser_name: str,
    source: str | None = None,
) -> dict[str, Any]:
    extra = _extra(article)
    content = str(article.get("content") or "")
    content_type = str(article.get("content_type") or "").strip().lower()
    source_type = _source_type(article)

    warnings: list[str] = []
    if not content:
        warnings.append("content 없음")

    pdf_text_chars = _int_or_none(extra.get("pdf_text_chars"))
    if pdf_text_chars is None and content_type == "pdf":
        pdf_text_chars = len(content)

    return {
        "ok": bool(content),
        "parser": parser_name,
        "source": source or source_type,
        "source_type": source_type,
        "source_name": article.get("source_name"),
        "title": article.get("title"),
        "url": article.get("url"),
        "publisher": article.get("publisher"),
        "content_type": content_type or None,
        "content_chars": len(content),
        "metadata": {
            "pdf_url": extra.get("pdf_url"),
            "pdf_path": extra.get("pdf_path"),
            "pdf_text_chars": pdf_text_chars,
            "page_count": extra.get("page_count"),
            "pdf_pages": extra.get("pdf_pages"),
            "pdf_parsed_pages": extra.get("pdf_parsed_pages"),
            "pdf_page_blocks": extra.get("pdf_page_blocks"),
            "pdf_parse_strategy": extra.get("pdf_parse_strategy"),
            "report_type": extra.get("report_type"),
            "firm": extra.get("firm"),
            "item_code": extra.get("item_code"),
            "matched_sectors": extra.get("matched_sectors"),
            "visual_images": extra.get("visual_images"),
            "visual_image_count": extra.get("visual_image_count"),
            "contains_images": extra.get("contains_images"),
            "image_count": extra.get("image_count"),
            "drawing_count": extra.get("drawing_count"),
            "ocr_applied": extra.get("ocr_applied"),
            "ocr_pages": extra.get("ocr_pages"),
            "contains_tables": extra.get("contains_tables"),
            "table_count": extra.get("table_count"),
            "tables": extra.get("tables"),
            "table_parse_strategy": extra.get("table_parse_strategy"),
            "chart_parse_strategy": extra.get("chart_parse_strategy"),
        },
        "warnings": warnings,
    }


def _extra(article: dict[str, Any]) -> dict[str, Any]:
    extra = article.get("extra") or article.get("metadata") or {}
    return extra if isinstance(extra, dict) else {}


def _source_type(article: dict[str, Any]) -> str:
    return str(article.get("source_type") or "").strip().lower()


def _source_name(article: dict[str, Any]) -> str:
    return str(article.get("source_name") or "").strip().lower()


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DocumentParserRouter"]
