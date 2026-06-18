# 작성일: 2026-06-01
# 작성자: 박지원
# 변경이력:
#   2026-06-01 박지원 — peer 프로필 스냅샷 파이프라인 추가 시 함께 작성
"""PDF 원문을 다시 받아 OCR 포함 content/pdf_page_blocks를 갱신한다.

기존 reprocess_*_analysis.py는 DB에 저장된 raw_articles.content를 재파싱한다.
이미 저장된 PDF 원문 자체가 이미지형 페이지를 놓친 경우에는 이 스크립트로
raw_articles.content와 PDF metadata를 먼저 갱신한 뒤 reprocess를 실행해야 한다.

사용 예:
  uv run python scripts/backfill_pdf_ocr_content.py --source-type ir --peer-id samsung_sds
  uv run python scripts/backfill_pdf_ocr_content.py --source-type ir --peer-id samsung_sds --apply
  uv run python scripts/backfill_pdf_ocr_content.py --source-type securities_report --limit 5
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.env_loader import load_profile

load_profile()

from src.crawler.parsers.pdf_payload import extract_pdf_payload
from src.db.postgres import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_pdf_ocr_content")

SOURCE_TYPES = ("ir", "securities_report")
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_TEXT_CHARS = 200_000


def main() -> None:
    args = _parse_args()
    source_types = SOURCE_TYPES if args.source_type == "all" else (args.source_type,)
    updated_ids: list[int] = []

    for source_type in source_types:
        articles = _load_pdf_articles(
            source_type=source_type,
            peer_id=args.peer_id,
            since_year=args.since_year,
            limit=args.limit,
            article_ids=args.article_ids,
        )
        log.info(
            "대상 PDF article 조회 완료 | source_type=%s count=%d apply=%s",
            source_type,
            len(articles),
            args.apply,
        )

        for article in articles:
            result = _backfill_article(
                article,
                apply=args.apply,
                max_text_chars=args.max_text_chars,
                timeout_seconds=args.timeout_seconds,
            )
            if result["updated"]:
                updated_ids.append(int(article["id"]))

    if updated_ids:
        print("updated_article_ids=" + ",".join(str(article_id) for article_id in updated_ids))
    else:
        print("updated_article_ids=")

    if args.apply and updated_ids:
        print("\n다음 단계 예시:")
        ir_ids = [
            str(article_id)
            for article_id in updated_ids
            if _source_type_for_article_id(article_id) == "ir"
        ]
        report_ids = [
            str(article_id)
            for article_id in updated_ids
            if _source_type_for_article_id(article_id) == "securities_report"
        ]
        if ir_ids:
            ir_article_args = " ".join(f"--article-id {article_id}" for article_id in ir_ids)
            print(
                "uv run python scripts/reprocess_ir_analysis.py "
                f"{ir_article_args} --reparse --upsert-signals --replace-signals "
                "--upsert-metrics --replace-metrics"
            )
        if report_ids:
            report_article_args = " ".join(
                f"--article-id {article_id}" for article_id in report_ids
            )
            print(
                "uv run python scripts/reprocess_securities_report_analysis.py "
                f"{report_article_args} --reparse --upsert-signals --replace-signals "
                "--upsert-metrics --replace-metrics"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-type",
        choices=(*SOURCE_TYPES, "all"),
        default="ir",
        help="OCR backfill 대상 source_type",
    )
    parser.add_argument("--peer-id", help="특정 peer id만 처리")
    parser.add_argument("--since-year", type=int, help="published_at 연도 필터")
    parser.add_argument("--limit", type=int, default=0, help="처리 개수 제한")
    parser.add_argument(
        "--article-ids",
        type=_parse_article_ids,
        default=[],
        help="쉼표로 구분한 raw_articles.id 목록",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="PDF 본문 저장 최대 길이",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="PDF 다운로드 timeout",
    )
    parser.add_argument("--apply", action="store_true", help="실제 DB 업데이트")
    return parser.parse_args()


def _parse_article_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_pdf_articles(
    *,
    source_type: str,
    peer_id: str | None,
    since_year: int | None,
    limit: int,
    article_ids: list[int],
) -> list[dict[str, Any]]:
    metadata_join, metadata_expr = _metadata_join_and_expr(source_type)
    filters = ["ra.source_type = :source_type"]
    params: dict[str, Any] = {"source_type": source_type}

    if article_ids:
        filters.append("ra.id = ANY(:article_ids)")
        params["article_ids"] = article_ids
    if peer_id:
        filters.append("ra.company::text ILIKE :peer_id_like")
        params["peer_id_like"] = f"%{peer_id}%"
    if since_year:
        filters.append("ra.published_at >= make_timestamptz(:since_year, 1, 1, 0, 0, 0)")
        params["since_year"] = since_year
    if limit > 0:
        params["limit"] = limit

    limit_sql = "LIMIT :limit" if limit > 0 else ""
    where_sql = " AND ".join(filters)

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
                    {metadata_expr} AS metadata
                FROM raw_articles ra
                {metadata_join}
                WHERE {where_sql}
                ORDER BY ra.published_at DESC NULLS LAST, ra.id DESC
                {limit_sql}
            """),
            params,
        ).fetchall()

    return [dict(row._mapping) for row in rows]


def _metadata_join_and_expr(source_type: str) -> tuple[str, str]:
    with SessionLocal() as db:
        has_unified = _table_exists(db, "raw_article_metadata_unified")
        has_source_metadata = _table_exists(db, "raw_article_source_metadata")
        legacy_table = (
            "raw_article_metadata_ir"
            if source_type == "ir"
            else "raw_article_metadata_securities_report"
        )
        has_legacy = _table_exists(db, legacy_table)

    if has_unified:
        return (
            """
                LEFT JOIN raw_article_metadata_unified md
                  ON md.raw_article_id = ra.id
            """,
            "COALESCE(md.metadata, md.source_metadata, ra.metadata, '{}'::jsonb)",
        )
    if has_source_metadata:
        return (
            """
                LEFT JOIN raw_article_source_metadata md
                  ON md.raw_article_id = ra.id
                 AND md.source_type = ra.source_type
            """,
            "COALESCE(md.source_metadata, ra.metadata, '{}'::jsonb)",
        )
    if has_legacy:
        return (
            f"""
                LEFT JOIN {legacy_table} md
                  ON md.raw_article_id = ra.id
            """,
            "COALESCE(md.source_metadata, ra.metadata, '{}'::jsonb)",
        )
    return "", "COALESCE(ra.metadata, '{}'::jsonb)"


def _backfill_article(
    article: dict[str, Any],
    *,
    apply: bool,
    max_text_chars: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    metadata = _as_dict(article.get("metadata"))
    pdf_url = _pdf_url(article, metadata)
    if not pdf_url:
        log.warning("PDF URL 없음 | id=%s title=%s", article["id"], article.get("title"))
        return {"updated": False, "reason": "missing_pdf_url"}

    try:
        pdf_bytes = _download_pdf(pdf_url, timeout_seconds=timeout_seconds)
        payload = extract_pdf_payload(pdf_bytes, max_text_chars=max_text_chars)
    except Exception as exc:
        log.warning("PDF OCR backfill 실패 | id=%s url=%s error=%s", article["id"], pdf_url, exc)
        return {"updated": False, "reason": str(exc)}

    new_text = str(payload.get("text") or "")
    old_text = str(article.get("content") or "")
    should_update = bool(new_text) and (
        len(new_text) > len(old_text) or bool(payload.get("ocr_applied"))
    )

    log.info(
        (
            "PDF OCR backfill 확인 | id=%s source=%s old_chars=%d new_chars=%d "
            "ocr=%s pages=%s update=%s title=%s"
        ),
        article["id"],
        article.get("source_type"),
        len(old_text),
        len(new_text),
        payload.get("ocr_applied"),
        payload.get("ocr_pages"),
        should_update and apply,
        article.get("title"),
    )

    if apply and should_update:
        patch = _metadata_patch(payload, pdf_url=pdf_url)
        _update_article_pdf_payload(
            article_id=int(article["id"]),
            source_type=str(article["source_type"]),
            content=new_text,
            metadata_patch=patch,
        )

    return {"updated": apply and should_update, "reason": "ok"}


def _pdf_url(article: dict[str, Any], metadata: dict[str, Any]) -> str:
    url = str(metadata.get("pdf_url") or "").strip()
    if url:
        return url
    article_url = str(article.get("url") or "").strip()
    if article_url.lower().endswith(".pdf") or "stock-research" in article_url:
        return article_url
    if str(article.get("content_type") or "").lower() == "pdf":
        return article_url
    return ""


def _download_pdf(pdf_url: str, *, timeout_seconds: int) -> bytes:
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 AXIS-Crawler/1.0"},
        )
        response.raise_for_status()
        return response.content


def _metadata_patch(payload: dict[str, Any], *, pdf_url: str) -> dict[str, Any]:
    return {
        "pdf_url": pdf_url,
        "pdf_text_chars": len(str(payload.get("text") or "")),
        "pdf_pages": payload.get("page_count"),
        "pdf_parsed_pages": payload.get("parsed_page_count"),
        "pdf_page_blocks": payload.get("pages"),
        "pdf_parse_strategy": payload.get("pdf_parse_strategy"),
        "contains_images": payload.get("contains_images"),
        "image_count": payload.get("image_count"),
        "drawing_count": payload.get("drawing_count"),
        "ocr_applied": payload.get("ocr_applied"),
        "ocr_pages": payload.get("ocr_pages"),
        "contains_tables": payload.get("contains_tables"),
        "table_count": payload.get("table_count"),
        "tables": payload.get("tables"),
        "table_parse_strategy": payload.get("table_parse_strategy"),
        "chart_parse_strategy": payload.get("chart_parse_strategy"),
        "pdf_ocr_backfilled_at": datetime.now(timezone.utc).isoformat(),
    }


def _update_article_pdf_payload(
    *,
    article_id: int,
    source_type: str,
    content: str,
    metadata_patch: dict[str, Any],
) -> None:
    patch_json = json.dumps(_sanitize_jsonish(metadata_patch), ensure_ascii=False)

    with SessionLocal() as db:
        db.execute(
            text("""
                UPDATE raw_articles
                SET content = :content,
                    content_type = 'pdf',
                    metadata = COALESCE(metadata, '{}'::jsonb) || CAST(:metadata_patch AS jsonb),
                    collected_at = COALESCE(collected_at, NOW())
                WHERE id = :article_id
            """),
            {
                "article_id": article_id,
                "content": content,
                "metadata_patch": patch_json,
            },
        )

        if _table_exists(db, "raw_article_source_metadata"):
            db.execute(
                text("""
                    INSERT INTO raw_article_source_metadata (
                        raw_article_id, source_type, source_metadata
                    ) VALUES (
                        :article_id, :source_type, CAST(:metadata_patch AS jsonb)
                    )
                    ON CONFLICT (raw_article_id, source_type) DO UPDATE SET
                        source_metadata =
                          COALESCE(raw_article_source_metadata.source_metadata, '{}'::jsonb)
                          || EXCLUDED.source_metadata
                """),
                {
                    "article_id": article_id,
                    "source_type": source_type,
                    "metadata_patch": patch_json,
                },
            )

        legacy_table = (
            "raw_article_metadata_ir"
            if source_type == "ir"
            else "raw_article_metadata_securities_report"
        )
        if _table_exists(db, legacy_table):
            db.execute(
                text(f"""
                    UPDATE {legacy_table}
                    SET source_metadata = COALESCE(source_metadata, '{{}}'::jsonb)
                      || CAST(:metadata_patch AS jsonb)
                    WHERE raw_article_id = :article_id
                """),
                {"article_id": article_id, "metadata_patch": patch_json},
            )

        db.commit()


def _table_exists(db: Any, table_name: str) -> bool:
    return bool(
        db.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": f"public.{table_name}"},
        ).scalar_one_or_none()
    )


def _source_type_for_article_id(article_id: int) -> str:
    with SessionLocal() as db:
        value = db.execute(
            text("SELECT source_type FROM raw_articles WHERE id = :article_id"),
            {"article_id": article_id},
        ).scalar_one_or_none()
    return str(value or "")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sanitize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_jsonish(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonish(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
