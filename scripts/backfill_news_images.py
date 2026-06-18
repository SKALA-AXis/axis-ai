# 작성일: 2026-05-28
# 작성자: 박지원
# 변경이력:
#   2026-05-28 박지원 — 뉴스 대표 이미지 URL 백필 스크립트 추가
#   2026-06-12 박지원 — 카드뉴스 클러스터링/리프레시 파이프라인 수정에 따른 반영
"""raw_articles.metadata.image_urls 만 백필한다.

LLM/전처리/클러스터링은 실행하지 않는다. 기사 URL HTML을 다시 fetch 해서
대표 이미지 URL만 추출한 뒤 metadata.image_urls 에 병합한다.

사용 예:
  uv run python scripts/backfill_news_images.py --env local --days 7 --limit 500
  uv run python scripts/backfill_news_images.py --env local --days 7 --limit 500 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.env_loader import load_profile  # noqa: E402
from src.crawler.parsers.article_content import extract_image_urls  # noqa: E402
from src.crawler.sources.naver import REQUEST_HEADERS, resolve_fetch_url  # noqa: E402
from src.db.postgres import SessionLocal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("backfill_news_images")

_TARGET_SQL = text("""
    SELECT id, title, url, metadata
    FROM raw_articles
    WHERE source_type = ANY(:source_types)
      AND crawl_status = 'success'
      AND url IS NOT NULL
      AND url <> ''
      AND (
        metadata -> 'image_urls' IS NULL
        OR jsonb_typeof(metadata -> 'image_urls') <> 'array'
        OR jsonb_array_length(metadata -> 'image_urls') = 0
      )
      AND (:days <= 0 OR created_at >= now() - (:days * interval '1 day'))
      AND (:source_name IS NULL OR source_name = :source_name)
    ORDER BY created_at DESC
    LIMIT :limit
""")

_COUNT_SQL = text("""
    SELECT count(*)
    FROM raw_articles
    WHERE source_type = ANY(:source_types)
      AND crawl_status = 'success'
      AND url IS NOT NULL
      AND url <> ''
      AND (
        metadata -> 'image_urls' IS NULL
        OR jsonb_typeof(metadata -> 'image_urls') <> 'array'
        OR jsonb_array_length(metadata -> 'image_urls') = 0
      )
      AND (:days <= 0 OR created_at >= now() - (:days * interval '1 day'))
      AND (:source_name IS NULL OR source_name = :source_name)
""")

_UPDATE_SQL = text("""
    UPDATE raw_articles
    SET metadata = COALESCE(metadata, '{}'::jsonb) || CAST(:metadata_patch AS jsonb)
    WHERE id = :id
""")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="뉴스 이미지 URL metadata 백필")
    parser.add_argument(
        "--env",
        choices=["local", "cloud"],
        default=None,
        help="DB 프로파일. .env.{profile} 파일이 있으면 로드한다.",
    )
    parser.add_argument("--days", type=int, default=7, help="최근 N일 대상. 0이면 전체.")
    parser.add_argument("--limit", type=int, default=500, help="한 번에 조회할 최대 기사 수.")
    parser.add_argument(
        "--source-name",
        default=None,
        help="특정 source_name만 처리. 예: naver_news",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        dest="source_types",
        default=None,
        help="처리할 source_type. 여러 번 지정 가능. 기본: news, official",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="기사 HTML fetch timeout.")
    parser.add_argument("--concurrency", type=int, default=8, help="동시 fetch 수.")
    parser.add_argument("--apply", action="store_true", help="실제 DB에 반영.")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    profile = load_profile(args.env)
    limit = max(1, args.limit)
    days = max(0, args.days)
    concurrency = max(1, args.concurrency)

    source_types = args.source_types or ["news", "official"]
    rows, total = _load_targets(
        days=days,
        limit=limit,
        source_name=args.source_name,
        source_types=source_types,
    )
    log.info(
        (
            "뉴스 이미지 백필 대상 확인 | profile=%s days=%d source_types=%s "
            "source_name=%s total=%d loaded=%d"
        ),
        profile,
        days,
        ",".join(source_types),
        args.source_name or "*",
        total,
        len(rows),
    )

    patches = await _collect_image_patches(
        rows,
        timeout=args.timeout,
        concurrency=concurrency,
    )

    for patch in patches[:10]:
        log.info(
            "이미지 백필 샘플 | id=%d image_count=%d first=%s",
            patch["id"],
            len(patch["image_urls"]),
            patch["image_urls"][0],
        )

    if not args.apply:
        log.info(
            "dry-run 완료 | fetched=%d images_found=%d updated=0. 적용하려면 --apply 사용",
            len(rows),
            len(patches),
        )
        return

    updated = _apply_patches(patches)
    log.info(
        "뉴스 이미지 백필 완료 | fetched=%d images_found=%d updated=%d",
        len(rows),
        len(patches),
        updated,
    )


def _load_targets(
    *,
    days: int,
    limit: int,
    source_name: str | None,
    source_types: list[str],
) -> tuple[list[dict[str, Any]], int]:
    params = {
        "days": days,
        "limit": limit,
        "source_name": source_name,
        "source_types": source_types,
    }
    with SessionLocal() as db:
        total = int(db.execute(_COUNT_SQL, params).scalar() or 0)
        rows = [
            {"id": int(row.id), "title": row.title, "url": row.url, "metadata": row.metadata}
            for row in db.execute(_TARGET_SQL, params).fetchall()
        ]
    return rows, total


async def _collect_image_patches(
    rows: list[dict[str, Any]],
    *,
    timeout: float,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=REQUEST_HEADERS,
        follow_redirects=True,
    ) as client:
        tasks = [_image_patch_for_row(client, semaphore, row) for row in rows]
        results = await asyncio.gather(*tasks)
    return [result for result in results if result]


async def _image_patch_for_row(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    async with semaphore:
        article_id = int(row["id"])
        url = str(row["url"])
        try:
            resp = await client.get(resolve_fetch_url(url))
            if resp.status_code != 200:
                log.warning(
                    "이미지 fetch 실패 | id=%d status=%d url=%s",
                    article_id,
                    resp.status_code,
                    url,
                )
                return None
            image_urls = extract_image_urls(resp.text, str(resp.url))
        except Exception as exc:
            log.warning("이미지 fetch 예외 | id=%d error=%s url=%s", article_id, exc, url)
            return None

    if not image_urls:
        return None

    return {
        "id": article_id,
        "image_urls": image_urls,
        "image_backfilled_at": datetime.now(UTC).isoformat(),
    }


def _apply_patches(patches: list[dict[str, Any]]) -> int:
    if not patches:
        return 0

    with SessionLocal() as db:
        for patch in patches:
            metadata_patch = {
                "image_urls": patch["image_urls"],
                "image_backfilled_at": patch["image_backfilled_at"],
            }
            db.execute(
                _UPDATE_SQL,
                {
                    "id": patch["id"],
                    "metadata_patch": json.dumps(metadata_patch, ensure_ascii=False),
                },
            )
        db.commit()
    return len(patches)


if __name__ == "__main__":
    asyncio.run(main())
