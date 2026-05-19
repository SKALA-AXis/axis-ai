"""raw_article_metadata_market_data 안의 주가 payload를 OHLCV 테이블로 백필한다.

사용 예:
  uv run python scripts/reprocess_market_data.py
  uv run python scripts/reprocess_market_data.py --limit 20
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base import RawArticle
from src.db.article_store import (
    _ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL,
    _ENSURE_MARKET_PRICE_OHLCV_SQL,
    _UPSERT_MARKET_PRICE_OHLCV_SQL,
    _market_price_ohlcv_rows,
)
from src.db.postgres import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reprocess_market_data")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="market_data OHLCV payload를 테이블로 백필")
    parser.add_argument("--limit", type=int, default=0, help="처리할 market_data 문서 수 제한")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    articles = _load_market_data_articles(limit=args.limit)
    log.info("market_data 문서 로드 완료 | count=%d", len(articles))

    upserted = 0
    with SessionLocal() as db:
        db.execute(_ENSURE_MARKET_PRICE_OHLCV_SQL)
        db.execute(_ENSURE_MARKET_PRICE_OHLCV_INDEX_SQL)
        for article_id, article in articles:
            rows = _market_price_ohlcv_rows(article_id, article)
            for row in rows:
                db.execute(_UPSERT_MARKET_PRICE_OHLCV_SQL, row)
            upserted += len(rows)
        db.commit()

    log.info("market_price_ohlcv 백필 완료 | rows=%d", upserted)


def _load_market_data_articles(*, limit: int = 0) -> list[tuple[int, RawArticle]]:
    limit_sql = "LIMIT :limit" if limit > 0 else ""
    params = {"limit": limit} if limit > 0 else {}
    with SessionLocal() as db:
        rows = db.execute(
            text(f"""
                SELECT
                    ra.id,
                    ra.company,
                    ra.title,
                    ra.content,
                    ra.url,
                    ra.source_name,
                    ra.source_type,
                    ra.content_type,
                    ra.publisher,
                    ra.language,
                    ra.crawl_status,
                    ra.error_message,
                    ra.published_at,
                    ra.collected_at,
                    COALESCE(md.source_metadata, '{{}}'::jsonb) AS metadata
                FROM raw_articles ra
                LEFT JOIN raw_article_metadata_market_data md
                  ON md.raw_article_id = ra.id
                WHERE ra.source_type = 'market_data'
                ORDER BY ra.collected_at DESC NULLS LAST, ra.id DESC
                {limit_sql}
            """),
            params,
        ).fetchall()

    articles: list[tuple[int, RawArticle]] = []
    for row in rows:
        item = dict(row._mapping)
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        article = RawArticle(
            url=str(item.get("url") or ""),
            title=str(item.get("title") or ""),
            content=item.get("content"),
            source_name=str(item.get("source_name") or "stock_market"),
            published_at=item.get("published_at"),
            collected_at=item.get("collected_at") or datetime.now().astimezone(),
            source_type="market_data",
            content_type=item.get("content_type") or "api",
            publisher=item.get("publisher"),
            company=_company_list(item.get("company")),
            language=item.get("language") or "ko",
            crawl_status=item.get("crawl_status") or "success",
            error_message=item.get("error_message"),
            extra=metadata,
        )
        articles.append((int(item["id"]), article))
    return articles


def _company_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    return []


if __name__ == "__main__":
    main()
